# Install necessary libraries
# !pip install transformers datasets wandb

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, interleave_datasets
import wandb
from itertools import islice, cycle
import argparse

# RTX 3050 optimizations with memory constraints
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Minimize HuggingFace caching to save storage
import os
os.environ['HF_DATASETS_CACHE'] = './hf_cache_minimal'  # Small local cache only
os.environ['HF_DATASETS_OFFLINE'] = '0'  # Allow downloads but minimize cache
os.environ['TRANSFORMERS_CACHE'] = './hf_cache_minimal'

parser = argparse.ArgumentParser(description='Training script for 17M Samantha model optimized for RTX 3050.')
parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')
args = parser.parse_args()

# Initialize wandb only when not resuming (to avoid multiple initializations)
if not args.resume:
    run = wandb.init(project='samantha-17m', name='samantha-17m-rtx3050-training')
    run_id = wandb.run.id
    os.makedirs('./emergency_checkpoint_17m', exist_ok=True)
    with open('./emergency_checkpoint_17m/run_id.txt', 'w') as f:
        f.write(run_id)
else:
    if os.path.exists('./emergency_checkpoint_17m/run_id.txt'):
        with open('./emergency_checkpoint_17m/run_id.txt', 'r') as f:
            run_id = f.read().strip()
        run = wandb.init(project='samantha-17m',
                         id=run_id,
                         resume="allow")
        print(f"Resuming WandB run with id {run_id}")
    else:
        run = wandb.init(project='samantha-17m', name='samantha-17m-rtx3050-training')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

global block_size
block_size = 256  # Reduced for RTX 3050 memory constraints; original was 512

# List of dataset names
dataset_names = [
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-18",
    "CC-MAIN-2023-50",
]

def load_datasets(dataset_names, split, streaming=True):
    datasets_list = []
    for name in dataset_names:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=name,
            split=split,
            streaming=streaming
        )
        datasets_list.append(dataset)
    return datasets_list

# Load datasets with true streaming (no caching)
training_datasets = load_datasets(dataset_names, split="train", streaming=True)
training_dataset = interleave_datasets(training_datasets)

# Function to tokenize on-the-fly (no .map() caching)
def tokenize_and_chunk_stream(dataset_iter, chunk_size=10):  # Reduced chunk size for memory
    """Process data in small chunks to minimize memory usage"""
    buffer_input_ids = []
    buffer_attention_mask = []

    for example in dataset_iter:
        # Tokenize example
        tokenized = tokenizer(example['text'], truncation=True, max_length=block_size,
                            return_tensors='pt', padding=False)

        # Add to buffers
        buffer_input_ids.append(tokenized['input_ids'].squeeze().tolist())
        buffer_attention_mask.append(tokenized['attention_mask'].squeeze().tolist())

        # Yield when buffer is full
        if len(buffer_input_ids) >= chunk_size:
            yield {
                'input_ids': buffer_input_ids,
                'attention_mask': buffer_attention_mask
            }
            buffer_input_ids = []
            buffer_attention_mask = []

    # Yield remaining items
    if buffer_input_ids:
        yield {
            'input_ids': buffer_input_ids,
            'attention_mask': buffer_attention_mask
        }

# group texts into blocks
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated['input_ids'])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    result['labels'] = result['input_ids'].copy()
    return result

# Create streaming iterator
stream_iter = tokenize_and_chunk_stream(training_dataset, chunk_size=10)
grouped_train_dataset = (group_texts(batch) for batch in stream_iter)

# Define a custom IterableDataset
class StreamDataset(IterableDataset):
    def __init__(self, grouped_dataset):
        self.grouped_dataset = grouped_dataset

    def __iter__(self):
        for batch in self.grouped_dataset:
            for i in range(len(batch['input_ids'])):
                yield {
                    'input_ids': torch.tensor(batch['input_ids'][i], dtype=torch.long),
                    'labels': torch.tensor(batch['labels'][i], dtype=torch.long),
                }

# training dataset and dataloader (optimized for RTX 3050 with storage constraints)
train_dataset = StreamDataset(grouped_train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=0)  # Further reduced for memory

# Create validation dataset by taking a finite sample from the training data
def create_validation_dataset(train_dataset, num_batches):
    validation_data = []
    train_iter = iter(train_dataset)
    for _ in range(num_batches):
        try:
            data = next(train_iter)
            validation_data.append(data)
        except StopIteration:
            break
    return validation_data

# Number of validation batches to sample
num_validation_batches = 100

# validation dataset (optimized for RTX 3050)
validation_data = create_validation_dataset(train_dataset, num_validation_batches)
validation_dataloader = DataLoader(validation_data, batch_size=4, num_workers=0)  # Reduced batch size for 6GB VRAM

# Model with Pre-LayerNorm and Biases aka GPT-3 (Samantha is based on GPT-2)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
    CausalLMOutputWithCrossAttentions,
)

class CustomGPT2Config(GPT2Config):
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config, is_cross_attention)
        # Ensure biases are included
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

class CustomGPT2MLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=True)
        self.act = nn.GELU()  # Use standard GeLU

class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        self.use_pre_layernorm = config.use_pre_layernorm
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        self.mlp = CustomGPT2MLP(4 * config.n_embd, config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if self.use_pre_layernorm:
            # Pre-LayerNorm
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]  # present, (attentions)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
        else:
            # Original GPT-2 Post-LayerNorm
            residual = hidden_states
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]  # present, (attentions)

            hidden_states = residual + attn_output
            hidden_states = self.ln_1(hidden_states)

            residual = hidden_states
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
            hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)

class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

# Configuration for 17M model (optimized for RTX 3050)
config = CustomGPT2Config(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512,   # Reduced context length for memory efficiency
    n_ctx=512,                     # Reduced context length for memory efficiency
    hidden_size=256,               # Embedding dimension (optimized for 17M parameters)
    num_hidden_layers=6,           # Number of transformer blocks
    num_attention_heads=8,         # Number of attention heads
    intermediate_size=1024,        # Feedforward dimension (4x hidden_size)
    activation_function='gelu',
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_pre_layernorm=True,
)

# Initialize the model from scratch
model = CustomGPT2LMHeadModel(config)

# Custom Weight Initialization
def custom_init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Standard initialization
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
        # Apply scaling to residual projections
        module.weight.data.mul_(1 / math.sqrt(2 * config.num_hidden_layers))

model.apply(custom_init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model optimizations for RTX 3050
# Disable torch.compile on Windows to avoid C++ compiler issues
if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode='max-autotune')  # Compile for better performance
        print("Model compiled successfully with torch.compile")
    except Exception as e:
        print(f"torch.compile failed: {e}. Using regular model.")
else:
    print("CUDA not available, skipping torch.compile")

# model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {total_params:,} parameters")
print(f"Memory for parameters (BF16): {total_params * 2 / 1024 / 1024:.2f} MB")
print(f"Estimated training memory: ~{total_params * 2 * 3 / 1024 / 1024:.2f} MB (params + grads + optimizer)")
print(f"Using device: {device}")
s
# Force CUDA if available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    model.to(device)
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")
    model.to(device)

# optimizer and learning rate scheduler
total_steps = 600000  # desired total training steps
warmup_steps = 60000  # Typically 10% of total_steps

# optimizer hyperparameters
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

gradient_accumulation_steps = 8  # Increased to compensate for smaller batch size (4 -> effective batch of 32)

##################
# RTX 3050 Memory Optimization:
# 256 tokens per batch * 4 batches = 1024 tokens per batch
# 1024 tokens per batch * 8 gradient accumulation steps = 8192 tokens per batch
# 8192 tokens per batch * 600k steps = ~4.9 billion tokens
##################

max_grad_norm = 1.0

# Resume training if --resume flag is used
if args.resume:
    checkpoint_dir = './emergency_checkpoint_17m'
    if os.path.exists(checkpoint_dir):
        print("Loading checkpoint from the emergency directory...")
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model_state.pt')))
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer_state.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scheduler_state.pt')))
        with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'r') as f:
            global_step = int(f.readline().strip())
        print(f"Resumed training from step {global_step}")
    else:
        print("No checkpoint found. Starting from scratch.")

# Training loop with gradient accumulation, gradient clipping, and perplexity computation
model.train()
logging_steps = 100   # Reduced logging frequency to save resources
save_steps = 10000    # Reduced checkpoint frequency to save storage
eval_steps = 2000     # Reduced evaluation frequency
emergency_save_steps = 500  # Reduced emergency checkpoint frequency

# Use cycle to create an infinite iterator over the dataloader
train_iterator = cycle(train_dataloader)

try:
    while True:
        optimizer.zero_grad()
        accumulated_loss = 0.0  # Accumulate loss over gradient_accumulation_steps

        for accumulation_step in range(gradient_accumulation_steps):
            batch = next(train_iterator)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            accumulated_loss += loss.item()
            loss = loss / gradient_accumulation_steps  # Normalize loss
            loss.backward()

        # Apply Gradient Clipping Here
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        global_step += 1  # Increment global_step after each optimizer step

        # Emergency save (every emergency_save_steps)
        if global_step % emergency_save_steps == 0:
            checkpoint_dir = './emergency_checkpoint_17m'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                f.write(f"{global_step}\n")
            print(f"Temporary checkpoint saved at step {global_step}.")

        # Logging
        if global_step % logging_steps == 0:
            avg_loss = accumulated_loss / gradient_accumulation_steps
            perplexity = math.exp(avg_loss) if avg_loss < 7 else float('inf')  # Adjust threshold as needed
            print(f"Step {global_step}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Grad Norm: {grad_norm:.4f}")
            # Log train metrics
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/grad_norm': grad_norm,
                'train/lr': current_lr,
                'train/step': global_step,
            })

        # Validation
        if global_step % eval_steps == 0 or global_step == 1:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in validation_dataloader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**inputs)
                    loss = outputs.loss
                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps
            val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 8 else float('inf')

            print(f"Validation at step {global_step}: Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            # Log validation metrics
            wandb.log({
                'validation/loss': avg_val_loss,
                'validation/perplexity': val_perplexity,
                'validation/step': global_step,
            })

            model.train()

        # Saving the model
        if global_step > 0 and global_step % save_steps == 0:
            save_path = f'./samantha-17m-gpt2-rtx3050-step-{global_step}'
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at step {global_step} to {save_path}")

            # Auto-cleanup old checkpoints to save storage
            try:
                import subprocess
                subprocess.run(['python', 'cleanup_checkpoints.py'], capture_output=True)
            except:
                pass  # Silently continue if cleanup fails

        if global_step >= total_steps:
            break
except KeyboardInterrupt:
    print("Training interrupted by user. Saving model...")
    save_path = f'./samantha-17m-gpt2-rtx3050-step-{global_step}'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved at step {global_step} to {save_path}")

    # Save emergency checkpoint
    checkpoint_dir = './emergency_checkpoint_17m'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
    with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
        f.write(f"{global_step}\n")
    print(f"Emergency checkpoint saved at step {global_step}.")

final_save_path = './samantha-17m-gpt2-rtx3050-final'
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final model saved to {final_save_path}")

try:
    wandb.finish()
except:
    pass
