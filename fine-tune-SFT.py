import os
import math
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb
from itertools import islice, cycle
import evaluate
from lion_pytorch import Lion
import argparse

# Multi-GPU and H100 optimizations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # Use both H100 GPUs
os.environ["NCCL_IB_DISABLE"]="1"  # Disable InfiniBand for stability
os.environ["NCCL_SOCKET_IFNAME"]="eth0"  # Specify network interface

# H100 specific optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on H100
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN

#os.environ["WANDB_OFFLINE"]="1" # Uncomment if you want to run offline

# ---------------------------
# Parse command line arguments
# ---------------------------
parser = argparse.ArgumentParser(description='Fine-tune Samantha model with combined datasets')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
parser.add_argument('--world_size', type=int, default=1, help='World size for distributed training')
parser.add_argument('--batch_size', type=int, default=4, help='Per-device batch size (reduced for 40GB H100)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps (increased for memory efficiency)')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
args = parser.parse_args()

# Detect available GPUs
num_gpus = torch.cuda.device_count()
print(f"🔍 Detected {num_gpus} GPU(s)")
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB VRAM)")

# ---------------------------
# Initialize Weights & Biases (W&B / Wandb)
# ---------------------------
run_name = f'sft-triple-h100-{num_gpus}gpu-bs{args.batch_size*args.gradient_accumulation_steps*num_gpus}'
wandb.init(project='samantha-sft', name=run_name)

# ---------------------------
# Model Classes
# ---------------------------

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
    GPT2Model,
    CausalLMOutputWithCrossAttentions,
)
import torch.nn as nn

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config, is_cross_attention)
        # biases are included
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
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,  # Capture all additional keyword arguments
):
    # we don't need to pass num_items_in_batch to the transformer
    kwargs.pop("num_items_in_batch", None)

    transformer_outputs = self.transformer(
        input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,  # Pass additional arguments (без num_items_in_batch)
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift the logits and labels for causal language modeling
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )


def custom_collate(examples):
    # get maximum length of input_ids and use it for padding
    max_length = max(len(ex["input_ids"]) for ex in examples)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for ex in examples:
        seq_len = len(ex["input_ids"])
        pad_length = max_length - seq_len
        
        # now we can pad the input_ids, attention_mask and labels
        padded_ids = ex["input_ids"] + [tokenizer.pad_token_id] * pad_length
        # attention_mask is 1 for real tokens and 0 for padding
        padded_mask = ex["attention_mask"] + [0] * pad_length
        # for labels, we use -100 for padding and the rest of the labels as is
        # Note: -100 is the ignore_index for CrossEntropyLoss
        padded_labels = ex["labels"] + [-100] * pad_length
        
        input_ids.append(padded_ids)
        attention_masks.append(padded_mask)
        labels.append(padded_labels)
        
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }



# ---------------------------
# Initialize the Tokenizer
# ---------------------------
tokenizer = GPT2TokenizerFast.from_pretrained('gpt3-small-fineweb')  # Pre-trained model
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
#  Pre-trained Model
# ---------------------------
config = GPT2Config.from_pretrained('gpt3-small-fineweb')

# Initialize the custom model with the loaded configuration
# Load model
model = CustomGPT2LMHeadModel.from_pretrained('gpt3-small-fineweb', config=config)
print(f"🎯 Fine-tuning model with {model.num_parameters():,} parameters")

# Setup device for multi-GPU training
if num_gpus > 1:
    print(f"🚀 Using {num_gpus} GPUs for distributed training")
    # Use HuggingFace's device mapping for multi-GPU
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    model.to(device)

    # Enable gradient checkpointing for memory efficiency on H100
    model.gradient_checkpointing_enable()

    # Additional memory optimizations for 40GB H100
    torch.cuda.empty_cache()  # Clear any existing cache

    print(f"📊 Effective batch size: {args.batch_size * args.gradient_accumulation_steps * num_gpus}")
    print(f"💾 Memory optimizations enabled for 40GB H100 GPUs")
    print(f"   • Gradient checkpointing: Enabled")
    print(f"   • TF32 precision: Enabled")
    print(f"   • Memory pinning: Enabled")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"🔸 Using single GPU: {device}")

# ---------------------------
# Load and Prepare the Custom ShareGPT Dataset
# ---------------------------
import json

# Load custom dataset
def load_custom_dataset(file_path):
    """Load the custom ShareGPT dataset from JSONL file"""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                conversations.append(data)
    return conversations

# Convert custom format to standard format
def convert_custom_format_to_messages(conversations_data):
    """Convert custom ShareGPT format to standard messages format"""
    formatted_data = []

    for item in conversations_data:
        messages = []
        for conv in item["conversations"]:
            role = "user" if conv["from"] == "human" else "assistant"
            messages.append({
                "role": role,
                "content": conv["value"]
            })

        formatted_data.append({"messages": messages})

    return formatted_data

# Load and convert custom datasets
custom_dataset_path_1 = "merged_dataset_project_human_sharegpt (2).jsonl"
custom_dataset_path_2 = "train.jsonl"
print(f"Loading custom datasets:")
print(f"  1. {custom_dataset_path_1}")
print(f"  2. {custom_dataset_path_2}")

# Load all datasets
all_train_datasets = []
all_eval_datasets = []

# Load first custom ShareGPT dataset
try:
    custom_conversations_1 = load_custom_dataset(custom_dataset_path_1)
    custom_formatted_data_1 = convert_custom_format_to_messages(custom_conversations_1)

    # Create dataset from formatted data
    from datasets import Dataset
    custom_dataset_1 = Dataset.from_list(custom_formatted_data_1)

    # Split into train and eval (90/10 split)
    custom_dataset_split_1 = custom_dataset_1.train_test_split(test_size=0.1, seed=42)
    all_train_datasets.append(custom_dataset_split_1["train"])
    all_eval_datasets.append(custom_dataset_split_1["test"])

    print(f"✅ ShareGPT dataset loaded: {len(custom_dataset_split_1['train'])} training, {len(custom_dataset_split_1['test'])} eval samples")

except FileNotFoundError:
    print(f"❌ First custom dataset file not found: {custom_dataset_path_1}")
    print("⚠️  Skipping first custom dataset")

# Load second custom dataset (text revision tasks)
try:
    custom_conversations_2 = load_custom_dataset(custom_dataset_path_2)
    custom_formatted_data_2 = convert_custom_format_to_messages(custom_conversations_2)

    custom_dataset_2 = Dataset.from_list(custom_formatted_data_2)

    # Split into train and eval (90/10 split)
    custom_dataset_split_2 = custom_dataset_2.train_test_split(test_size=0.1, seed=42)
    all_train_datasets.append(custom_dataset_split_2["train"])
    all_eval_datasets.append(custom_dataset_split_2["test"])

    print(f"✅ Text Revision dataset loaded: {len(custom_dataset_split_2['train'])} training, {len(custom_dataset_split_2['test'])} eval samples")

except FileNotFoundError:
    print(f"❌ Second custom dataset file not found: {custom_dataset_path_2}")
    print("⚠️  Skipping second custom dataset")

# Load no_robots dataset
try:
    no_robots_dataset = load_dataset("HuggingFaceH4/no_robots")
    no_robots_split = no_robots_dataset.train_test_split(test_size=0.1, seed=42)
    all_train_datasets.append(no_robots_split["train"])
    all_eval_datasets.append(no_robots_split["test"])

    print(f"✅ No Robots dataset loaded: {len(no_robots_split['train'])} training, {len(no_robots_split['test'])} eval samples")

except Exception as e:
    print(f"❌ Failed to load no_robots dataset: {e}")

# Combine datasets if we have multiple
if len(all_train_datasets) > 1:
    from datasets import concatenate_datasets

    print("🔄 Combining datasets...")
    combined_train = concatenate_datasets(all_train_datasets)
    combined_eval = concatenate_datasets(all_eval_datasets)

    # Shuffle the combined dataset
    combined_train = combined_train.shuffle(seed=42)
    combined_eval = combined_eval.shuffle(seed=42)

    train_dataset = combined_train
    eval_dataset = combined_eval

    dataset_names = []
    if len(all_train_datasets) >= 1:
        dataset_names.append("ShareGPT Conversations")
    if len(all_train_datasets) >= 2:
        dataset_names.append("Text Revision Tasks")
    if len(all_train_datasets) >= 3:
        dataset_names.append("No Robots Instructions")

    print(f"✅ Combined dataset created:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Evaluation samples: {len(eval_dataset)}")
    print(f"   Datasets combined: {' + '.join(dataset_names)}")

elif len(all_train_datasets) == 1:
    train_dataset = all_train_datasets[0]
    eval_dataset = all_eval_datasets[0]

    # Determine which dataset it is
    if len(all_train_datasets) == 1:
        if 'custom_dataset_split_1' in locals():
            dataset_name = "ShareGPT Conversations"
        elif 'custom_dataset_split_2' in locals():
            dataset_name = "Text Revision Tasks"
        else:
            dataset_name = "No Robots Instructions"

    print(f"✅ Using single dataset ({dataset_name}): {len(train_dataset)} training, {len(eval_dataset)} eval samples")

else:
    raise ValueError("❌ No datasets could be loaded!")

# ---------------------------
# Dataset processing is now handled above in the combined dataset section
# ---------------------------

# Preprocess datasets for SFT
def preprocess_dataset(sample, idx):
    """
        Formats an example for SFT with dynamic sequence length.
        Compatible with both custom and no_robots dataset formats.
    """
    max_length = 1024  # can be 2048 for other datasets but 1024 works fine

    # Handle different message formats
    if "messages" in sample:
        messages = sample["messages"]
    elif "conversations" in sample:
        # Convert custom format to standard format
        messages = []
        for conv in sample["conversations"]:
            role = "user" if conv["from"] == "human" else "assistant"
            messages.append({"role": role, "content": conv["value"]})
    else:
        # Fallback for other formats
        messages = [{"role": "user", "content": str(sample)}]

    conversation = ""

    # Build the complete conversation first
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        conversation += f"{role}: {content}{tokenizer.eos_token}\n"

    if idx < 2:
        if len(messages) >= 2:
            user_message = messages[0]["content"]
            assistant_message = messages[1]["content"]
            pair = f"User: {user_message}{tokenizer.eos_token}\nAssistant: {assistant_message}{tokenizer.eos_token}"
            print("\n=== Processed sample ===")
            print(pair)
            print("=== End of example ===\n")
        else:
            print("\nNo complete user-assistant pair found in the sample.\n")
    
    tokenized = tokenizer(
        conversation,
        truncation=False,
        padding=False,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    labels = [-100] * len(input_ids)
    
    # Define the ranges of tokens for assistant messages
    assistant_indices = []
    cumulative_length = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        role_tokens = tokenizer(f"{role}: ", add_special_tokens=False)["input_ids"]
        content_tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        total_tokens = len(role_tokens) + len(content_tokens) + 1  # +1 for eos_token
        if role.lower() == "assistant":
            assistant_indices.append((cumulative_length, cumulative_length + len(role_tokens) + len(content_tokens)))
        cumulative_length += total_tokens

    for start, end in assistant_indices:
        if start < len(input_ids):
            end = min(end, len(input_ids))
            labels[start:end] = input_ids[start:end]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Apply preprocessing to the datasets
print("🔄 Applying preprocessing to datasets...")

processed_train = train_dataset.map(
    preprocess_dataset,
    remove_columns=train_dataset.column_names,
    with_indices=True,
)

processed_eval = eval_dataset.map(
    preprocess_dataset,
    remove_columns=eval_dataset.column_names,
    with_indices=True,
)

print(f"✅ Preprocessing complete!")
print(f"   Training samples: {len(processed_train)}")
print(f"   Evaluation samples: {len(processed_eval)}")

# Determine dataset composition for logging
if len(all_train_datasets) > 1:
    dataset_names = []
    if len(all_train_datasets) >= 1:
        dataset_names.append("ShareGPT")
    if len(all_train_datasets) >= 2:
        dataset_names.append("Text Revision")
    if len(all_train_datasets) >= 3:
        dataset_names.append("No Robots")
    dataset_info = f"Combined ({' + '.join(dataset_names)})"
else:
    dataset_info = f"Single ({dataset_name})"

print(f"   Dataset composition: {dataset_info}")

# ---------------------------
# Define the compute_metrics Function
# ---------------------------
"""def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Move tensors to CPU for computation
    predictions = torch.tensor(logits).argmax(dim=-1)
    labels = torch.tensor(labels)

    # Compute Cross-Entropy Loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    # Flatten the tensors
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))
    perplexity = math.exp(loss.item())
    return {"eval_perplexity": perplexity}
"""

# ---------------------------
# Define Training Arguments for SFT
# ---------------------------
training_args = SFTConfig(
    output_dir='./outputs-sft-helpsteer+no_robots',
    save_steps=200,
    overwrite_output_dir=True,
    num_train_epochs=3,#6                   # works fine with 3
    per_device_train_batch_size=args.batch_size,       # Conservative for 40GB H100
    per_device_eval_batch_size=args.batch_size,        # Conservative for 40GB H100
    gradient_accumulation_steps=args.gradient_accumulation_steps,        # Increased for memory efficiency
    eval_strategy="steps",
    bf16=True,                           # H100 optimized mixed precision
    dataloader_pin_memory=True,          # Faster data transfer to GPU
    dataloader_num_workers=4,            # Multi-worker data loading
    dataloader_prefetch_factor=2,        # Prefetch optimization
    lr_scheduler_type="cosine", ###
    optim="adamw_hf", #adamw_torch
    remove_unused_columns=True,
    #deepspeed=True, ###
    eval_on_start=True,
    eval_steps=50,                      # Evaluate every 500 steps
    #save_steps=200,                     # Save model every 1000 steps
    logging_steps=1,                   # Log metrics every 100 steps
    learning_rate=args.learning_rate,                 # H100 optimized learning rate
    warmup_steps=200,                    # Warmup steps for scheduler
    save_total_limit=3,                  # Limit the total number of saved models
    report_to=['wandb'],                 # Enable logging to W&B
    run_name=run_name,  # Dynamic run name based on hardware
    load_best_model_at_end=True,         # Load the best model when finished training
    metric_for_best_model='eval_loss',   # eval_perplexity
    greater_is_better=False,             # Lower perplexity is better
)

# ---------------------------
# Initialize the Data Collator
# ---------------------------
#data_collator = CustomDataCollatorForLanguageModeling(
#    tokenizer=tokenizer,
    #mlm=False,  # Set to False for causal language modeling
#) # delete this later, we don't need it

# ---------------------------
# Define a Callback to Generate and Log Sample Outputs
# ---------------------------
class GenerateTextCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Retrieve the model and tokenizer from kwargs or fall back
        model = kwargs["model"]
        tok = kwargs.get("tokenizer") or tokenizer

        if state.global_step % 20 == 0:
            # generate some text with instructions in the same format as in dataset
            # and log to wandb to see how the model performs during training
            prompt = f"User:Write a thank you email{tok.eos_token}\nAssistant:" # this is a prompt structure, use same during inference
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            tok.pad_token = tok.eos_token
            try:
                # Enable Beam Search to make early_stopping relevant
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=256,
                    num_return_sequences=1,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    eos_token_id=tok.eos_token_id,
                    repetition_penalty=1.2,  # Optional but works well
                )
                text = tok.decode(outputs[0], skip_special_tokens=False)
                wandb.log({"sample_text": wandb.Html(f"<p>{text}</p>"), "step": state.global_step})
            except Exception as e:
                wandb.log({"sample_text": f"Error during generation: {str(e)}", "step": state.global_step})


print("Sample processed training example:")
print(processed_train[0])
# ---------------------------
# Initialize the Trainer
# ---------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    tokenizer=tokenizer,
    data_collator=custom_collate, #data_collator,
    #compute_metrics=compute_metrics,
)

trainer.add_callback(GenerateTextCallback)

# ---------------------------
# Start Training
# ---------------------------
trainer.train()

# ---------------------------
# Save the Final Model and Tokenizer
# ---------------------------
final_save_path = f'samantha-sft-h100-{num_gpus}gpu-triple'
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final model saved to {final_save_path}")

# ---------------------------
# Finish the W&B Run
# ---------------------------
wandb.finish()
gc.collect()
import gc
gc.collect()