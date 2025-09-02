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
import time
import argparse
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from lion_pytorch import Lion

start_time = time.time()

# Define setup and cleanup functions for distributed training
def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'  # Choose an open port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# H100 optimizations
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on H100
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN

# Training function
def train(rank, world_size, args):
    setup(rank, world_size)

    # Initialize wandb only on rank 0 to prevent multiple initializations
    if rank == 0:
        if not args.resume:
            run = wandb.init(project='samantha-medium', name='samantha-medium-fineweb-training')
            run_id = wandb.run.id
            os.makedirs('./emergency_checkpoint_medium', exist_ok=True)
            with open('./emergency_checkpoint_medium/run_id.txt', 'w') as f:
                f.write(run_id)
        else:
            if os.path.exists('./emergency_checkpoint_medium/run_id.txt'):
                with open('./emergency_checkpoint_medium/run_id.txt', 'r') as f:
                    run_id = f.read().strip()
                run = wandb.init(project='samantha-medium',
                                 id=run_id,
                                 resume="allow")
                print(f"Resuming WandB run with id {run_id}")
            else:
                run = wandb.init(project='samantha-medium', name='samantha-medium-fineweb-training')

    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    global block_size
    block_size = 512  # For testing purposes; set to 2048 for full training

    # Load datasets
    dataset1 = load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=True
    )

    dataset2 = load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2024-18",
        split="train",
        streaming=True
    )

    dataset3 = load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2023-50",
        split="train",
        streaming=True
    )

    # Combine the datasets
    dataset = interleave_datasets([dataset1, dataset2, dataset3])

    # Tokenize the dataset stream
    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, max_length=block_size)

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
    )

    # function to chunk data into batches
    def chunked_iterator(iterable, chunk_size):
        iterator = iter(iterable)
        for first in iterator:
            chunk = [first] + list(islice(iterator, chunk_size - 1))
            yield {
                key: [example[key] for example in chunk]
                for key in chunk[0].keys()
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

    # Create an iterable that yields grouped texts
    grouped_dataset = (
        group_texts(batch) for batch in chunked_iterator(tokenized_dataset, chunk_size=1000)
    )

    ###########
    # Dataset #
    ###########

    class StreamDataset(IterableDataset):
        def __init__(self, grouped_dataset, rank, world_size):
            super().__init__()
            self.grouped_dataset = grouped_dataset
            self.rank = rank
            self.world_size = world_size

        def __iter__(self):
            # distribute the dataset to different nodes
            for idx, batch in enumerate(self.grouped_dataset):
                # Only yield the batch if the index corresponds to the rank
                if (idx % self.world_size) == self.rank:
                    for i in range(len(batch['input_ids'])):
                        yield {
                            'input_ids': torch.tensor(batch['input_ids'][i], dtype=torch.long),
                            'labels': torch.tensor(batch['labels'][i], dtype=torch.long),
                        }

    train_dataset = StreamDataset(grouped_dataset=grouped_dataset, rank=rank, world_size=world_size)
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=0)

    # Load the validation dataset (Wikitext-2)
    validation_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    # Tokenize
    tokenized_validation_dataset = validation_dataset.map(
        tokenize_function,
        remove_columns=validation_dataset.column_names,
    )

    # Group
    def group_texts_validation(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
                for k in concatenated.keys()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    lm_validation_dataset = tokenized_validation_dataset.map(
        group_texts_validation,
        batched=True,
        batch_size=1000,
    )

    # Set format for PyTorch
    lm_validation_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    val_sampler = torch.utils.data.distributed.DistributedSampler(lm_validation_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    validation_dataloader = DataLoader(lm_validation_dataset, batch_size=4, num_workers=0, sampler=val_sampler)

    # Model with Pre-LayerNorm and Biases aka GPT-3 (Samantha uses GPT-2 architecture)
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

    # Configuration for 350M Samantha model (GPT-2 based architecture)
    config = CustomGPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=2048,     # Maximum position embeddings
        n_ctx=2048,           # Maximum context length for generation
        n_embd=1024,          # Embedding dimension for 350M model
        n_layer=24,           # Number of transformer blocks
        n_head=16,            # Number of attention heads
        n_inner=4096,         # Feedforward dimension (4x n_embd)
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
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            # Apply scaling to residual projections
            module.weight.data.mul_(1 / math.sqrt(2 * config.n_layer))

    model.apply(custom_init_weights)

    device = torch.device("cpu")
    model.to(device)

    model = torch.compile(model, mode='max-autotune')
    model = DDP(model)

    # Set up the optimizer and learning rate scheduler
    total_steps = 600000  # desired total training steps
    warmup_steps = 60000  # Typically 10% of total_steps

    # optimizer hyperparameters (Lion optimizer for medium model)
    optimizer = Lion(
        model.parameters(),
        lr=6e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    gradient_accumulation_steps = 16  # Increased for memory efficiency with smaller batch size (medium model)

    # gradient clipping
    max_grad_norm = 1.0

    # training variables
    global_step = 0  # Moved initialization here for resume functionality

    # Resume training if --resume flag is used
    if args.resume:
        checkpoint_dir = './emergency_checkpoint_medium'
        if os.path.exists(checkpoint_dir):
            if rank == 0:
                print("Loading checkpoint from the emergency directory...")
            # Load model state
            model.module.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model_state.pt')))
            # Load optimizer state
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer_state.pt')))
            # Load scheduler state
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scheduler_state.pt')))
            # Load global_step
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'r') as f:
                global_step = int(f.readline().strip())
            if rank == 0:
                print(f"Resumed training from step {global_step}")
        else:
            if rank == 0:
                print("No checkpoint found in the emergency directory. Starting from scratch.")

    # Training loop
    model.train()
    logging_steps = 100
    save_steps = 10000   # Save model every save_steps
    eval_steps = 1000    # Evaluate model every eval_steps
    emergency_save_steps = 100  # Save temporary checkpoint every 100 steps

    # Cycle to create an infinite iterator over the dataloader
    train_iterator = cycle(train_dataloader)

    start_time = time.time()

    try:
        while True:
            optimizer.zero_grad()
            accumulated_loss = 0.0

            for accumulation_step in range(gradient_accumulation_steps):
                batch = next(train_iterator)
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                accumulated_loss += loss.item()
                loss = loss / gradient_accumulation_steps
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Live updating line (only on rank 0)
            if rank == 0:
                avg_loss = accumulated_loss / gradient_accumulation_steps
                print(f"\rStep [{global_step}/{total_steps}], Loss: {avg_loss:.4f}", end='', flush=True)

            # Emergency save (only on rank 0)
            if global_step % emergency_save_steps == 0 and rank == 0:
                checkpoint_dir = './emergency_checkpoint_medium'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
                with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                    f.write(f"{global_step}\n")
                print(f"\nTemporary checkpoint saved at step {global_step}.")

            # Logging every logging_steps (only on rank 0)
            if global_step % logging_steps == 0 and rank == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                avg_loss = accumulated_loss / gradient_accumulation_steps
                perplexity = math.exp(avg_loss) if avg_loss < 7 else float('inf')

                print()
                print(f"Step {global_step}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, "
                      f"Grad Norm: {grad_norm:.4f}, Time per step: {elapsed_time / logging_steps:.4f} sec")
                # Log to wandb
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    'train/loss': avg_loss,
                    'train/perplexity': perplexity,
                    'train/grad_norm': grad_norm,
                    'train/lr': current_lr,
                    'train/step': global_step,
                })

            # Validation every eval_steps (only on rank 0)
            if (global_step % eval_steps == 0 or global_step == 0) and rank == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                max_val_batches = 100
                with torch.no_grad():
                    for i, batch in enumerate(validation_dataloader):
                        if i >= max_val_batches:
                            break
                        inputs = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**inputs)
                        loss = outputs.loss
                        val_loss += loss.item()
                        val_steps += 1

                avg_val_loss = val_loss / val_steps
                val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 7 else float('inf')

                print(f"\nValidation at step {global_step}: Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                wandb.log({
                    'validation/loss': avg_val_loss,
                    'validation/perplexity': val_perplexity,
                    'validation/step': global_step,
                })

                model.train()

            # Saving the model (only on rank 0)
            if global_step > 0 and global_step % save_steps == 0 and rank == 0:
                save_path = f'./samantha-medium-fineweb-step-{global_step}'
                model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"\nModel saved at step {global_step} to {save_path}")

            if global_step >= total_steps:
                break
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user. Saving model...")
            save_path = f'./samantha-medium-fineweb-step-{global_step}'
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at step {global_step} to {save_path}")

            checkpoint_dir = './emergency_checkpoint_medium'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                f.write(f"{global_step}\n")
            print(f"Emergency checkpoint saved at step {global_step}.")

    # Final save (only on rank 0)
    if rank == 0:
        final_save_path = './samantha-medium-fineweb'
        model.module.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"\nFinal model saved to {final_save_path}")

    # Finish wandb (only on rank 0)
    if rank == 0:
        try:
            wandb.finish()
        except:
            pass

    cleanup()

# Spawn processes
def main():
    parser = argparse.ArgumentParser(description='Training script for 350M Samantha model with resume functionality and DDP.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')
    args = parser.parse_args()

    #world_size = torch.cuda.device_count()
    world_size=2
    if world_size < 1:
        print("No GPUs available for training.")
        sys.exit(1)
    # Считываем rank, world_size из окружения:
    local_rank = int(os.environ["LOCAL_RANK"])   # local rank on this node
    rank = int(os.environ["RANK"])               # global rank (between all nodes)
    world_size = int(os.environ["WORLD_SIZE"])   # num processes

    train(rank, world_size, args)
    #mp.spawn(train,
    #         args=(world_size, args),
    #         nprocs=world_size,
    #         join=True)

if __name__ == "__main__":
    main()
