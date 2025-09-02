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

start_time = time.time()

torch.set_default_dtype(torch.bfloat16)

parser = argparse.ArgumentParser(description='Training script with resume functionality.')
parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')
args = parser.parse_args()

wandb.init(project='gpt3-small-fineweb', name='fineweb-training-600,000')

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
    def __init__(self, grouped_dataset):
        self.grouped_dataset = grouped_dataset

    def __iter__(self):
        for batch in self.grouped_dataset:
            for i in range(len(batch['input_ids'])):
                yield {
                    'input_ids': torch.tensor(batch['input_ids'][i], dtype=torch.long),
                    'labels': torch.tensor(batch['labels'][i], dtype=torch.long),
                }

train_dataset = StreamDataset(grouped_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=12, num_workers=0) # MANAGE BATCH SIZE HERE

# Velidate on different dataset to test how well the model generalizes
validation_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

# Tokenize
tokenized_validation_dataset = validation_dataset.map(
    tokenize_function,
    remove_columns=validation_dataset.column_names,
)

# Group
lm_validation_dataset = tokenized_validation_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
)

# Set format for PyTorch
lm_validation_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
validation_dataloader = DataLoader(lm_validation_dataset, batch_size=12, num_workers=0) # MANAGE BATCH SIZE HERE

class CustomGPT2Config(GPT2Config):
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm

# Import custom model classes from model.py
from model import (
    CustomGPT2Config,
    CustomGPT2LMHeadModel,
)

# Custom model classes are now imported from model.py

# Configuration for 125M model
config = CustomGPT2Config(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=2048, # Max position embeddings aka maximum length that the model can handle
    n_ctx=2048,                   # Max context length for generation
    hidden_size=768,              # Embedding dimension
    num_hidden_layers=12,         # Number of transformer blocks
    num_attention_heads=12,       # Number of attention heads
    intermediate_size=3072,       # Dimension of the feedforward layer
    activation_function='gelu',   # Standard GeLU
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

# Move the model to the appropriate device (GPU, MPS, or CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
model.to(device)

# Set up the optimizer and learning rate scheduler
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

gradient_accumulation_steps = 1 #4

# gradient clipping
max_grad_norm = 1.0

# training variables
global_step = 0  # Moved initialization here for resume functionality

# Resume training if --resume flag is used
if args.resume:
    checkpoint_dir = './emergency_checkpoint'
    if os.path.exists(checkpoint_dir):
        print("Loading checkpoint from the emergency directory...")
        # Load model state
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model_state.pt')))
        # Load optimizer state
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer_state.pt')))
        # Load scheduler state
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scheduler_state.pt')))
        # Load global_step
        with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'r') as f:
            global_step = int(f.readline().strip())
        print(f"Resumed training from step {global_step}")
    else:
        print("No checkpoint found in the emergency directory. Starting from scratch.")

# Training loop
model.train()
logging_steps = 100
save_steps = 10000   # Save model every save_steps
eval_steps = 1000    # Evaluate model every eval_steps
emergency_save_steps = 100  # Save temporary checkpoint every 100 steps

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

        # emergency save
        if global_step % emergency_save_steps == 0:
            checkpoint_dir = './emergency_checkpoint'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                f.write(f"{global_step}\n")
            print(f"Temporary checkpoint saved at step {global_step}.")

        # Logging
        if global_step % logging_steps == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            avg_loss = accumulated_loss / gradient_accumulation_steps
            perplexity = math.exp(avg_loss) if avg_loss < 7 else float('inf')  # To avoid overflow, fix later
            print(f"Step {global_step}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Grad Norm: {grad_norm:.4f}, Time per step: {elapsed_time / logging_steps:.4f} sec")
            # Log metrics to wandb with 'train/' prefix
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/grad_norm': grad_norm,
                'train/lr': current_lr,
                'train/step': global_step,
            })

        # Validation every eval_steps
        if global_step % eval_steps == 0 or global_step == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            max_val_batches = 100  # Limit the number of validation batches
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

            print(f"Validation at step {global_step}: Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            # Log validation metrics to wandb with 'validation/' prefix
            wandb.log({
                'validation/loss': avg_val_loss,
                'validation/perplexity': val_perplexity,
                'validation/step': global_step,
            })

            model.train()

        # Saving the model
        if global_step > 0 and global_step % save_steps == 0:
            save_path = f'./gpt3-small-fineweb-step-{global_step}'
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at step {global_step} to {save_path}")

        if global_step >= total_steps:
            break
except KeyboardInterrupt:
    # Save emergency checkpoint upon interruption
    print("Training interrupted by user. Saving model...")
    save_path = f'./gpt3-small-fineweb-step-{global_step}'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved at step {global_step} to {save_path}")

    checkpoint_dir = './emergency_checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
    with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
        f.write(f"{global_step}\n")
    print(f"Emergency checkpoint saved at step {global_step}.")

final_save_path = './gpt3-small-fineweb'
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final model saved to {final_save_path}")

try:
    wandb.finish()
except:
    pass
