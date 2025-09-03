# Install necessary libraries
# !pip install transformers datasets wandb

import os
import math
import logging
import torch
import torch.nn as nn
import shutil
import glob
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from datasets import load_dataset, interleave_datasets
import wandb
from itertools import islice, cycle
import time
import argparse
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

start_time = time.time()

# Retry function for handling streaming errors
def retry_with_backoff(func, max_retries=5, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    """
    Retry a function with exponential backoff on failures.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Factor to multiply delay by after each failure
    """
    last_exception = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                print(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")

    raise last_exception

# Define cleanup function for distributed training
def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

# H100 optimizations
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on H100
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN

# Training function
def train(rank, world_size, args):

    # Initialize wandb only on rank 0 to prevent multiple initializations
    if rank == 0:
        if not args.resume:
            run = wandb.init(project='samantha-350m', name='samantha-350m-fineweb-training')
            run_id = wandb.run.id
            os.makedirs('./emergency_checkpoint_samantha_350m', exist_ok=True)
            with open('./emergency_checkpoint_samantha_350m/run_id.txt', 'w') as f:
                f.write(run_id)
        else:
            if os.path.exists('./emergency_checkpoint_samantha_350m/run_id.txt'):
                with open('./emergency_checkpoint_samantha_350m/run_id.txt', 'r') as f:
                    run_id = f.read().strip()
                run = wandb.init(project='samantha-350m',
                                 id=run_id,
                                 resume="allow")
                print(f"Resuming WandB run with id {run_id}")
            else:
                run = wandb.init(project='samantha-350m', name='samantha-350m-fineweb-training')

    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    global block_size
    block_size = 512  # For testing purposes; set to 2048 for full training

    # Load datasets with retry logic for streaming errors
    print("Loading dataset 1 (CC-MAIN-2024-10)...")
    dataset1 = retry_with_backoff(
        lambda: load_dataset(
            "HuggingFaceFW/fineweb",
            name="CC-MAIN-2024-10",
            split="train",
            streaming=True
        )
    )

    print("Loading dataset 2 (CC-MAIN-2024-18)...")
    dataset2 = retry_with_backoff(
        lambda: load_dataset(
            "HuggingFaceFW/fineweb",
            name="CC-MAIN-2024-18",
            split="train",
            streaming=True
        )
    )

    print("Loading dataset 3 (CC-MAIN-2023-50)...")
    dataset3 = retry_with_backoff(
        lambda: load_dataset(
            "HuggingFaceFW/fineweb",
            name="CC-MAIN-2023-50",
            split="train",
            streaming=True
        )
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
            batch_idx = 0
            grouped_iter = iter(self.grouped_dataset)

            while True:
                try:
                    batch = next(grouped_iter)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error during dataset iteration at batch {batch_idx}: {str(e)}")
                    print("Attempting to skip this batch and continue...")
                    batch_idx += 1
                    continue

                # Only yield the batch if the index corresponds to the rank
                if (batch_idx % self.world_size) == self.rank:
                    try:
                        for i in range(len(batch['input_ids'])):
                            yield {
                                'input_ids': torch.tensor(batch['input_ids'][i], dtype=torch.long),
                                'labels': torch.tensor(batch['labels'][i], dtype=torch.long),
                            }
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}, sample {i}: {str(e)}")
                        print("Skipping this sample and continuing...")
                        continue

                batch_idx += 1

    train_dataset = StreamDataset(grouped_dataset=grouped_dataset, rank=rank, world_size=world_size)
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=0)

    # Load the validation dataset (Wikitext-2) with retry logic
    print("Loading validation dataset (Wikitext-2)...")
    validation_dataset = retry_with_backoff(
        lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    )

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
    # Import custom model classes from model.py
    from model import (
        CustomGPT2Config,
        CustomGPT2LMHeadModel,
    )

    # Custom model classes are now imported from model.py

    # Configuration for 350M Samantha model (GPT-2 based architecture)
    config = CustomGPT2Config(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2048,     # Maximum position embeddings
        n_ctx=2048,                       # Maximum context length for generation
        hidden_size=1024,                 # Embedding dimension for 350M model
        num_hidden_layers=24,             # Number of transformer blocks
        num_attention_heads=16,           # Number of attention heads
        intermediate_size=4096,           # Feedforward dimension (4x hidden_size)
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
            module.weight.data.mul_(1 / math.sqrt(2 * config.num_hidden_layers))

    model.apply(custom_init_weights)

    device = torch.device("cpu")
    model.to(device)

    model = DDP(model)

    # Set up the optimizer and learning rate scheduler for 3 epochs
    # Estimate dataset size based on the streaming datasets (approximate)
    # CC-MAIN-2024-10: ~15M samples, CC-MAIN-2024-18: ~15M samples, CC-MAIN-2023-50: ~10M samples
    # Total approximate: ~40M samples, each batch has ~4 samples * block_size tokens
    # Note: These are rough estimates. The actual dataset size may vary.
    estimated_samples_per_epoch = 40000000  # Approximate total samples in dataset
    samples_per_batch = 4  # batch_size
    steps_per_epoch = estimated_samples_per_epoch // samples_per_batch // world_size  # Divide by world_size for distributed training

    num_epochs = 3
    total_steps = steps_per_epoch * num_epochs  # Total steps for 3 epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    if rank == 0:
        print(f"Training configuration:")
        print(f"- Number of epochs: {num_epochs}")
        print(f"- Estimated steps per epoch: {steps_per_epoch}")
        print(f"- Total training steps: {total_steps}")
        print(f"- Warmup steps: {warmup_steps}")
        print(f"- World size (processes): {world_size}")
        print(f"- Batch size per process: {samples_per_batch}")
        print("-" * 50)

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

    gradient_accumulation_steps = 8  # Increased for memory efficiency with smaller batch size

    # gradient clipping
    max_grad_norm = 1.0

    # training variables
    global_step = 0  # Moved initialization here for resume functionality
    current_epoch = 0
    steps_in_current_epoch = 0

    # Resume training if --resume flag is used
    if args.resume:
        checkpoint_dir = './emergency_checkpoint_samantha_350m'
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

    def cleanup_old_checkpoints(max_checkpoints=50):
        """Keep only the most recent max_checkpoints checkpoints."""
        checkpoint_pattern = "./samantha-350m-fineweb-step-*"
        checkpoint_dirs = glob.glob(checkpoint_pattern)

        if len(checkpoint_dirs) <= max_checkpoints:
            return

        # Extract step numbers and sort by step number (newest first)
        checkpoints_with_steps = []
        for checkpoint_dir in checkpoint_dirs:
            try:
                # Extract step number from directory name like "samantha-350m-fineweb-step-5000000"
                step_str = checkpoint_dir.split("-step-")[-1]
                step_num = int(step_str)
                checkpoints_with_steps.append((step_num, checkpoint_dir))
            except (ValueError, IndexError):
                # Skip directories that don't match the expected pattern
                continue

        # Sort by step number (descending - newest first)
        checkpoints_with_steps.sort(key=lambda x: x[0], reverse=True)

        # Keep only the most recent max_checkpoints
        checkpoints_to_keep = checkpoints_with_steps[:max_checkpoints]
        checkpoints_to_delete = checkpoints_with_steps[max_checkpoints:]

        # Delete old checkpoints
        for _, checkpoint_dir in checkpoints_to_delete:
            try:
                shutil.rmtree(checkpoint_dir)
                print(f"Deleted old checkpoint: {checkpoint_dir}")
            except Exception as e:
                print(f"Failed to delete checkpoint {checkpoint_dir}: {e}")

        if checkpoints_to_delete:
            print(f"Kept {len(checkpoints_to_keep)} most recent checkpoints, deleted {len(checkpoints_to_delete)} old ones.")

    # Training loop
    model.train()
    logging_steps = 100
    save_steps = 5000000   # Save model every 5M steps
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
            steps_in_current_epoch += 1

            # Check if we've completed an epoch
            if steps_in_current_epoch >= steps_per_epoch:
                current_epoch += 1
                steps_in_current_epoch = 0
                if rank == 0:
                    print(f"\n{'='*50}")
                    print(f"Epoch {current_epoch}/{num_epochs} completed!")
                    print(f"{'='*50}")

            # Live updating line (only on rank 0)
            if rank == 0:
                avg_loss = accumulated_loss / gradient_accumulation_steps
                epoch_progress = steps_in_current_epoch / steps_per_epoch
                print(f"\rEpoch {current_epoch + 1}/{num_epochs} [{steps_in_current_epoch}/{steps_per_epoch}] "
                      f"Step [{global_step}/{total_steps}], Loss: {avg_loss:.4f}", end='', flush=True)

            # Emergency save (only on rank 0)
            if global_step % emergency_save_steps == 0 and rank == 0:
                checkpoint_dir = './emergency_checkpoint_samantha_350m'
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
                print(f"Epoch {current_epoch + 1}/{num_epochs}, Step {global_step}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, "
                      f"Grad Norm: {grad_norm:.4f}, Time per step: {elapsed_time / logging_steps:.4f} sec")
                # Log to wandb
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    'train/loss': avg_loss,
                    'train/perplexity': perplexity,
                    'train/grad_norm': grad_norm,
                    'train/lr': current_lr,
                    'train/step': global_step,
                    'train/epoch': current_epoch + 1,
                    'train/epoch_progress': steps_in_current_epoch / steps_per_epoch,
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

                print(f"\nValidation at Epoch {current_epoch + 1}/{num_epochs}, Step {global_step}: Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                wandb.log({
                    'validation/loss': avg_val_loss,
                    'validation/perplexity': val_perplexity,
                    'validation/step': global_step,
                    'validation/epoch': current_epoch + 1,
                })

                model.train()

            # Saving the model (only on rank 0)
            if global_step > 0 and global_step % save_steps == 0 and rank == 0:
                save_path = f'./samantha-350m-fineweb-step-{global_step}'
                model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"\nModel saved at step {global_step} to {save_path}")

                # Clean up old checkpoints, keeping only the 50 most recent
                cleanup_old_checkpoints(max_checkpoints=50)

            # Check if we've completed all epochs
            if current_epoch >= num_epochs:
                if rank == 0:
                    print(f"\n{'='*60}")
                    print(f"Training completed! Finished {num_epochs} epochs ({global_step} total steps)")
                    print(f"{'='*60}")
                break
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user. Saving model...")
            save_path = f'./samantha-350m-fineweb-step-{global_step}'
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved at step {global_step} to {save_path}")

            # Clean up old checkpoints, keeping only the 50 most recent
            cleanup_old_checkpoints(max_checkpoints=50)

            checkpoint_dir = './emergency_checkpoint_samantha_350m'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                f.write(f"{global_step}\n")
            print(f"Emergency checkpoint saved at step {global_step}.")

    # Final save (only on rank 0)
    if rank == 0:
        final_save_path = './samantha-350m-fineweb'
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
    parser.add_argument('--world_size', type=int, default=2, help='Number of processes for distributed training.')
    args = parser.parse_args()

    # Use environment variables if available, otherwise use defaults
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        # Environment variables are set (likely by torchrun or similar)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Manual setup for single node
        rank = 0
        world_size = args.world_size
        local_rank = 0

    print(f"Starting training with rank {rank}, world_size {world_size}, local_rank {local_rank}")

    # Use torchrun compatible distributed training
    if world_size > 1:
        # Initialize the process group
        torch.distributed.init_process_group(
            backend="gloo",  # Use gloo for CPU training
            rank=rank,
            world_size=world_size,
            init_method="env://"  # Use environment variables
        )

    train(rank, world_size, args)

if __name__ == "__main__":
    main()
