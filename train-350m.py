# train_fixed.py
# Dependências: transformers, datasets, wandb, torch
import os
import math
import logging
import torch
import torch.nn as nn
import shutil
import glob
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset, interleave_datasets
import wandb
from itertools import islice
import time
import argparse
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -------------------------
# util: retry/backoff
# -------------------------
def retry_with_backoff(func, max_retries=5, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    last_exception = None
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                print(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
    raise last_exception

# -------------------------
# DDP setup/cleanup
# -------------------------
def setup(rank, world_size):
    # set MASTER_ADDR/PORT if absent (useful para torchrun também)
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12355')
    # NCCL is preferred for GPU; fallback to gloo if no CUDA
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# -------------------------
# Train function
# -------------------------
def train(rank, world_size, args):
    setup(rank, world_size)

    # Only rank 0 initializes wandb
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
                run = wandb.init(project='samantha-350m', id=run_id, resume="allow")
                print(f"Resuming WandB run with id {run_id}")
            else:
                run = wandb.init(project='samantha-350m', name='samantha-350m-fineweb-training')

    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    block_size = 512  # para testes; você pode colocar 2048

    # Load streaming datasets with retry
    dataset1 = retry_with_backoff(lambda: load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True))
    dataset2 = retry_with_backoff(lambda: load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-18", split="train", streaming=True))
    dataset3 = retry_with_backoff(lambda: load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2023-50", split="train", streaming=True))
    dataset = interleave_datasets([dataset1, dataset2, dataset3])

    def tokenize_function(example):
        # some examples may be empty or non-text; guard
        txt = example.get("text", "") if isinstance(example, dict) else ""
        return tokenizer(txt, truncation=True, max_length=block_size)

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

    # chunked iterator (gera batches grandes de exemplos tokenizados que serão agrupados em blocks)
    def chunked_iterator(iterable, chunk_size):
        iterator = iter(iterable)
        for first in iterator:
            chunk = [first] + list(islice(iterator, chunk_size - 1))
            yield {
                key: [example[key] for example in chunk]
                for key in chunk[0].keys()
            }

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [concatenated[k][i: i + block_size] for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    grouped_dataset_iter = (group_texts(batch) for batch in chunked_iterator(tokenized_dataset, chunk_size=1000))

    class StreamDataset(IterableDataset):
        def __init__(self, grouped_iterator, rank, world_size):
            super().__init__()
            self.grouped_iterator = grouped_iterator
            self.rank = rank
            self.world_size = world_size

        def __iter__(self):
            grouped_iter = iter(self.grouped_iterator)
            batch_idx = 0
            while True:
                try:
                    batch = next(grouped_iter)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"[rank {self.rank}] error iter dataset batch {batch_idx}: {e}, skipping")
                    batch_idx += 1
                    continue

                if (batch_idx % self.world_size) != self.rank:
                    batch_idx += 1
                    continue

                for i in range(len(batch['input_ids'])):
                    try:
                        yield {
                            'input_ids': torch.tensor(batch['input_ids'][i], dtype=torch.long),
                            'labels': torch.tensor(batch['labels'][i], dtype=torch.long),
                        }
                    except Exception as e:
                        print(f"[rank {self.rank}] error processing sample {i} in batch {batch_idx}: {e}, skipping")
                        continue
                batch_idx += 1

    train_dataset = StreamDataset(grouped_dataset_iter, rank=rank, world_size=world_size)
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=0)

    # validation dataset (non-streaming)
    validation_dataset = retry_with_backoff(lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="validation"))
    tokenized_validation_dataset = validation_dataset.map(tokenize_function, remove_columns=validation_dataset.column_names)

    def group_texts_validation(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [concatenated[k][i: i + block_size] for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    lm_validation_dataset = tokenized_validation_dataset.map(group_texts_validation, batched=True, batch_size=1000)
    lm_validation_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    val_sampler = torch.utils.data.distributed.DistributedSampler(lm_validation_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    validation_dataloader = DataLoader(lm_validation_dataset, batch_size=4, num_workers=0, sampler=val_sampler)

    # Import your model
    from model import CustomGPT2Config, CustomGPT2LMHeadModel

    # Config - adapte caso necessário (nomes compatíveis ao seu model.py)
    config = CustomGPT2Config(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2048,
        n_ctx=2048,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        activation_function='gelu',
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_pre_layernorm=True,
    )

    model = CustomGPT2LMHeadModel(config)

    # custom init (se quiser)
    def custom_init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    model.apply(custom_init_weights)

    # device & DDP
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # wrap DDP with device_ids when using single-node multi-gpu
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model = DDP(model, find_unused_parameters=False)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # estimate steps (you may want to compute more accurately)
    estimated_samples_per_epoch = 40000000
    samples_per_batch = 4
    steps_per_epoch = max(1, estimated_samples_per_epoch // samples_per_batch // world_size)
    num_epochs = 3
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps))

    gradient_accumulation_steps = 8
    max_grad_norm = 1.0

    # AMP scaler (mixed precision)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # resume logic (model.module is available since we wrapped with DDP)
    global_step = 0
    current_epoch = 0
    steps_in_current_epoch = 0

    if args.resume:
        checkpoint_dir = './emergency_checkpoint_samantha_350m'
        if os.path.exists(checkpoint_dir):
            if rank == 0:
                print("Loading checkpoint from the emergency directory...")
            # load on map_location device
            map_loc = {'cuda:%d' % 0: 'cuda:%d' % rank} if torch.cuda.is_available() else 'cpu'
            state_path = os.path.join(checkpoint_dir, 'model_state.pt')
            if os.path.exists(state_path):
                model.module.load_state_dict(torch.load(state_path, map_location=map_loc))
            opt_path = os.path.join(checkpoint_dir, 'optimizer_state.pt')
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=map_loc))
            sch_path = os.path.join(checkpoint_dir, 'scheduler_state.pt')
            if os.path.exists(sch_path):
                scheduler.load_state_dict(torch.load(sch_path, map_location=map_loc))
            train_state = os.path.join(checkpoint_dir, 'training_state.txt')
            if os.path.exists(train_state):
                with open(train_state, 'r') as f:
                    global_step = int(f.readline().strip())
            if rank == 0:
                print(f"Resumed training from step {global_step}")
        else:
            if rank == 0:
                print("No emergency checkpoint found - starting from scratch")

    # checkpoint cleanup util
    def cleanup_old_checkpoints(max_checkpoints=50):
        checkpoint_pattern = "./samantha-350m-fineweb-step-*"
        checkpoint_dirs = glob.glob(checkpoint_pattern)
        if len(checkpoint_dirs) <= max_checkpoints:
            return
        checkpoints_with_steps = []
        for d in checkpoint_dirs:
            try:
                step_str = d.split("-step-")[-1]
                step_num = int(step_str)
                checkpoints_with_steps.append((step_num, d))
            except:
                continue
        checkpoints_with_steps.sort(key=lambda x: x[0], reverse=True)
        to_delete = checkpoints_with_steps[max_checkpoints:]
        for _, d in to_delete:
            try:
                shutil.rmtree(d)
                print(f"Deleted {d}")
            except Exception as e:
                print(f"Failed to delete {d}: {e}")

    # TRAIN LOOP
    model.train()
    logging_steps = 100
    save_steps = 5_000_000
    eval_steps = 1000
    emergency_save_steps = 100

    # Create an iterator over the dataloader that we can re-create when exhausted
    train_loader_iter = iter(train_dataloader)

    start_time = time.time()
    try:
        while True:
            optimizer.zero_grad()
            accumulated_loss = 0.0

            for accumulation_step in range(gradient_accumulation_steps):
                # get next batch, re-create iterator if exhausted
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_dataloader)
                    batch = next(train_loader_iter)
                inputs = {k: v.to(device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(**inputs)
                    loss = outputs.loss

                accumulated_loss += loss.item()
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

            # gradient clipping + optimizer step via scaler
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            steps_in_current_epoch += 1

            if steps_in_current_epoch >= steps_per_epoch:
                current_epoch += 1
                steps_in_current_epoch = 0
                if rank == 0:
                    print(f"\nEpoch {current_epoch}/{num_epochs} completed")

            if rank == 0:
                avg_loss = accumulated_loss / gradient_accumulation_steps
                print(f"\rEpoch {current_epoch + 1}/{num_epochs} Step [{global_step}/{total_steps}] Loss: {avg_loss:.4f}", end='', flush=True)

            # Emergency save
            if global_step % emergency_save_steps == 0 and rank == 0:
                checkpoint_dir = './emergency_checkpoint_samantha_350m'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
                with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                    f.write(f"{global_step}\n")
                print(f"\nTemporary checkpoint saved at step {global_step}.")

            # Logging
            if global_step % logging_steps == 0 and rank == 0:
                elapsed = time.time() - start_time
                start_time = time.time()
                avg_loss = accumulated_loss / gradient_accumulation_steps
                perp = math.exp(avg_loss) if avg_loss < 7 else float('inf')
                curr_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None
                print()
                print(f"Step {global_step}, Loss {avg_loss:.4f}, Perp {perp:.2f}, GradNorm {grad_norm:.4f}, LR {curr_lr}, time/100steps {elapsed:.4f}")
                wandb.log({
                    'train/loss': avg_loss,
                    'train/perplexity': perp,
                    'train/grad_norm': grad_norm,
                    'train/lr': curr_lr,
                    'train/step': global_step,
                })

            # Validation
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
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            outputs = model(**inputs)
                            loss = outputs.loss
                        val_loss += loss.item()
                        val_steps += 1
                avg_val_loss = val_loss / max(1, val_steps)
                val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 7 else float('inf')
                print(f"\nValidation Step {global_step}: Loss {avg_val_loss:.4f}, Perp {val_perplexity:.2f}")
                wandb.log({
                    'validation/loss': avg_val_loss,
                    'validation/perplexity': val_perplexity,
                    'validation/step': global_step,
                })
                model.train()

            # Save final checkpoints periodically
            if global_step > 0 and global_step % save_steps == 0 and rank == 0:
                save_path = f'./samantha-350m-fineweb-step-{global_step}'
                os.makedirs(save_path, exist_ok=True)
                model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"\nModel saved at step {global_step} to {save_path}")
                cleanup_old_checkpoints(max_checkpoints=50)

            if current_epoch >= num_epochs:
                if rank == 0:
                    print(f"\nTraining finished after {num_epochs} epochs ({global_step} steps)")
                break

    except KeyboardInterrupt:
        if rank == 0:
            print("\nInterrupted by user: saving model...")
            save_path = f'./samantha-350m-fineweb-step-{global_step}'
            os.makedirs(save_path, exist_ok=True)
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            checkpoint_dir = './emergency_checkpoint_samantha_350m'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                f.write(f"{global_step}\n")
            print(f"Saved emergency checkpoint at step {global_step}.")

    # final save
    if rank == 0:
        final_save_path = './samantha-350m-fineweb'
        os.makedirs(final_save_path, exist_ok=True)
        model.module.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"\nFinal model saved to {final_save_path}")
        try:
            wandb.finish()
        except:
            pass

    cleanup()

# -------------------------
# Entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # get ranks from environment (torchrun sets these)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # sanity: if torchrun not used, try to infer world_size from cuda count
    if "WORLD_SIZE" not in os.environ:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if torch.cuda.is_available() and local_rank >= torch.cuda.device_count():
        print("LOCAL_RANK >= cuda count; adjust launcher or env vars.")
        sys.exit(1)

    train(rank, world_size, args)

if __name__ == "__main__":
    main()
