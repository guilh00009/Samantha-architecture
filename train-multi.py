#!/usr/bin/env python3
# train-multi-samantha-fixed.py
# Fixed/cleaned version of the script you provided.

import os
import math
import logging
import argparse
import time
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import load_dataset, interleave_datasets

# -------- CLI ----------
parser = argparse.ArgumentParser(description="Training script for Samantha variants (17M / 125M / 250M / 350M)")
parser.add_argument("--model_size", choices=["17m", "125m", "250m", "350m"], default="250m",
                    help="Which model size to train.")
parser.add_argument("--resume", action="store_true", help="Resume training from last emergency checkpoint")
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--block_size", type=int, default=256, help="Context length (reduce for memory)")
parser.add_argument("--micro_batch", type=int, default=1, help="Per-device micro-batch size")
parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--total_steps", type=int, default=600000)
parser.add_argument("--stream_chunk", type=int, default=25000)
parser.add_argument("--num_validation_batches", type=int, default=100)
# Tie by default; pass --no_tie_lm_head to disable
parser.add_argument("--no_tie_lm_head", action="store_true", help="Disable tying lm_head to token embeddings")
args = parser.parse_args()

# -------- quick environment safe defaults ----------
# Setting default dtype may be dangerous in some environments; leave as-is only if it succeeds.
try:
    torch.set_default_dtype(torch.bfloat16)
except Exception:
    # ignore if not supported
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# minimize HF cache footprint
os.environ['HF_DATASETS_CACHE'] = './hf_cache_minimal'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_CACHE'] = './hf_cache_minimal'

# -------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("samantha-train")

# ------------------- DDP SETUP (place near the top, after parser and args) -------------------
import torch.distributed as dist

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

use_ddp = WORLD_SIZE > 1

if use_ddp:
    # initialize process group; launcher usually already sets env vars
    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")
    logger.info(f"DDP mode: rank {RANK}/{WORLD_SIZE}, local_rank={LOCAL_RANK}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Single-process mode")

# -------- tokenizer ----------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
pad_token_id = tokenizer.pad_token_id

vocab_size = tokenizer.vocab_size
n_positions = args.block_size

# -------- model size to config mapping (values chosen earlier) ----------
size_map = {
    "17m": dict(n_embd=256, n_layer=6, n_head=8, intermediate=1024),
    "125m": dict(n_embd=768, n_layer=12, n_head=12, intermediate=3072),
    # new >125M sizes:
    "250m": dict(n_embd=896, n_layer=22, n_head=14, intermediate=3584),
    "350m": dict(n_embd=1024, n_layer=24, n_head=16, intermediate=4096),
}

if args.model_size not in size_map:
    raise ValueError("unsupported model_size")

cfg_spec = size_map[args.model_size]
logger.info(f"Selected model_size={args.model_size} -> {cfg_spec}")

# -------- dataset names (your original list) ----------
dataset_names = [
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-18",
    "CC-MAIN-2023-50",
]

def load_datasets(dataset_names, split="train", streaming=True):
    datasets_list = []
    for name in dataset_names:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb", name=name, split=split, streaming=streaming)
            datasets_list.append(ds)
            logger.info(f"Loaded streaming dataset {name}")
        except Exception as e:
            logger.warning(f"Could not load dataset {name}: {e}")
    if not datasets_list:
        raise RuntimeError("No datasets loaded. Check network and dataset names.")
    return datasets_list

training_datasets = load_datasets(dataset_names, split="train", streaming=True)
# interleave streaming datasets to form a single stream
training_dataset = interleave_datasets(training_datasets)

# ---------- streaming tokenization (on-the-fly) ----------
def tokenize_and_chunk_stream(dataset_iter, chunk_size=10, block_size=256):
    buffer_input_ids = []
    buffer_attention_mask = []
    for example in dataset_iter:
        # example can be a dict-like object; prefer 'text' key
        text = example.get("text", None)
        if not text:
            continue
        tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False, return_tensors='pt', return_attention_mask=True)
        ids = tokenized['input_ids'].squeeze().tolist()
        if isinstance(ids, int):
            ids = [ids]
        attn = tokenized['attention_mask'].squeeze().tolist()
        if isinstance(attn, int):
            attn = [attn]
        buffer_input_ids.append(ids)
        buffer_attention_mask.append(attn)
        if len(buffer_input_ids) >= chunk_size:
            yield {'input_ids': buffer_input_ids, 'attention_mask': buffer_attention_mask}
            buffer_input_ids = []
            buffer_attention_mask = []
    if buffer_input_ids:
        yield {'input_ids': buffer_input_ids, 'attention_mask': buffer_attention_mask}

def group_texts(examples, block_size):
    # examples['input_ids'] is list-of-lists
    concatenated = sum(examples['input_ids'], [])
    total_length = len(concatenated)
    total_length = (total_length // block_size) * block_size
    out = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    labels = [b.copy() for b in out]
    return {'input_ids': out, 'labels': labels}

stream_iter = tokenize_and_chunk_stream(training_dataset, chunk_size=args.stream_chunk, block_size=args.block_size)
# grouped_train_dataset is a generator that yields dicts with lists-of-blocks
grouped_train_dataset = (group_texts(batch, args.block_size) for batch in stream_iter)

# ------------------- MODIFY StreamDataset to shard across processes -------------------
class StreamDataset(IterableDataset):
    def __init__(self, grouped_dataset, rank=0, world_size=1):
        self.grouped_dataset = grouped_dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        # Each top-level batch yields many blocks; we'll stride so each process gets disjoint blocks.
        idx = 0
        for batch in self.grouped_dataset:
            blocks = batch['input_ids']
            labels = batch['labels']
            for i in range(len(blocks)):
                if (idx % self.world_size) == self.rank:
                    yield {
                        'input_ids': torch.tensor(blocks[i], dtype=torch.long),
                        'labels': torch.tensor(labels[i], dtype=torch.long),
                    }
                idx += 1

# Replace instantiation:
if use_ddp:
    train_dataset = StreamDataset(grouped_train_dataset, rank=RANK, world_size=WORLD_SIZE)
else:
    train_dataset = StreamDataset(grouped_train_dataset, rank=0, world_size=1)

# DataLoader: for IterableDataset, set batch_size to args.micro_batch to stack items into batches
train_dataloader = DataLoader(train_dataset, batch_size=args.micro_batch, num_workers=0)

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

# Similarly when building validation_data, run only on rank 0 (or shard similarly)
if use_ddp:
    if RANK == 0:
        validation_data = create_validation_dataset(train_dataset, args.num_validation_batches)
    else:
        validation_data = []
    dist.barrier()
else:
    validation_data = create_validation_dataset(train_dataset, args.num_validation_batches)

validation_dataloader = DataLoader(validation_data, batch_size=max(1, min(4, len(validation_data))), num_workers=0) if len(validation_data) > 0 else None

# -------- Custom GPT-2 components (based on your code) ----------
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
)

class CustomGPT2Config(GPT2Config):
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False):
        # call parent init so internal shapes are correct
        super().__init__(config, is_cross_attention=is_cross_attention)
        # Recreate linear layers to ensure explicit shapes (keeps semantics)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

class CustomGPT2MLP(GPT2MLP):
    def __init__(self, config):
        # GPT2MLP typically expects intermediate size as config.n_inner
        super().__init__(config.n_inner, config)
        # override layers to ensure consistent shapes and activation
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.act = nn.GELU()

class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        # call base init to ensure things like causal mask buffers are created
        super().__init__(config)
        self.use_pre_layernorm = getattr(config, "use_pre_layernorm", True)
        # replace attention and mlp with our custom versions (shapes consistent)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        # Use config.n_inner for mlp intermediate size (consistent with config)
        self.mlp = CustomGPT2MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    # keep forward behavior of GPT2Block (we rely on parent implementation)
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        # replace embeddings with explicit sizes (in case config changed)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # override blocks with our custom ones
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.init_weights()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        # create lm_head but we may tie it later
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # shift tokens for causal language modeling
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values if hasattr(transformer_outputs, 'past_key_values') else None,
            hidden_states=transformer_outputs.hidden_states if hasattr(transformer_outputs, 'hidden_states') else None,
            attentions=transformer_outputs.attentions if hasattr(transformer_outputs, 'attentions') else None,
            cross_attentions=transformer_outputs.cross_attentions if hasattr(transformer_outputs, 'cross_attentions') else None,
        )

# -------- Build configuration & model ----------
cfg = CustomGPT2Config(
    vocab_size=vocab_size,
    n_positions=n_positions,
    n_ctx=n_positions,
    n_embd=cfg_spec['n_embd'],
    n_layer=cfg_spec['n_layer'],
    n_head=cfg_spec['n_head'],
    n_inner=cfg_spec['intermediate'],  # GPT2 uses n_inner as MLP intermediate size
    activation_function='gelu',
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_pre_layernorm=True,
)

model = CustomGPT2LMHeadModel(cfg)

# Optionally tie lm_head to embedding to save memory (enabled by default unless --no_tie_lm_head)
tie_lm_head = not args.no_tie_lm_head
if tie_lm_head:
    try:
        model.lm_head.weight = model.transformer.wte.weight
        tied_ok = True
        logger.info("Tied lm_head weights to token embeddings (memory-saving).")
    except Exception as e:
        tied_ok = False
        logger.warning(f"Could not tie lm_head: {e}")
else:
    tied_ok = False

# compute and print parameter counts (detailed)
def compute_counts(vocab_size, n_positions, n_embd, n_layer, intermediate, tied):
    token_embed = vocab_size * n_embd
    pos_embed = n_positions * n_embd
    c_attn_w = n_embd * (3 * n_embd)
    c_attn_b = 3 * n_embd
    c_proj_w = n_embd * n_embd
    c_proj_b = n_embd
    c_fc_w = n_embd * intermediate
    c_fc_b = intermediate
    mlp_proj_w = intermediate * n_embd
    mlp_proj_b = n_embd
    # layernorm params: weight + bias per LayerNorm = 2 * n_embd
    ln_params = 2 * (2 * n_embd)  # ln_1 and ln_2 (weight + bias each)
    per_block = (c_attn_w + c_attn_b + c_proj_w + c_proj_b + c_fc_w + c_fc_b + mlp_proj_w + mlp_proj_b + ln_params)
    total_blocks = per_block * n_layer
    ln_f = n_embd * 2
    lm_head_untied = n_embd * vocab_size
    total_untied = token_embed + pos_embed + total_blocks + ln_f + lm_head_untied
    total_tied = total_untied - lm_head_untied
    return {
        'token_embed': token_embed,
        'pos_embed': pos_embed,
        'per_block': per_block,
        'total_blocks': total_blocks,
        'ln_f': ln_f,
        'lm_head_untied': lm_head_untied,
        'total_untied': total_untied,
        'total_tied': total_tied if tied else total_untied
    }

counts = compute_counts(vocab_size, n_positions, cfg_spec['n_embd'], cfg_spec['n_layer'], cfg_spec['intermediate'], tied_ok)
def bytes_to_mb(param_count, bytes_per_param=2):
    return param_count * bytes_per_param / 1024.0 / 1024.0

logger.info(f"Parameter breakdown (computed): token_embed={counts['token_embed']:,}, pos_embed={counts['pos_embed']:,}")
logger.info(f"Per-block params ~ {counts['per_block']:,}, blocks total ~ {counts['total_blocks']:,}")
logger.info(f"LM-head (untied) would add ~{counts['lm_head_untied']:,} params.")
if tied_ok:
    logger.info(f"Total parameters (tied): {counts['total_tied']:,} params -> BF16 params memory ≈ {bytes_to_mb(counts['total_tied']):.2f} MB")
    logger.info(f"Total parameters (untied would be): {counts['total_untied']:,} params -> BF16 ≈ {bytes_to_mb(counts['total_untied']):.2f} MB")
else:
    logger.info(f"Total parameters (untied): {counts['total_untied']:,} params -> BF16 params memory ≈ {bytes_to_mb(counts['total_untied']):.2f} MB")

# -------- small init tweaks and gradient checkpointing ----------
def custom_init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
        module.weight.data.mul_(1.0 / math.sqrt(2.0 * max(1, cfg.n_layer)))

model.apply(custom_init_weights)
# enable gradient checkpointing to save activation memory if supported
try:
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing.")
except Exception:
    logger.warning("Could not enable gradient checkpointing on this model/version.")

# ------------------- After model creation, move model to device and wrap DDP -------------------
model.to(device)

# enable BF16 autocast context where appropriate
use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(device.index if hasattr(device, 'index') else 0)[0] >= 7)

if tie_lm_head:
    try:
        model.lm_head.weight = model.transformer.wte.weight
    except Exception:
        pass

if use_ddp:
    # wrap in DistributedDataParallel (module must be on device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)

# ------------------- Training loop changes: use autocast BF16 and DDP-safe checkpointing -------------------
from contextlib import contextmanager

@contextmanager
def autocast_if_bf16():
    if use_bf16 and torch.cuda.is_available():
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        with ctx:
            yield
    else:
        yield

# -------- optimizer and scheduler ----------
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * args.total_steps), num_training_steps=args.total_steps)

# GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler() if use_bf16 else None

# -------- wandb init (optional) ----------
run = None
if args.wandb:
    try:
        import wandb
        run_name = f"samantha-{args.model_size}-{int(time.time())}"
        os.makedirs('./emergency_checkpoint_17m', exist_ok=True)
        run = wandb.init(project="samantha-multi", name=run_name)
        with open('./emergency_checkpoint_17m/run_id.txt', 'w') as f:
            f.write(wandb.run.id)
        logger.info("WandB initialized")
    except Exception as e:
        logger.warning(f"WandB init failed: {e}")
        run = None

# -------- resume emergency checkpoint if requested ----------
checkpoint_dir = './emergency_checkpoint_17m'
global_step = 0
if args.resume and os.path.exists(checkpoint_dir):
    try:
        logger.info("Loading emergency checkpoint...")
        # Note: if model is DDP-wrapped, need to load into module
        map_loc = {"cuda:0": f"cuda:{LOCAL_RANK}"} if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(os.path.join(checkpoint_dir, 'model_state.pt'), map_location=map_loc)
        # If DDP, model might be wrapped; load to module if needed
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer_state.pt'), map_location=map_loc))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scheduler_state.pt'), map_location=map_loc))
        if scaler is not None and os.path.exists(os.path.join(checkpoint_dir, 'scaler_state.pt')):
            scaler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scaler_state.pt'), map_location=map_loc))
        with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'r') as f:
            global_step = int(f.readline().strip())
        logger.info(f"Resumed from step {global_step}")
    except Exception as e:
        logger.warning(f"Could not resume cleanly: {e}")
        global_step = 0

# -------- training hyperparams ----------
gradient_accumulation_steps = args.grad_accum
logging_steps = 100
save_steps = 5000000
eval_steps = 2000
emergency_save_steps = 500
total_steps = args.total_steps
max_grad_norm = 1.0

train_iterator = cycle(train_dataloader)

# -------- training loop ----------
model.train()
try:
    while global_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for acc_step in range(gradient_accumulation_steps):
            batch = next(train_iterator)
            # batch is dict with 'input_ids' shape (micro_batch, seq_len)
            inputs = {k: v.to(device) for k, v in batch.items()}

            # pad if necessary (should be rare)
            if inputs['input_ids'].shape[-1] != args.block_size:
                to_pad = args.block_size - inputs['input_ids'].shape[-1]
                if to_pad > 0:
                    pad_tok = pad_token_id or tokenizer.eos_token_id
                    pad_tensor = torch.full((inputs['input_ids'].shape[0], to_pad), pad_tok, dtype=torch.long, device=device)
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], pad_tensor], dim=1)
                    inputs['labels'] = torch.cat([inputs['labels'], pad_tensor], dim=1)

            with autocast_if_bf16():
                out = model(input_ids=inputs['input_ids'], attention_mask=(inputs['input_ids'] != pad_token_id).long(), labels=inputs['labels'])
                loss = out.loss
                if loss is None:
                    raise RuntimeError("No loss returned; check labels shape.")
                loss = loss / gradient_accumulation_steps

                # Use scaler for BF16 training
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            accumulated_loss += loss.item() * gradient_accumulation_steps  # accumulate raw for logging

        # gradient clipping and optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        global_step += 1

        # Checkpoint saving (only rank 0 writes full model files to disk)
        if global_step % emergency_save_steps == 0:
            if use_ddp:
                dist.barrier()
                if RANK == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(save_state, os.path.join(checkpoint_dir, 'model_state.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
                    if scaler is not None:
                        torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, 'scaler_state.pt'))
                    with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                        f.write(f"{global_step}\n")
                    logger.info(f"Emergency checkpoint saved at step {global_step}")
                dist.barrier()
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
                if scaler is not None:
                    torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, 'scaler_state.pt'))
                with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                    f.write(f"{global_step}\n")
                logger.info(f"Emergency checkpoint saved at step {global_step}")

        # logging
        if global_step % logging_steps == 0:
            avg_loss = accumulated_loss / gradient_accumulation_steps
            perp = math.exp(avg_loss) if avg_loss < 50 else float('inf')
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else None
            logger.info(f"Step {global_step}, loss={avg_loss:.4f}, perp={perp:.2f}, grad_norm={grad_norm:.4f}, lr={lr}")
            if run:
                run.log({'train/loss': avg_loss, 'train/perplexity': perp, 'train/grad_norm': grad_norm, 'train/lr': lr, 'train/step': global_step})

        # evaluation
        if (global_step % eval_steps == 0) or (global_step == 1):
            if validation_dataloader is not None and len(validation_dataloader) > 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for vb in validation_dataloader:
                        vb = {k: v.to(device) for k, v in vb.items()}
                        out = model(input_ids=vb['input_ids'], attention_mask=(vb['input_ids'] != pad_token_id).long(), labels=vb['labels'])
                        val_loss += out.loss.item()
                        val_steps += 1
                if val_steps > 0:
                    avg_val_loss = val_loss / val_steps
                    val_perp = math.exp(avg_val_loss) if avg_val_loss < 50 else float('inf')
                    logger.info(f"Validation at step {global_step}: loss={avg_val_loss:.4f}, perp={val_perp:.2f}")
                    if run:
                        run.log({'validation/loss': avg_val_loss, 'validation/perplexity': val_perp, 'validation/step': global_step})
                model.train()

        # periodic save (long interval)
        if global_step % save_steps == 0:
            if use_ddp:
                dist.barrier()
                if RANK == 0:
                    save_path = f'./samantha-{args.model_size}-step-{global_step}'
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        model.module.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"Model saved to {save_path}")
                    except Exception:
                        torch.save(model.module.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"State dict saved to {save_path} (custom save fallback)")
                dist.barrier()
            else:
                save_path = f'./samantha-{args.model_size}-step-{global_step}'
                os.makedirs(save_path, exist_ok=True)
                try:
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Model saved to {save_path}")
                except Exception:
                    torch.save(model.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"State dict saved to {save_path} (custom save fallback)")

        if global_step >= total_steps:
            break

except KeyboardInterrupt:
    logger.info("Training interrupted by user. Saving emergency checkpoint...")
    if use_ddp:
        dist.barrier()
        if RANK == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(save_state, os.path.join(checkpoint_dir, 'model_state.pt'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
            if scaler is not None:
                torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, 'scaler_state.pt'))
            with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
                f.write(f"{global_step}\n")
            logger.info(f"Emergency checkpoint saved at step {global_step}")
        dist.barrier()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state.pt'))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pt'))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler_state.pt'))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, 'scaler_state.pt'))
        with open(os.path.join(checkpoint_dir, 'training_state.txt'), 'w') as f:
            f.write(f"{global_step}\n")
        logger.info(f"Emergency checkpoint saved at step {global_step}")

# final save
if use_ddp:
    dist.barrier()
    if RANK == 0:
        final_save_path = f'./samantha-{args.model_size}-final'
        os.makedirs(final_save_path, exist_ok=True)
        try:
            model.module.save_pretrained(final_save_path)
        except Exception:
            torch.save(model.module.state_dict(), os.path.join(final_save_path, 'pytorch_model.bin'))
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Final model saved to {final_save_path}")
    dist.barrier()
else:
    final_save_path = f'./samantha-{args.model_size}-final'
    os.makedirs(final_save_path, exist_ok=True)
    try:
        model.save_pretrained(final_save_path)
    except Exception:
        torch.save(model.state_dict(), os.path.join(final_save_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Final model saved to {final_save_path}")

if run:
    try:
        run.finish()
    except Exception:
        pass
