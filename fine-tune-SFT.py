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

import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"
#os.environ["WANDB_OFFLINE"]="1" # if you want to run offline

# ---------------------------
# Initialize Weights & Biases (W&B / Wandb)
# ---------------------------
wandb.init(project='gpt3-small-fineweb', name='sft-no_robots')

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
model = CustomGPT2LMHeadModel.from_pretrained('gpt3-small-fineweb', config=config)
print("Fine-tuning model with" , model.num_parameters(), "parameters")
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# ---------------------------
# Load and Prepare the no_robots Dataset
# ---------------------------
no_robots_train_dataset = load_dataset("HuggingFaceH4/no_robots", split="train")
no_robots_eval_dataset = load_dataset("HuggingFaceH4/no_robots", split="test")

# Preprocess the dataset
def preprocess_no_robots(sample, idx):
    """
        Formats an example for SFT with dynamic sequence length.
        First, tokenize the entire conversation without truncation, compute labels,
        and then truncate the sequence to max_length.
    """
    max_length = 1024  # can be 2048 for other dataaset but now it works fine
    messages = sample["messages"]
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
processed_no_robots_train = no_robots_train_dataset.map(
    preprocess_no_robots,
    remove_columns=no_robots_train_dataset.column_names,
    with_indices=True,
)

processed_no_robots_eval = no_robots_eval_dataset.map(
    preprocess_no_robots,
    remove_columns=no_robots_eval_dataset.column_names,
    with_indices=True,
)

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
    per_device_train_batch_size=12,       # Adjust based on GPU memory
    per_device_eval_batch_size=12,        # Adjust based on GPU memory
    gradient_accumulation_steps=4,        # To simulate larger batch sizes
    eval_strategy="steps",
    bf16=True,
    lr_scheduler_type="cosine", ###
    optim="adamw_hf", #adamw_torch
    remove_unused_columns=True,
    #deepspeed=True, ###
    eval_on_start=True,
    eval_steps=50,                      # Evaluate every 500 steps
    #save_steps=200,                     # Save model every 1000 steps
    logging_steps=1,                   # Log metrics every 100 steps
    learning_rate=1e-4, #3e-5                 # Lower learning rate for fine-tuning
    warmup_steps=200,                    #500# Warmup steps for scheduler
    save_total_limit=3,                  # Limit the total number of saved models
    report_to=['wandb'],                 # Enable logging to W&B
    run_name='sft-training-no_robots',
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


print(processed_no_robots_train[0]) ###
# ---------------------------
# Initialize the Trainer
# ---------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_no_robots_train,
    eval_dataset=processed_no_robots_eval,
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
final_save_path = 'gpt3-small-fineweb-no_robots'
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