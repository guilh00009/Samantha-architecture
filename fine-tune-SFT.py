# Install necessary libraries (uncomment if not already installed)
# !pip install transformers datasets wandb evaluate

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
from datasets import load_dataset
import wandb
from itertools import islice, cycle
import evaluate

# ---------------------------
# Initialize Weights & Biases (W&B / Wandb)
# ---------------------------
wandb.init(project='gpt3-small-fineweb-sft', name='sft-training')

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
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,  # Pass additional arguments
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


# ---------------------------
#  Tokenizer and Model
# ---------------------------
tokenizer = GPT2TokenizerFast.from_pretrained('./gpt3-small-fineweb')  # pre-trained model
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config.from_pretrained('./gpt3-small-fineweb')

model = CustomGPT2LMHeadModel.from_pretrained('./gpt3-small-fineweb', config=config)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# Load and Prepare the Dataset
# ---------------------------
helpsteer_train_dataset = load_dataset("nvidia/HelpSteer", split="train")
helpsteer_eval_dataset = load_dataset("nvidia/HelpSteer", split="validation")

def preprocess(sample):
    """
    Concatenates the prompt and response, tokenizes them, and sets labels to -100 for prompt tokens
    to ignore them during loss computation.
    """
    # Concatenate prompt and response with eos_token as separator
    concatenated = sample["prompt"] + tokenizer.eos_token + sample["response"] + tokenizer.eos_token
    
    # Tokenize the concatenated text
    tokenized = tokenizer(
        concatenated,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"].squeeze()
    attention_mask = tokenized["attention_mask"].squeeze()
    
    # length of the prompt
    prompt_length = len(tokenizer(sample["prompt"], truncation=True, max_length=512)["input_ids"])
    
    # Init labels with -100 (ignore)
    labels = torch.full(input_ids.shape, -100)
    
    # Tokenize the response to find its length
    response_tokenized = tokenizer(sample["response"], truncation=True, max_length=512)
    response_length = len(response_tokenized["input_ids"])
    
    # Set labels for the response tokens
    labels[prompt_length + 1 : prompt_length + 1 + response_length] = input_ids[prompt_length + 1 : prompt_length + 1 + response_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

"""def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.tensor(logits).argmax(dim=-1)
        # Compute Cross-Entropy Loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(torch.tensor(logits).view(-1, logits.shape[-1]), torch.tensor(labels).view(-1))
        perplexity = math.exp(loss.item())
        return {"eval_perplexity": perplexity}      #caused an error, so I commented it out.
"""

processed_train_dataset = helpsteer_train_dataset.map(
    preprocess,
    remove_columns=helpsteer_train_dataset.column_names
)

processed_eval_dataset = helpsteer_eval_dataset.map(
    preprocess,
    remove_columns=helpsteer_eval_dataset.column_names
)

# ---------------------------
# Training Args for SFT
# ---------------------------
training_args = TrainingArguments(
    output_dir='./outputs-sft',
    overwrite_output_dir=True,
    lr_scheduler_type="cosine",
    num_train_epochs=3,                  # Train for 3 epochs
    per_device_train_batch_size=4,       # Adjust based on GPU memory
    per_device_eval_batch_size=8,        # Adjust based on GPU memory
    gradient_accumulation_steps=4,       # To simulate larger batch sizes
    eval_strategy="steps",
    eval_on_start=True,
    eval_steps=500,                      # Evaluate every 500 steps
    save_steps=1000,                     # Save model every 1000 steps
    logging_steps=100,                   # Log metrics every 100 steps
    learning_rate=3e-5,                  # Lower learning rate because we fine-tune
    warmup_steps=500,                    # Warmup steps for scheduler
    save_total_limit=3,                  # Limit the total number of saved models
    report_to=['wandb'],                 # Logging to W&B
    run_name='sft-training',
    load_best_model_at_end=True,         # Load the best model when finished training
    metric_for_best_model='eval_loss',   # here we use eval_loss or eval_perplexity (don't forget to uncomment compute_metrics)
    greater_is_better=False,
)

# ---------------------------
# Initialize the Data Collator and Trainer
# ---------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False for causal language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)

# ---------------------------
# Callback to log some sequences during the run
# ---------------------------
class GenerateTextCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step % 1000 == 0 and state.global_step != 0:
            prompt = "Once upon a time"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Pass attention_mask to generate
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                #early_stopping=True
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            wandb.log({"sample_text": wandb.Html(f"<p>{text}</p>"), "step": state.global_step}) # we log sample texts as html and send them to wandb

trainer.add_callback(GenerateTextCallback)

# ---------------------------
# Start Training
# ---------------------------
trainer.train()

final_save_path = './gpt3-small-fineweb-sft'
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final model saved to {final_save_path}")

wandb.finish()
