# eval.py

import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import gc

# ----------------------------
# Custom GPT-2 Model Definitions
# ----------------------------

class CustomGPT2Config(GPT2Config):
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
    CausalLMOutputWithCrossAttentions,
)

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

# ----------------------------
# Model and Tokenizer Loading
# ----------------------------

def load_model_and_tokenizer(model_path, device):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = CustomGPT2LMHeadModel.from_pretrained(model_path)

    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    return model, tokenizer

# ----------------------------
# Evaluation Functions
# ----------------------------

def evaluate_on_hellaswag(model, tokenizer, device, temperature=1.0, batch_size=8, block_size=512):
    print("\nStarting HellaSwag Evaluation...\n")
    # Load HellaSwag validation dataset without streaming
    hellaswag_dataset = load_dataset("Rowan/hellaswag", split="validation")

    total = 0
    correct = 0
    num_examples = len(hellaswag_dataset)
    num_choices = 4  # HellaSwag has 4 choices per question

    for i in tqdm(range(0, num_examples, batch_size), desc="Evaluating HellaSwag"):
        batch = hellaswag_dataset[i:i+batch_size]

        # Extract data from the batch
        contexts = batch['ctx']  # Updated from 'context' to 'ctx'
        endings_list = batch['endings']
        labels = batch['label']
        labels = [int(l) for l in labels]  # Convert labels from strings to integers

        inputs = []
        for context, endings in zip(contexts, endings_list):
            # For each ending, concatenate it with the context
            for ending in endings:
                input_text = context + ' ' + ending
                inputs.append(input_text)

        # Tokenize the concatenated inputs
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=block_size
        )

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        # Compute loss for each input
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits / temperature

            # Shift logits and labels to align them
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Compute loss per token and then sum to get loss per example
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss_per_token = loss_per_token.view(shift_labels.size())
            loss_per_example = loss_per_token.sum(dim=1)

            # Reshape losses to [batch_size, num_choices]
            actual_batch_size = len(batch['ctx'])
            loss_per_example = loss_per_example.view(actual_batch_size, num_choices)

            # Choose the option with the lowest loss (highest likelihood)
            predicted_labels = torch.argmin(loss_per_example, dim=1)
            labels_tensor = torch.tensor(labels, device=device)

            total += actual_batch_size
            correct += (predicted_labels == labels_tensor).sum().item()

    accuracy = correct / total
    print(f"\nHellaSwag Accuracy: {accuracy:.4f}\n")
    return accuracy

def evaluate_on_mmlu(model, tokenizer, device, temperature=1.0, batch_size=8, block_size=512, task='abstract_algebra'):
    print(f"\nStarting MMLU Evaluation on task: {task}\n")
    # Load MMLU dataset with the correct subtask name
    try:
        mmlu_dataset = load_dataset(
            "cais/mmlu",
            task,  # Use the task parameter
            split="test",
            trust_remote_code=True
        )
        print(f"Loaded dataset for task: {task} with {len(mmlu_dataset)} examples.")
    except Exception as e:
        print(f"Error loading MMLU task '{task}': {e}")
        return None

    total = 0
    correct = 0
    num_examples = len(mmlu_dataset)
    num_choices = 4  # MMLU has 4 choices per question

    for i in tqdm(range(0, num_examples, batch_size), desc=f"Evaluating MMLU ({task})"):
        batch = mmlu_dataset[i:i + batch_size]

        # Extract data from the batch
        questions = batch['question']
        choices_list = batch['choices']
        answers = batch['answer']  # This should be integer indices (0,1,2,3)

        inputs = []
        for question, choices in zip(questions, choices_list):
            for choice in choices:
                input_text = f"{question} {choice}"
                inputs.append(input_text)

        # Tokenize the concatenated inputs
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=block_size
        )

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        # Compute loss for each input
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits / temperature

            # Shift logits and labels to align them
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Compute loss per token and then sum to get loss per example
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss_per_token = loss_per_token.view(shift_labels.size())
            loss_per_example = loss_per_token.sum(dim=1)

            # Reshape losses to [batch_size, num_choices]
            actual_batch_size = len(batch['question'])
            loss_per_example = loss_per_example.view(actual_batch_size, num_choices)

            # Choose the option with the lowest loss (highest likelihood)
            predicted_labels = torch.argmin(loss_per_example, dim=1)
            labels_tensor = torch.tensor(answers, device=device)

            total += actual_batch_size
            correct += (predicted_labels == labels_tensor).sum().item()

    accuracy = correct / total
    print(f"\nMMLU Accuracy on '{task}': {accuracy:.4f}\n")
    return accuracy


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = "gpt3-small-fineweb-sft"  # Replace with your model path
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEMPERATURE = 1.0
    BATCH_SIZE = 8
    BLOCK_SIZE = 512

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DEVICE)

    # Evaluate on HellaSwag
    hellaswag_accuracy = evaluate_on_hellaswag(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE
    )

    # Evaluate on MMLU
    # Specify the MMLU task you want to evaluate on
    # Example tasks: 'abstract_algebra', 'college_biology', etc.
    """MMLU_TASK = 'abstract_algebra'  # Change as needed

    mmlu_accuracy = evaluate_on_mmlu(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        task=MMLU_TASK
    )"""

    # Optional: Evaluate on multiple MMLU tasks
    # Uncomment the following lines to evaluate on all available MMLU tasks
    
    mmlu_tasks = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    for task in mmlu_tasks:
        mmlu_accuracy = evaluate_on_mmlu(
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            task=task
        )
        gc.collect()

    print("Evaluation Completed.")
    print(f"HellaSwag Accuracy: {hellaswag_accuracy:.4f}")