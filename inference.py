import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

import torch
import logging
from transformers import GPT2TokenizerFast, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch.nn as nn
import math

# Custom Configuration and Model Classes (must match those used during training)
class CustomGPT2Config(GPT2Config):
    #model_type = "gpt3dev"

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

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,        # Added this parameter
        encoder_attention_mask=None,       # Added this parameter
        use_cache=False,
        output_attentions=False,
    ):
        return super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,      # Pass this parameter
            encoder_attention_mask=encoder_attention_mask,    # Pass this parameter
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

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
        encoder_hidden_states=None,        # Added this parameter
        encoder_attention_mask=None,       # Added this parameter
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
                encoder_hidden_states=encoder_hidden_states,      # Pass this parameter
                encoder_attention_mask=encoder_attention_mask,    # Pass this parameter
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
                encoder_hidden_states=encoder_hidden_states,      # Pass this parameter
                encoder_attention_mask=encoder_attention_mask,    # Pass this parameter
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

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            # Check if past_key_values has the expected structure
            if len(past_key_values) > 0 and past_key_values[0] is not None and len(past_key_values[0]) > 0 and past_key_values[0][0] is not None:
                past_length = past_key_values[0][0].size(-2)
            else:
                past_length = 0
                past_key_values = tuple([None] * len(self.h))

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Prepare attention mask
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logging.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


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
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
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

# Helper function for top-k and nucleus (top-p) sampling
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), min_p=0.05):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    logits = logits.clone()

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:  # Additional safety check
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

    # Nucleus (top-p) filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to include the token that crosses the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    if min_p == 0.0:
        return logits
    else:
        # min_p
        probabilities = torch.softmax(logits, dim=-1)
        low_prob_mask = probabilities < min_p
        logits[low_prob_mask] = filter_value

    # Final safety check: ensure at least one logit is finite
    if not torch.isfinite(logits).any():
        # Reset the highest logit to prevent all being -inf
        max_idx = torch.argmax(logits)
        logits[max_idx] = 0.0  # Reset to neutral value

    return logits

def generate_text_stream(prompt, model, tokenizer, max_length, temperature, top_p, top_k, min_p, repetition_penalty=1.2, no_repeat_ngram_size=3, streaming=True):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    past_key_values = None
    generated_text = prompt
    generated_tokens = input_ids.tolist()[0]

    print("\n\n" + prompt, end='', flush=True)

    max_new_tokens = max_length - input_ids.shape[-1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask if past_key_values is None else attention_mask[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(generated_tokens):
                next_token_logits[0, token_id] /= repetition_penalty

            # Enforce no_repeat_ngram_size
            if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size:
                ngram_list = [tuple(generated_tokens[i:i + no_repeat_ngram_size]) for i in range(len(generated_tokens) - no_repeat_ngram_size + 1)]
                last_ngram = tuple(generated_tokens[-(no_repeat_ngram_size - 1):])

                banned_tokens = [ngram[-1] for ngram in ngram_list if ngram[:-1] == last_ngram]
                next_token_logits[0, banned_tokens] = -float('inf')

            # Apply top_k and top_p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, min_p=min_p)

            # Safety check: ensure at least one logit is finite
            if not torch.isfinite(filtered_logits).any():
                # If all logits are -inf, reset to original logits
                filtered_logits = next_token_logits.clone()

            probabilities = torch.softmax(filtered_logits, dim=-1)

            # Additional safety: check for invalid probabilities
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                # Fallback to uniform distribution over non-inf logits
                valid_mask = torch.isfinite(next_token_logits)
                if valid_mask.any():
                    probabilities = torch.zeros_like(next_token_logits)
                    probabilities[valid_mask] = 1.0 / valid_mask.sum()
                else:
                    # Last resort: uniform over all tokens
                    probabilities = torch.ones_like(next_token_logits) / next_token_logits.numel()

            # Sample the next token
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Append the new token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
            generated_tokens.append(next_token.item())

            past_key_values = outputs.past_key_values

            # Decode and print the new token
            generated_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
            if streaming == True:
                print(generated_token, end='', flush=True)
            generated_text += generated_token

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated_text

# Main script
def main():
    model_path = "./samantha-17m-gpt2-rtx3050-step-304139" # or a local path
    
    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the custom configuration
    config = CustomGPT2Config.from_pretrained(model_path)

    # Load the model
    model = CustomGPT2LMHeadModel.from_pretrained(model_path, config=config)
    model.eval()

    # Move the model to the appropriate device
    device = torch.device("cpu") # interesting, cpu inference is faster than mps on m2 macbook air, change for "cuda", "rocm", etc
    model.to(device)
    
    # Get user input
    prompt = input(f"Enter a prompt: ")
    # Generate text
    generate_text_stream( #generated_text = 
        prompt,
        model,
        tokenizer,
        max_length=512,
        temperature=0.1, #0.1 is great for pre-trained only model
        repetition_penalty=2.0,
        no_repeat_ngram_size=2,
        top_p=0.95,
        min_p=0.05,
        top_k=50, #200 is too much but works very good with model from H100, 50 is good, should try 20
        streaming=True,
    )
    print("\n\n")
if __name__ == "__main__":
    main()
