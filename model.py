import torch
from transformers import GPT2TokenizerFast, GPT2Config
import torch.nn as nn
import math
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
    CausalLMOutputWithCrossAttentions,
)

class CustomGPT2Config(GPT2Config):
    model_type = "custom_gpt2"

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
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

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
        # Prepare arguments for parent forward call
        forward_kwargs = {
            'hidden_states': hidden_states,
            'layer_past': layer_past,
            'attention_mask': attention_mask,
            'head_mask': head_mask,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
        }

        # Only add encoder parameters if they are not None
        if encoder_hidden_states is not None:
            forward_kwargs['encoder_hidden_states'] = encoder_hidden_states
        if encoder_attention_mask is not None:
            forward_kwargs['encoder_attention_mask'] = encoder_attention_mask

        return super().forward(**forward_kwargs)

class CustomGPT2MLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()  # Use standard GeLU

class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        self.use_pre_layernorm = config.use_pre_layernorm
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        self.mlp = CustomGPT2MLP(4 * config.hidden_size, config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

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

            # Prepare attention arguments
            attn_kwargs = {
                'hidden_states': hidden_states,
                'layer_past': layer_past,
                'attention_mask': attention_mask,
                'head_mask': head_mask,
                'use_cache': use_cache,
                'output_attentions': output_attentions,
            }

            # Only add encoder parameters if they are not None
            if encoder_hidden_states is not None:
                attn_kwargs['encoder_hidden_states'] = encoder_hidden_states
            if encoder_attention_mask is not None:
                attn_kwargs['encoder_attention_mask'] = encoder_attention_mask

            attn_outputs = self.attn(**attn_kwargs)
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

            # Prepare attention arguments
            attn_kwargs = {
                'hidden_states': hidden_states,
                'layer_past': layer_past,
                'attention_mask': attention_mask,
                'head_mask': head_mask,
                'use_cache': use_cache,
                'output_attentions': output_attentions,
            }

            # Only add encoder parameters if they are not None
            if encoder_hidden_states is not None:
                attn_kwargs['encoder_hidden_states'] = encoder_hidden_states
            if encoder_attention_mask is not None:
                attn_kwargs['encoder_attention_mask'] = encoder_attention_mask

            attn_outputs = self.attn(**attn_kwargs)
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
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Initialize weights
        self.post_init()

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
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
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

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