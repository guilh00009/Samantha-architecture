import math
import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

class SamanthaConfig(GPT2Config):
    model_type = "samantha"
    
    def __init__(self, use_pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pre_layernorm = use_pre_layernorm
        self.d_head = kwargs.get("d_head", self.n_embd // self.n_head)

# Custom GPT-2 with Pre-LayerNorm and Biases aka GPT-3
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
)

class SamanthaAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.hidden_size = config.n_embd
        self.scale = 1 / math.sqrt(self.d_head)

        self.c_attn = nn.Linear(self.hidden_size, 3 * self.n_head * self.d_head, bias=True)
        self.c_proj = nn.Linear(self.n_head * self.d_head, self.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_head, self.d_head)
        return x.transpose(1, 2)

    def merge_heads(self, x, batch_size):
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.n_head * self.d_head)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False):
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_head * self.d_head, dim=-1)
        query = self.split_heads(q, batch_size)
        key = self.split_heads(k, batch_size)
        value = self.split_heads(v, batch_size)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs, value)
        attn_output = self.merge_heads(attn_output, batch_size)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if use_cache:
            return attn_output, present, (attn_probs if output_attentions else None)
        else:
            return (attn_output, attn_probs) if output_attentions else (attn_output,)

class SamanthaMLP(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=True)
        self.act = nn.GELU()  # Use standard GeLU

class SamanthaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_pre_layernorm = config.use_pre_layernorm
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = SamanthaAttention(config)
        self.mlp = SamanthaMLP(4 * config.n_embd, config)
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

class SamanthaModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([SamanthaBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=None):
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )[0]

        hidden_states = self.ln_f(hidden_states)
        return (hidden_states,)

class SamanthaLMHeadModel(PreTrainedModel):
    config_class = SamanthaConfig

    def __init__(self, config):
        super().__init__(config)  # calls PreTrainedModel.__init__
        self.transformer = SamanthaModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()  # Initializes weights and registers modules for pretrain saving

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
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return (lm_logits,) if loss is None else (loss, lm_logits)

AutoConfig.register("samantha", SamanthaConfig)
AutoModel.register(SamanthaConfig, SamanthaModel)
AutoModelForCausalLM.register(SamanthaConfig, SamanthaLMHeadModel)
