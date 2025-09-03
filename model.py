import torch
import torch.nn as nn
from transformers import GPT2Config
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
        # armazena no objeto config para uso pelos blocos
        self.use_pre_layernorm = use_pre_layernorm


class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False):
        # chama o init pai — ele já cria c_attn/c_proj etc.
        super().__init__(config, is_cross_attention=is_cross_attention)
        # se quiser garantir bias explicitamente (o pai pode já ter):
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
        # chama o forward do pai explicitamente com os params necessários
        return super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class CustomGPT2MLP(GPT2MLP):
    # mantive mesma assinatura do pai (intermediate_size, config)
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        # redefino com Linear simples (compatível)
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        # função de ativação
        self.act = nn.GELU()


class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        # usa flag do config
        self.use_pre_layernorm = getattr(config, "use_pre_layernorm", True)
        # re-defino normais e sub-blocos para garantir comportamento
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CustomGPT2Attention(config)
        # mlp com fator 4x como padrão
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
        """
        Retorna similar ao GPT2Block: (hidden_states, present, attentions)
        Quando use_cache=True, 'present' (past_key_values) estará presente.
        """

        outputs = ()  # placeholder para extras (present, attentions)
        if self.use_pre_layernorm:
            # Pre-LayerNorm estilo Transformer-PreNorm
            residual = hidden_states
            hidden_states_norm = self.ln_1(hidden_states)

            attn_outputs = self.attn(
                hidden_states_norm,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            extras = attn_outputs[1:]  # (present, attentions) possivelmente

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states_norm = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states_norm)
            hidden_states = residual + feed_forward_hidden_states

            outputs = extras
        else:
            # Post-LayerNorm (estilo GPT-2 original)
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
            extras = attn_outputs[1:]

            hidden_states = residual + attn_output
            hidden_states = self.ln_1(hidden_states)

            residual = hidden_states
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
            hidden_states = self.ln_2(hidden_states)

            outputs = extras

        # outputs contém possivelmente (present, attentions)
        if use_cache:
            return (hidden_states,) + outputs
        else:
            # se outputs = (present, attentions) mas use_cache=False,
            # queremos devolver (hidden_states, None, attentions) compatível
            if len(outputs) == 2:
                _, attentions = outputs
                return (hidden_states, None, attentions)
            elif len(outputs) == 1:
                return (hidden_states, None, outputs[0])
            else:
                return (hidden_states,)


class CustomGPT2Model(GPT2Model):
    def __init__(self, config: CustomGPT2Config):
        # chama init do pai (isso configura buffers como position_ids, etc)
        super().__init__(config)
        # re-defino módulos (mesmos nomes do GPT2Model original para compatibilidade)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CustomGPT2Block(config) for _ in range(config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # inicialização padrão do transformers
        self.post_init()


class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: CustomGPT2Config):
        super().__init__(config)
        # substitui o transformer por nossa versão custom
        self.transformer = CustomGPT2Model(config)
        # lm_head: tipicamente weight-tied com wte; criamos como Linear sem bias
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # faz weight-tying
        try:
            self.lm_head.weight = self.transformer.wte.weight
        except Exception:
            # fallback: se algo não permitir tie imediato, ignore (não crítico)
            pass

        # inicialização padrão
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # garantimos que transformer retorne um objeto (return_dict=True)
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

        # último estado oculto
        # note: attribute name standard é `last_hidden_state`
        hidden_states = transformer_outputs.last_hidden_state

        # logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # shift para causal language modeling
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # constrói o objeto de saída com os campos esperados
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=getattr(transformer_outputs, "past_key_values", None),
            hidden_states=getattr(transformer_outputs, "hidden_states", None),
            attentions=getattr(transformer_outputs, "attentions", None),
            cross_attentions=getattr(transformer_outputs, "cross_attentions", None),
        )
