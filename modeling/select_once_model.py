import os
import pickle

import torch
from transformers.models.llama.modeling_llama import *
from modeling.compressor import LlamaCompressorConfig
from modeling.spec_cache import DynamicSubCache


class TokenOnceConfig(LlamaCompressorConfig):
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def select_tokens_by_attn(raw_attn, hidden_states, position_ids, past_key_value: DynamicCache,
                          num_selected_tokens):
    bsz, q_len, _ = hidden_states.size()
    assert q_len == 1 and past_key_value

    if raw_attn.config.pretraining_tp > 1:
        key_value_slicing = (raw_attn.num_key_value_heads * raw_attn.head_dim) // raw_attn.config.pretraining_tp
        query_slices = raw_attn.q_proj.weight.split(
            (raw_attn.num_heads * raw_attn.head_dim) // raw_attn.config.pretraining_tp, dim=0
        )
        key_slices = raw_attn.k_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(raw_attn.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(raw_attn.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)
    else:
        query_states = raw_attn.q_proj(hidden_states)
        key_states = raw_attn.k_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, raw_attn.num_heads, raw_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim).transpose(1, 2)

    cos, sin = raw_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = past_key_value.key_cache[raw_attn.layer_idx]

    key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)

    attn_weights = torch.max(attn_weights[..., -1, :], dim=1).values
    # attn_weights = attn_weights[..., -1, :]
    v, idx = torch.topk(attn_weights, k=num_selected_tokens, dim=-1)
    return idx


if os.environ.get("CHECK_DISTRIB", False):
    num_cnt = 0
    pos_cnt = torch.zeros(int(1e5) + 100, device=0, dtype=torch.int64)


class TokenOnceLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super(TokenOnceLayer, self).__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.prefill_len = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        global num_cnt, pos_cnt
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if hidden_states.shape[1] > 1 and self.layer_idx == self.config.get('decoder_start_layer_idx'):
            if os.environ.get("CHECK_DISTRIB", False):
                if num_cnt > 0:
                    with open(f'debug_logs/rel_cnt/{num_cnt}.pkl', 'wb') as _out:
                        pickle.dump(pos_cnt, _out)
                num_cnt += 1
                pos_cnt = torch.zeros(int(1e5) + 100, device=0, dtype=torch.int64)

        if past_key_value:
            assert isinstance(past_key_value, DynamicCache)
            if not isinstance(past_key_value, DynamicSubCache):
                past_key_value = DynamicSubCache.from_dynamic_cache(past_key_value)
        if hidden_states.shape[1] == 1 and past_key_value and \
                self.layer_idx == self.config.get('decoder_start_layer_idx'):
            idx = select_tokens_by_attn(self.self_attn, hidden_states, position_ids, past_key_value,
                                        self.config.get('num_of_selected_tokens', 4096))
            for l in range(self.layer_idx, self.config.num_hidden_layers):
                past_key_value.set_out_idx(idx, l)
            if os.environ.get("CHECK_DISTRIB", False):
                assert idx.shape[0] == 1
                pos_cnt[idx.view(-1)] += 1

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TokenOnceModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TokenOnceLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class TokenOnceLM(LlamaForCausalLM):
    def __init__(self, config):
        if (fac := config.get("rope_factor", -1)) > 0:
            logger.warning("在TokenOnceLM中直接设置了rope_scaling")
            config.rope_scaling = {"type": "dynamic", "factor": fac}
        super().__init__(config)
        self.model = TokenOnceModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_context_len = config.get("max_context_len", 50_000)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # TODO major
        hidden_states = outputs[0][:, -1:]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        # TODO major
        # if logits.shape[1] == 1:
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
