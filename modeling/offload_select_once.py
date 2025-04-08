import os

import torch
from transformers.models.llama.modeling_llama import *
from modeling.compressor import OmniKVCompressorConfig
from modeling.spec_cache import get_cache_cls
from modeling.patch_of_llama3_1 import PatchLlamaRotaryEmbedding
import time
from tiny_tools.log import logger, warning_once
from tiny_tools.tensor_tools import idx_tracer

last_call_t = time.time()


def time_analyze():
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


class TokenOnceConfig(OmniKVCompressorConfig):
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TempKVCache(Cache):  # noqa
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k = key_states
            self.v = value_states
            return key_states, key_states
        self.k = torch.cat([self.k, key_states], dim=-2)
        self.v = torch.cat([self.v, value_states], dim=-2)
        return self.k, self.v


def add_front_and_tail_tokens(idx, kv_seq_len, add_len=128):
    # will add tokens with size=add_len*2
    bs, raw_len = idx.shape[0], idx.shape[1]
    idx = torch.cat([
        torch.arange(0, add_len, device=idx.device)[None, :].repeat(bs, 1), idx,
        torch.arange(kv_seq_len - add_len, kv_seq_len, device=idx.device)[None, :].repeat(bs, 1)
    ], dim=-1).unique(dim=-1)
    logger.debug(f"actually add tokens = {idx.shape[1] - raw_len}")
    return idx


def analyze_prefill(raw_attn, hidden_states, position_ids):
    bsz, q_len, _ = hidden_states.size()
    assert q_len > 1
    assert os.environ.get('SAVE_SELECTED_IDX', False)

    if raw_attn.config.get("pretraining_tp", -1) > 1:
        raise NotImplementedError

    query_states = raw_attn.q_proj(hidden_states)
    key_states = raw_attn.k_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, raw_attn.num_heads, raw_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim).transpose(1, 2)

    cos, sin = raw_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)
    qs = torch.split(query_states, 10, dim=2)
    attn_sum = None
    for q in qs:
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
        attn_score = torch.nn.functional.softmax(attn_weights.float(), dim=-1)
        attn_score = torch.max(attn_score, dim=1).values  # remove head
        attn_score = torch.sum(attn_score, dim=-2) / q_len  # remove query dim
        if attn_sum is None:
            attn_sum = attn_score
        else:
            attn_sum += attn_score
    _v, _idx = torch.sort(attn_sum, dim=-1, descending=True)
    idx_tracer.append_prefill(_idx, _v, raw_attn.layer_idx)


def select_tokens_by_attn(raw_attn, hidden_states, position_ids, past_key_value,
                          num_selected_tokens, prefill_len, layer_idx=None):
    bsz, q_len, _ = hidden_states.size()
    assert q_len == 1 and past_key_value

    if raw_attn.config.get("pretraining_tp", -1) > 1:
        raise NotImplementedError
    else:
        query_states = raw_attn.q_proj(hidden_states)
        key_states = raw_attn.k_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, raw_attn.num_heads, raw_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim).transpose(1, 2)

    cos, sin = raw_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = past_key_value.key_cache[raw_attn.layer_idx][:, :, :prefill_len]

    key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)

    attn_weights = torch.max(attn_weights[..., -1, :], dim=1).values
    num_selected_tokens = min(num_selected_tokens, attn_weights.shape[-1])
    v, idx = torch.topk(attn_weights, k=num_selected_tokens, dim=-1, sorted=True)
    # 查看模型如何选择token
    if os.environ.get("SAVE_SELECTED_IDX", False) and layer_idx is not None:
        attn_score = torch.nn.functional.softmax(attn_weights, dim=-1)
        _v, _idx = torch.sort(attn_score, dim=-1, descending=True)
        idx_tracer.append_spec_layer(_idx, _v, prefill_len, raw_attn.layer_idx)

    idx = torch.sort(idx, descending=False).values
    return idx


class TokenOnceLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        config.rope_scaling = config.rope_scaling_
        try:
            super().__init__(config, layer_idx)
        except Exception:
            warning_once(logger, "ENSURE using Llama3.1!")
            config.rope_scaling = None
            super().__init__(config, layer_idx)
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
            config.rope_scaling = config.rope_scaling_
            self.self_attn.rotary_emb = PatchLlamaRotaryEmbedding(config=config)

        self.config = config
        self.layer_idx = layer_idx
        self.prefill_len = None
        self.cache_cls = get_cache_cls(config)
        self.sparse_in_prefill = config.get('sparse_in_prefill', False)
        self.max_len_can_hold = config.get('max_len_can_hold', 32_000)
        self.attn_seg_sz = config.get('attn_seg_sz', 8000)

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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]
            # 只在分析prefill时用
            if os.environ.get('SAVE_SELECTED_IDX', False) and \
                    self.layer_idx == self.config.get('decoder_start_layer_idx'):
                analyze_prefill(self.self_attn, hidden_states, position_ids)
        if past_key_value:
            assert isinstance(past_key_value, self.cache_cls)
        # 分析idx的选择
        if hidden_states.shape[1] == 1 and os.environ.get('SAVE_SELECTED_IDX', False):
            # print(f"analyse layer{self.layer_idx}")
            _idx = select_tokens_by_attn(self.self_attn, hidden_states, position_ids, past_key_value,
                                         self.config.get('num_of_selected_tokens', 4096), self.prefill_len,
                                         layer_idx=self.layer_idx)
        if hidden_states.shape[1] == 1 and past_key_value and \
                self.layer_idx == self.config.get('decoder_start_layer_idx'):
            idx = select_tokens_by_attn(self.self_attn, hidden_states, position_ids, past_key_value,
                                        self.config.get('num_of_selected_tokens', 4096), self.prefill_len)
            if self.config.get('dense_more', False):
                past_key_value.set_out_idx(idx, self.config.get('offload_sid'))
            else:
                past_key_value.set_out_idx(idx, self.layer_idx)
            past_key_value.stage = 'decoding'

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
        hs = torch.split(hidden_states, 2000, dim=1)
        hidden_states = [self.mlp(h) for h in hs]
        hidden_states = torch.cat(hidden_states, dim=1)
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


class TokenOnceOffloadLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        if (fac := config.get("rope_factor", -1)) > 0:
            logger.warning("在TokenOnceLM中直接设置了rope_scaling")
            config.rope_scaling = {"type": "dynamic", "factor": fac}
        config.rope_scaling_ = config.rope_scaling
        config.rope_scaling = None
        super().__init__(config)
        self.model = TokenOnceModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_context_len = config.get("max_context_len", 50_000)
        self.cache_cls = get_cache_cls(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
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
        n = input_ids.shape[1]
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if self.config.get('cache_cls', 'default') == 'eff':
                past_key_values = self.cache_cls.from_dynamic_cache(past_key_values, self.config)
            else:
                kwargs = {}
                if self.config.get("cache_cls", 'default') == 'lazy':
                    kwargs['skip_threshold'] = self.config.get('skip_threshold', 0.4)
                past_key_values = self.cache_cls.from_dynamic_cache(past_key_values, self.config.offload_sid,
                                                                    self.config.num_hidden_layers, **kwargs)
        if n == 1:
            past_key_values.stage = 'decoding'

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

        hidden_states = outputs[0][:, -1:]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
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

        if n > 1:
            logger.info(f"---prefill time {time_analyze()}")
        else:
            logger.info(f"---decoding time {time_analyze()}")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
