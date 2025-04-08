import torch
import time
from transformers.models.llama.modeling_llama import *
from modeling.compressor import OmniKVCompressorConfig
from modeling.spec_cache import DynamicSubCache
import os
import pickle
from tiny_tools.log import logger

last_call_t = time.time()


def time_analyze():
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


token_analyser = None
large_tokens_percentage = {}
cal_cnt = 0
if os.environ.get("CHECK_TOKENS", False):
    token_analyser = {
        "idx": {},
        "iou": {},
    }


def tensor_union(t1, t2):
    # 确保t1和t2是一维的
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    # 合并张量
    combined = torch.cat((t1_flat, t2_flat))
    # 使用unique去除重复值，注意sorted参数保持结果有序
    union_result = torch.unique(combined, sorted=True)
    return union_result


def tensor_intersection(t1, t2):
    # 确保t1和t2是一维的
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    # 计算交集，这里使用了isin方法来找出t1中也存在于t2中的元素
    intersection_result = torch.unique(t1_flat[torch.isin(t1_flat, t2_flat)])
    return intersection_result


def cal_iou(idx, iou):
    for i, a in idx.items():
        for j, b in idx.items():
            if f"{i} {j}" not in iou:
                iou[f"{i} {j}"] = []
            iou[f"{i} {j}"] += ([tensor_intersection(a, b).shape[0] / tensor_union(a, b).shape[0]])
    global cal_cnt
    cal_cnt += 1
    if cal_cnt % 100 == 0:
        with open('debug_logs/iou.pkl', 'wb') as _out:
            pickle.dump(iou, _out)


def count_big_tokens(attn_weights, layer_idx):
    if layer_idx not in large_tokens_percentage:
        large_tokens_percentage[layer_idx] = []
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
    large_tokens_percentage[layer_idx] += [(attn_weights > 0.01).sum().item() / attn_weights.numel()]
    if cal_cnt % 100 == 0:
        with open('debug_logs/big_tokens.pkl', 'wb') as _out:
            pickle.dump(large_tokens_percentage, _out)


class TokenConfig(OmniKVCompressorConfig):
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _rope_scaling_validation(self):
        logger.warning("为llama3.1做的patch，不验证了")
        return


def select_tokens_by_attn(raw_attn, hidden_states, position_ids, past_key_value: DynamicCache,
                          num_selected_tokens, layer_idx=None, ensure_head_tail=False):
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

    del key_states
    all_key_states = past_key_value.key_cache[raw_attn.layer_idx]
    all_key_states = all_key_states.split(32000, dim=-2)
    attn_weights = []
    for key_states in all_key_states:
        key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)
        _attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
        attn_weights += [_attn_weights]
    attn_weights = torch.cat(attn_weights, dim=-1)
    if os.environ.get("CHECK_TOKENS", False):
        count_big_tokens(attn_weights, layer_idx)

    attn_weights = torch.max(attn_weights[..., -1, :], dim=1).values
    if ensure_head_tail:
        attn_weights[:, :ensure_head_tail] += 10
        attn_weights[:, -ensure_head_tail:] += 10
    # attn_weights = attn_weights[..., -1, :]
    num_selected_tokens = min(num_selected_tokens, attn_weights.shape[-1])
    v, idx = torch.topk(attn_weights, k=num_selected_tokens, dim=-1)
    return idx


class TokenLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super(TokenLayer, self).__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.default_k = self.config.get('num_of_selected_tokens', 4096)
        self.ensure_ht = self.config.get('ensure_head_tail_len', False)
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
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]
        if past_key_value:
            assert isinstance(past_key_value, DynamicCache)
            if not isinstance(past_key_value, DynamicSubCache):
                past_key_value = DynamicSubCache.from_dynamic_cache(past_key_value)
        if hidden_states.shape[1] == 1 and past_key_value and \
                self.layer_idx >= self.config.get('decoder_start_layer_idx'):
            k = self.config.get(f'num_of_selected_tokens_{self.layer_idx}', self.default_k)
            if isinstance(k, float):
                k = int(k * self.prefill_len)
            idx = select_tokens_by_attn(self.self_attn, hidden_states, position_ids, past_key_value, k,
                                        layer_idx=self.layer_idx, ensure_head_tail=self.ensure_ht)
            if past_key_value:
                past_key_value.set_out_idx(idx, self.layer_idx)
            if os.environ.get("CHECK_TOKENS", False):
                token_analyser['idx'][self.layer_idx] = idx
                if self.layer_idx == self.config.num_hidden_layers - 1:
                    cal_iou(**token_analyser)

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
        hsl = torch.split(hidden_states, 4000, dim=1)
        hidden_states = [self.mlp(hs) for hs in hsl]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TokenModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TokenLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class TokenLM(LlamaForCausalLM):
    def __init__(self, config):
        ntk_factor = config.get('ntk_factor', -1)
        if ntk_factor > 0:
            logger.warning("在TokenLM中直接设置了rope_scaling")
            config.rope_scaling = {"type": "dynamic", "factor": config.get('ntk_factor', 8.0)}
        super().__init__(config)
        self.model = TokenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        n = input_ids.shape[1]
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

        hidden_states = outputs[0][:, -1:]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
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
            logger.info(f"---prefill time {round(time_analyze(), 3)}s")
        else:
            logger.info(f"---decoding time {round(time_analyze(), 3)}s")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
