import torch

from modeling.offload_select_once import *
from modeling.spec_cache import OmniKVMultiStageCache, WOPackCache
import logging as lgt


def select_tokens_by_attn_universal(
    raw_attn,
    hidden_states,
    position_ids,
    past_key_value,
    num_selected_tokens,
    consider_len,
    layer_idx=None,
    selector_cls="last",
):
    bsz, q_len, _ = hidden_states.size()
    assert past_key_value

    if raw_attn.config.get("pretraining_tp", -1) > 1:
        raise NotImplementedError
    else:
        query_states = raw_attn.q_proj(hidden_states)
        key_states = raw_attn.k_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, raw_attn.num_heads, raw_attn.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim
    ).transpose(1, 2)

    cos, sin = raw_attn.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = past_key_value.key_cache[raw_attn.layer_idx][:, :, :consider_len]
    key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)

    if selector_cls == "last":
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            raw_attn.head_dim
        )
        attn_score = torch.max(
            attn_score[..., -1, :], dim=1
        ).values  # remove query, then head
        num_selected_tokens = min(num_selected_tokens, attn_score.shape[-1])
        v, idx = torch.topk(attn_score, k=num_selected_tokens, dim=-1, sorted=True)
    elif selector_cls == "softmax_before_last":
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            raw_attn.head_dim
        )
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)  # to attn score
        attn_score = torch.max(
            attn_score[..., -1, :], dim=1
        ).values  # remove query, then head
        num_selected_tokens = min(num_selected_tokens, attn_score.shape[-1])
        v, idx = torch.topk(attn_score, k=num_selected_tokens, dim=-1, sorted=True)
    elif selector_cls == "uniform":
        qs = torch.split(query_states, 1, dim=2)
        first_flag = True
        attn_sum = None
        logger.debug(f"before = {torch.cuda.memory_allocated() / 1e9} GB")
        for q in qs:
            attn_score = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(
                raw_attn.head_dim
            )
            attn_score = torch.nn.functional.softmax(
                attn_score, dim=-1
            )  # to attn score
            attn_score = torch.max(attn_score, dim=1).values  # remove head dim
            attn_score = torch.sum(attn_score, dim=-2)  # remove query dim
            if first_flag:
                first_flag = False
                attn_sum = attn_score
            else:
                attn_sum += attn_score
        num_selected_tokens = min(num_selected_tokens, attn_sum.shape[-1])
        v, idx = torch.topk(attn_sum, k=num_selected_tokens, dim=-1, sorted=True)
        logger.debug(f"after = {torch.cuda.memory_allocated() / 1e9} GB")
    elif selector_cls == "exp":
        qs = torch.split(query_states, 1, dim=2)
        first_flag = True
        attn_sum = None
        for q in qs:
            attn_score = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(
                raw_attn.head_dim
            )
            attn_score = torch.nn.functional.softmax(
                attn_score, dim=-1
            )  # to attn score
            attn_score = torch.max(attn_score, dim=1).values  # remove head dim
            q_len = attn_score.shape[-2]
            alpha = (
                2
                ** torch.arange(-q_len + 1, 1, device=attn_score.device)[None, :, None]
            )
            attn_score = torch.sum(attn_score * alpha, dim=-2)  # remove query dim
            if first_flag:
                first_flag = False
                attn_sum = attn_score
            else:
                attn_sum = attn_sum * (2**-q_len) + attn_score
        num_selected_tokens = min(num_selected_tokens, attn_sum.shape[-1])
        v, idx = torch.topk(attn_sum, k=num_selected_tokens, dim=-1, sorted=True)
    else:
        raise NotImplementedError

    idx = torch.sort(idx, descending=False).values
    return idx


class OmniKVMulLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        config.rope_scaling = config.rope_scaling_
        try:
            super().__init__(config, layer_idx)
        except Exception:
            warning_once(logger, "ENSURE using Llama3.1!")
            config.rope_scaling = None
            super().__init__(config, layer_idx)
            self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
                config=config, layer_idx=layer_idx
            )
            config.rope_scaling = config.rope_scaling_
            self.self_attn.rotary_emb = PatchLlamaRotaryEmbedding(config=config)

        self.config = config
        self.layer_idx = layer_idx
        self.prefill_len = None
        self.cache_cls = get_cache_cls(config)
        self.sparse_in_prefill = config.get("sparse_in_prefill", False)
        self.max_len_can_hold = config.get("max_len_can_hold", 32_000)
        self.attn_seg_sz = config.get("attn_seg_sz", 8000)
        self.do_select_layers = [
            int(i) for i in config.get("do_select_layers").split(",")
        ]
        self.hidden_state_window = None
        self.selector_cls = config.get("selector_cls", "softmax_before_last")
        self.window_size = config.get("window_size", 16)
        self.decode_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[OmniKVMultiStageCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]
            if (
                "last" not in self.selector_cls
                and self.layer_idx in self.do_select_layers
            ):
                self.hidden_state_window = hidden_states[:, -self.window_size :]
                self.decode_step = 1  # 生成了一个token
        if past_key_value:
            assert isinstance(past_key_value, self.cache_cls)

        consider_len = self.prefill_len
        num_selected_tokens = self.config.get("num_of_selected_tokens", 4096)
        if isinstance(num_selected_tokens, float):
            num_selected_tokens = int(num_selected_tokens * consider_len)
        if (
            hidden_states.shape[1] == 1
            and past_key_value
            and self.layer_idx in self.do_select_layers
        ):
            window_hs = hidden_states
            num_prefill_token_in_window = max(0, self.window_size - self.decode_step)
            if "last" not in self.selector_cls:
                self.hidden_state_window = torch.cat(
                    [self.hidden_state_window, hidden_states], dim=1
                )[
                    :, -self.window_size :
                ]  # noqa
                window_hs = self.hidden_state_window
                consider_len -= num_prefill_token_in_window
                num_selected_tokens -= num_prefill_token_in_window
                num_selected_tokens = max(1, num_selected_tokens)
            idx = select_tokens_by_attn_universal(
                self.self_attn,
                window_hs,
                position_ids,
                past_key_value,
                num_selected_tokens,
                consider_len,
                self.layer_idx,
                self.selector_cls,
            )
            if "last" not in self.selector_cls:
                idx = torch.cat(
                    [
                        idx,
                        torch.arange(
                            self.prefill_len - num_prefill_token_in_window,
                            self.prefill_len,
                            device=idx.device,
                        )[None, :].repeat(idx.shape[0], 1),
                    ],
                    dim=1,
                )  # noqa
            if self.config.get("dense_more", False):
                past_key_value.set_idx_on_gpu(idx, self.layer_idx)
            else:
                raise ValueError("不支持dense_more=False")
            past_key_value.stage = "decoding"
            self.decode_step += 1

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


class OmniKVMulModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                OmniKVMulLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class OmniKVMulLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        if (fac := config.get("rope_factor", -1)) > 0:
            logger.warning("直接设置了rope_scaling")
            config.rope_scaling = {"type": "dynamic", "factor": fac}
        config.rope_scaling_ = config.rope_scaling
        config.rope_scaling = None
        super().__init__(config)
        self.model = OmniKVMulModel(config)
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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        n = input_ids.shape[1]
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if not isinstance(past_key_values, self.cache_cls):
            kwargs = {}
            if (
                cache_cls_name := self.config.get("cache_cls", "default")
            ) == "multi" or cache_cls_name == "without_pack":
                do_sel_layers = [
                    int(i) for i in self.config.get("do_select_layers").split(",")
                ]
                # TODO 这个地方还可以再考虑下
                full_attn_layers = (
                    list(range(0, do_sel_layers[0]))
                    + do_sel_layers
                    + [self.config.num_hidden_layers]
                )
                kwargs["full_attn_layers"] = full_attn_layers
                kwargs["num_hidden_layers"] = self.config.num_hidden_layers
                kwargs["num_wait_load_layers"] = self.config.get(
                    "num_wait_load_layers", 2
                )
                kwargs["real_offload"] = self.config.get("real_offload", True)
            else:
                raise NotImplementedError
            past_key_values = self.cache_cls.from_dynamic_cache(
                past_key_values, **kwargs
            )

        if n == 1:
            past_key_values.stage = "decoding"
        else:
            past_key_values.stage = "prefill"

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
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            raise NotImplementedError

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
