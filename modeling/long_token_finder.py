import torch
from transformers.models.llama.modeling_llama import *
from tiny_tools.log import logger, warning_once
from modeling.omnikv_config import LlamaCompressorConfig
from tiny_tools.tensor_tools import idx_tracer
import pickle
import os
import time
from modeling.patch_of_llama3_1 import PatchLlamaRotaryEmbedding
from modeling.spec_cache import SinkCache

last_call_t = time.time()


def time_analyze():
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


def read_cache(name, path="./cached_files/"):
    path = os.path.join(path, name)
    try:
        with open(path, "rb") as _input:
            d = pickle.load(_input)
        return d
    except FileNotFoundError:
        return {}


def write_cache(d, name, path="./cached_files/"):
    path = os.path.join(path, name)
    with open(path, "wb") as _out:
        pickle.dump(d, _out)


class EffLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

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
        hs = torch.split(hidden_states, 8000, dim=1)
        hidden_states = [self.mlp(h) for h in hs]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class EffModel(LlamaModel):
    def __init__(self, config):
        super(EffModel, self).__init__(config)
        self.layers = nn.ModuleList(
            [
                EffLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )


class LongTokenFinder(LlamaForCausalLM):
    _no_split_modules = ["EffLayer", "LlamaDecoderLayer"]

    def __init__(self, config: LlamaCompressorConfig):
        if config.get("rope_factor", -1) > 0:
            logger.warning("在LongTokenFinder中直接设置了rope_scaling")
            config.rope_scaling = {"type": "dynamic", "factor": 32.0}
        config.rope_scaling_ = config.rope_scaling
        config.rope_scaling = None
        self.config = config
        super(LongTokenFinder, self).__init__(config)
        self.model = EffModel(config)
        self.use_sink_cache = config.get("use_sink_cache", False)
        if self.use_sink_cache:
            print("使用Sink Cache")
        self.sink_window_length = config.get("sink_window_length", 1024)
        self.num_sink_tokens = config.get("num_sink_tokens", 128)

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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        if os.environ.get("SAVE_SELECTED_IDX", False):
            output_attentions = True
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        assert input_ids is not None
        n = input_ids.shape[1]
        if self.use_sink_cache and n > 1:
            window_len, num_sink_tokens = self.sink_window_length, self.num_sink_tokens
            if isinstance(window_len, float):
                window_len = int(n * window_len)
            if isinstance(num_sink_tokens, float):
                num_sink_tokens = int(n * num_sink_tokens)
            past_key_values = SinkCache(
                window_length=window_len, num_sink_tokens=num_sink_tokens
            )
        if self.use_sink_cache:
            assert isinstance(past_key_values, SinkCache)
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
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            # 主要修改
            loss_fct = CrossEntropyLoss(reduction="none")
            bs, seq_len = shift_labels.shape
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels).view(bs, seq_len)

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
