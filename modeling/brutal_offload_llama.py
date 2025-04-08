from transformers.models.llama.modeling_llama import *
from modeling.compressor import OmniKVCompressorConfig
from modeling.spec_cache import DynamicBrutalOffloadCache
import time
from tiny_tools.log import logger

last_call_t = time.time()
torch.set_num_threads(12)


def time_analyze():
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


class TokenOnceConfig(OmniKVCompressorConfig):
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if hidden_states.shape[1] > 1:
            self.prefill_len = hidden_states.shape[1]
        if past_key_value:
            assert isinstance(past_key_value, DynamicCache)
            if not isinstance(past_key_value, DynamicBrutalOffloadCache):
                past_key_value = DynamicBrutalOffloadCache.from_dynamic_cache(
                    past_key_value, self.config.get("offload_sid", 12), self.config.num_hidden_layers)
        if hidden_states.shape[1] == 1 and past_key_value and \
                self.layer_idx == self.config.get('decoder_start_layer_idx'):
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


class BrutalOffloadLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
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
            past_key_values: Optional[DynamicBrutalOffloadCache] = None,
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
            past_key_values = DynamicBrutalOffloadCache.from_dynamic_cache(past_key_values, self.config.offload_sid,
                                                                           self.config.num_hidden_layers)
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
