import os

import torch
import json
import time
import transformers
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
)
from typing import Dict, AnyStr
from tiny_tools.read_json import read_config
from tiny_tools.log import logger
from tiny_tools.tensor_tools import dict_to_cuda
from torch.cuda.amp import autocast
from argparse import ArgumentParser
from tiny_tools.tensor_tools import idx_tracer

if transformers.__version__ >= "4.40":
    from transformers import (
        BitsAndBytesConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        GPTQConfig,
    )
    from modeling.long_token_finder import LongTokenFinder
    from modeling.token_model import TokenLM, TokenConfig
    from modeling.select_once_model import TokenOnceLM, TokenOnceConfig
    from modeling.offload_select_once import TokenOnceOffloadLM
    from modeling.omnikv import OmniKVMulLM
    from modeling.qwen2_offload import Qwen2TokenOnceOffloadLM, QwenOffOnceConfig
    from modeling.brutal_offload_llama import BrutalOffloadLM
    from modeling.omnikv_config import LlamaCompressorConfig
    from modeling.qwen2_eff import Qwen2EffForCausalLM, Qwen2EffConfig
    from modeling.qwen2_offload_mul import Qwen2OmniKVMulLM
    from configs.template_for_chat import get_chat_template
    from baselines.infllm import get_infllm_api

    try:
        # TODO 加入quest
        import sys

        # 为了让里面的import可以找到对应位置
        sys.path.append("baselines/quest")
        from baselines.quest.quest.models import llama as quest_bsl
    except:
        pass
else:
    from baselines.raw_h2o import get_h2o_api

set_seed(42)
# old_prompt = "You are a pirate chatbot who always responds in pirate speak!"
# fixed_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
#
# {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
transformers.logging.set_verbosity_error()
if os.environ.get("BE_CAREFUL", False):
    transformers.logging.set_verbosity_debug()


def inference_bs1(
    prompt,
    tkn,
    model,
    generation_config=None,
    use_chat_template=False,
    model_name=None,
    use_fixed_prompt=False,
    use_cot=False,
    return_input_ids=False,
    **kwargs,
):
    st = time.time()
    with torch.no_grad():
        with autocast():
            terminators = [] + [tkn.eos_token_id] if tkn.eos_token_id else []
            for eos_token in ["<|eot_id|>", "<|endoftext|>", "<|im_end|>"]:
                if eos_token in tkn.vocab:
                    terminators += [tkn.convert_tokens_to_ids(eos_token)]
            if use_chat_template:
                template = get_chat_template(model_name, use_cot)
                prompt = template.format(
                    user_message=prompt, system_prompt="You are a helpful assistant."
                )
                # logger.debug(f"prompt is {prompt}")
                input_ids = tkn(prompt, return_tensors="pt")["input_ids"]
            else:
                input_ids = tkn(prompt, return_tensors="pt")["input_ids"]
            if return_input_ids:
                return input_ids
            n = input_ids.shape[1]
            temp = model.generate(
                input_ids.cuda(model.device),
                generation_config=generation_config,
                eos_token_id=terminators,
                **kwargs,
            )[:, n:]
            if os.environ.get("USE_TIMER", False):
                print(f"-------inference_bs1 time {round(time.time() - st, 4)} s")
            return temp


def get_ntk_llama_chat_api_with_tokenizer_bs1(config_path):
    d_config = read_config(config_path)
    model_name = d_config["model_name"]
    model_cls = d_config["model_cls"]
    tkn = AutoTokenizer.from_pretrained(model_name)
    device = 0
    if "qwen" in model_cls:
        cfg_cls = Qwen2EffConfig
    elif "llama" in model_cls or "262" in model_cls:
        cfg_cls = LlamaCompressorConfig
    else:
        raise ValueError

    cfg = cfg_cls.from_pretrained(model_name)
    if hasattr(cfg, "set_config_of_compressor"):
        cfg.set_config_of_compressor(**d_config)

    use_flash_attn = d_config.get("use_flash_attn", False)
    load_in_8bit = d_config.get("load_in_8bit", False)
    load_in_4bit = d_config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    args = [model_name]
    kwargs = {
        "config": cfg,
        "quantization_config": quant_config,
        # "pretrained_model_name_or_path": model_name
    }
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if quant_config is None:
        kwargs["torch_dtype"] = torch.float16
    if mul_gpu := getattr(cfg, "use_multi_gpus", False):
        kwargs["device_map"] = "auto"

    if "qwen" in model_cls:
        model = Qwen2EffForCausalLM.from_pretrained(*args, **kwargs)
    elif "llama" in model_cls or "262" in model_cls:
        model = LongTokenFinder.from_pretrained(*args, **kwargs)
    else:
        raise ValueError
    if not load_in_8bit and not load_in_4bit and not mul_gpu:
        model = model.cuda(device)

    if tkn.pad_token is None:
        # 处理一下新加入的pad token，直接设置为全0
        logger.warning("因为llama没有pad token，设置tokenizer.pad_token=[PAD]")
        tkn.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tkn))

    use_cot = cfg.get("cot", False)
    use_chat_template = getattr(cfg, "use_chat_template", False)
    use_fixed_prompt = getattr(cfg, "use_fixed_prompt", True)
    assert use_fixed_prompt
    if use_cot:
        assert use_chat_template

    def chat(prompt, generation_config=None, skip_special_tokens=False, **kwargs):
        st = time.time()
        out_ids = inference_bs1(
            prompt,
            tkn,
            model,
            generation_config,
            use_chat_template=use_chat_template,
            use_fixed_prompt=use_fixed_prompt,
            use_cot=use_cot,
            model_name=model_name,
            **kwargs,
        )
        out = tkn.batch_decode(
            out_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]
        logger.info(f"--- chat time: {round(time.time() - st, 3)}s")
        return out

    return chat, tkn, d_config["max_context_len"], {"eos_token_id": tkn.eos_token_id}


def get_token_select_llama_chat_api_with_tokenizer_bs1(config_path):
    config = read_config(config_path)
    cls = config.get("model_cls", "token")
    model_name = config["model_name"]

    cfg_cls = QwenOffOnceConfig if cls.startswith("qwen") else TokenConfig
    cfg = read_config(config_path)
    config = cfg_cls.from_pretrained(model_name)
    config.set_config(**cfg)

    device = 0

    use_flash_attn = config.get("use_flash_attn", False)
    # prepare quantization config
    load_in_8bit = config.get("load_in_8bit", False)
    load_in_4bit = config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        if config.get("use_gptq", False):
            quant_config = GPTQConfig(load_in_4bit)
            raise NotImplementedError
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    args = [model_name]
    kwargs = {"config": config, "quantization_config": quant_config}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if quant_config is None:
        kwargs["torch_dtype"] = torch.float16

    if cls == "token":
        model = TokenLM.from_pretrained(*args, **kwargs).cuda(device)
    elif cls == "token_once":
        model = TokenOnceLM.from_pretrained(*args, **kwargs).cuda(device)
    elif cls == "once_offload":
        logger.warning("只用于分析模型")
        model = TokenOnceOffloadLM.from_pretrained(*args, **kwargs)
        # raise ValueError("change config to 'multi'")
    elif cls == "brutal_offload":
        model = BrutalOffloadLM.from_pretrained(*args, **kwargs)
    elif cls == "qwen_once_offload":
        # model = Qwen2TokenOnceOffloadLM.from_pretrained(*args, **kwargs)
        raise ValueError("change config to 'qwen_multi'")
    elif cls == "multi":
        model = OmniKVMulLM.from_pretrained(*args, **kwargs)
    elif cls == "qwen_multi":
        model = Qwen2OmniKVMulLM.from_pretrained(*args, **kwargs)
    else:
        raise ValueError

    if not load_in_8bit and not load_in_4bit:
        model = model.cuda(device)

    tkn = AutoTokenizer.from_pretrained(model_name)
    if tkn.pad_token is None:
        logger.warning("因为llama没有pad token，设置tokenizer.pad_token=[PAD]")
        tkn.add_special_tokens({"pad_token": "[PAD]"})
        # 处理一下新加入的pad token，直接设置为全0
        model.resize_token_embeddings(len(tkn))

    use_cot = config.get("cot", False)
    use_chat_template = config.get("use_chat_template", False)
    use_fixed_prompt = config.get("use_fixed_prompt", True)
    # use_sink_cache = config.get('use_sink', False)
    # sink_window_length = config.get('sink_window_length', 1024)
    # num_sink_tokens = config.get('num_sink_tokens', 128)
    assert use_fixed_prompt
    if use_cot:
        assert use_chat_template

    def chat(prompt, generation_config=None, skip_special_tokens=False, **kwargs):
        st = time.time()
        # if use_sink_cache:
        #     kwargs['cache_implementation'] = 'sink'
        #     kwargs['cache_config'] = {'window_length': sink_window_length, 'num_sink_tokens': num_sink_tokens}
        tkn_ids = inference_bs1(
            prompt,
            tkn,
            model,
            generation_config,
            use_cot=use_cot,
            model_name=model_name,
            use_chat_template=use_chat_template,
            use_fixed_prompt=use_fixed_prompt,
            **kwargs,
        )
        res = tkn.batch_decode(
            tkn_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )[0]
        logger.info(f"--- chat time: {round(time.time() - st, 3)}s")
        return res

    o_dict = {}
    o_dict["eos_token_id"] = config.eos_token_id
    return chat, tkn, config.get("max_context_len", 100_000), o_dict


def get_infllm_chat_api_bs1(config_path):
    from baselines.infllm.inf_llm.utils.greedy_search import GreedySearch

    config = read_config(config_path)
    infllm_cfg_path = config["infllm_cfg_path"]
    model_name = config["model_name"]

    # prepare quantization config
    load_in_8bit = config.get("load_in_8bit", False)
    load_in_4bit = config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        if config.get("use_gptq", False):
            quant_config = GPTQConfig(load_in_4bit)
            raise NotImplementedError
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model, tkn = get_infllm_api.get_model_tokenizer_others(config_path, quant_config)

    if tkn.pad_token is None:
        logger.warning("因为llama没有pad token，设置tokenizer.pad_token=[PAD]")
        tkn.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tkn))

    searcher = GreedySearch(model, tkn)
    use_cot = config.get("cot", False)  # removed
    use_chat_template = config.get("use_chat_template", False)
    use_fixed_prompt = config.get("use_fixed_prompt", True)
    assert use_fixed_prompt
    if use_cot:
        assert use_chat_template

    def chat(prompt, generation_config=None, skip_special_tokens=False, **kwargs):
        st = time.time()
        input_ids = inference_bs1(
            prompt,
            tkn,
            model,
            generation_config,
            use_cot=use_cot,
            model_name=model_name,
            use_chat_template=use_chat_template,
            return_input_ids=True,
            **kwargs,
        )
        extra_end_token_ids = [
            tkn.convert_tokens_to_ids(t)
            for t in ["<|eot_id|>", "<|endoftext|>", "<|im_end|>"]
        ]
        res = searcher.generate(
            input_ids=input_ids, extra_end_token_ids=extra_end_token_ids
        )[0]
        searcher.clear()
        logger.info(f"--- chat time: {round(time.time() - st, 3)}s")
        return res

    o_dict = {"eos_token_id": config.get("eos_token_id", None)}
    return chat, tkn, config.get("max_context_len", 100_000), o_dict


def get_quest_chat_api(cfg_path):
    raise NotImplementedError


def get_any_chat_api(cfg_path):
    cfg = read_config(cfg_path)
    torch.set_num_threads(cfg.get("cpu_num_threads", 12))
    model_cls = cfg["model_cls"]
    if "raw" in model_cls or "262" in model_cls:
        return get_ntk_llama_chat_api_with_tokenizer_bs1(cfg_path)
    elif "h2o" in model_cls:
        return get_h2o_api.get_chat_api(cfg_path)
    elif "quest" in model_cls:
        raise NotImplementedError
        # return get_quest_chat_api(cfg_path)
    elif "infllm" in model_cls:
        # raise ValueError("难以适配")
        return get_infllm_chat_api_bs1(cfg_path)
    else:
        return get_token_select_llama_chat_api_with_tokenizer_bs1(cfg_path)


if __name__ == "__main__":
    pass
