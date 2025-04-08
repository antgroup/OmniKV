import os
import sys
from omegaconf import OmegaConf
from baselines.infllm.benchmark.pred import get_model_and_tokenizer, parse_args_with_path
from tiny_tools.read_json import read_config


def get_model_tokenizer_others(cfg_path, quant_config=None):
    d_cfg = read_config(cfg_path)
    infllm_cfg = parse_args_with_path(d_cfg['infllm_cfg_path'])
    model, tkn = get_model_and_tokenizer(infllm_cfg.model, quant_config=quant_config)
    return model, tkn
