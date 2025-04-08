import torch
from transformers.models.llama.modeling_llama import *
from tiny_tools.log import logger

_CONFIG_FOR_DOC = "LlamaConfig"

logger.debug("start")


class LlamaCompressorConfig(LlamaConfig):
    def set_config_of_compressor(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        if not hasattr(self, key):
            setattr(self, key, default)
            logger.warning(f"{key}不存在，被设置为{default}")
        return getattr(self, key, default)

    def _rope_scaling_validation(self):
        logger.warning("为llama3.1做的patch，不验证了")
        return
