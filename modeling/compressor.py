import torch
from transformers.models.llama.modeling_llama import *
from tiny_tools.log import logger

_CONFIG_FOR_DOC = "LlamaConfig"

logger.debug("start")


class OmniKVCompressorConfig(LlamaConfig):
    def set_config_of_compressor(self, **kwargs):
        for key in ["tree_height", "minL", "segment_size"]:
            assert key in kwargs, "use kwargs for params"
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        if not hasattr(self, key):
            setattr(self, key, default)
            logger.warning(f"{key}不存在，被设置为{default}")
        return getattr(self, key, default)
