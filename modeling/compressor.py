# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""
import torch
from transformers.models.llama.modeling_llama import *
from tiny_tools.log import logger

_CONFIG_FOR_DOC = "LlamaConfig"

logger.debug("start")


class LlamaCompressorConfig(LlamaConfig):
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
