import os
import time

import transformers
import torch
from typing import List, Optional, Dict, Tuple, Any
from tiny_tools.log import logger


def get_idx_iou_score(new_idx, old_idx):
    """
    Calculate the IOU score between two 1D tensors.

    Parameters:
    new_idx (torch.Tensor): 1D tensor representing the new indices.
    old_idx (torch.Tensor): 1D tensor representing the old indices.

    Returns:
    float: The IOU score.
    """
    # Ensure the tensors are 1D
    assert new_idx.dim() == 1 and old_idx.dim() == 1, "Both tensors should be 1D"
    # Calculate the intersection
    # new_idx, old_idx = new_idx.unique(), old_idx.unique()
    un = torch.cat([new_idx, old_idx], dim=0).unique().shape[0]
    intersection_len = new_idx.shape[0] + old_idx.shape[0] - un
    # 这里先跳过，直接算交集除以长度
    # union = torch.cat((new_idx, old_idx)).unique().numel()
    # Calculate IOU
    iou_score = intersection_len / new_idx.shape[0]
    return iou_score


class DynamicSubCache(transformers.cache_utils.DynamicCache):

    def __init__(self):
        super(DynamicSubCache, self).__init__()
        self.idx = {}

    def set_out_idx(self, idx, layer_idx):
        self.idx[layer_idx] = idx

    def clear(self, layer_idx):
        while len(self.key_cache) > layer_idx + 1:
            del self.key_cache[-1], self.value_cache[-1]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        if layer_idx in self.idx:
            idx = (
                self.idx[layer_idx]
                .unsqueeze(1)
                .unsqueeze(-1)
                .expand(
                    -1,
                    self.key_cache[layer_idx].shape[1],
                    -1,
                    self.key_cache[layer_idx].shape[-1],
                )
            )
            return self.key_cache[layer_idx].gather(2, idx), self.value_cache[
                layer_idx
            ].gather(2, idx)
        else:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(cache: transformers.cache_utils.DynamicCache):
        c = DynamicSubCache()
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class DynamicSubOffloadTrueCache(transformers.cache_utils.DynamicCache):

    def __init__(self, offload_start_id=12, num_total_layers=32):
        super().__init__()
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = offload_start_id
        self.num_total_layers = num_total_layers
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = None
        self.mamba_v = None

    def set_out_idx(self, idx, sid):
        # 为了分析模型如何选择，通过config设置offload sid > total_layers，在这里需要跳过
        if os.environ.get("SAVE_SELECTED_IDX", False):
            assert self.offload_start_id >= self.num_total_layers
            return

        if (bs := idx.shape[0]) == 1:
            _idx = idx.cpu().view(-1)
        else:
            # TODO not tested
            _idx = idx.view(bs, -1, 1).repeat(
                1, 1, self.mamba_k.shape[-2], self.mamba_k.shape[-1]
            )
        st = time.time()
        for i in range(sid, self.num_total_layers):
            self.idx[i] = idx
        sz = self.mamba_k.shape[2] // (self.num_total_layers - self.offload_start_id)
        if bs == 1:
            k = self.mamba_k[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
            v = self.mamba_v[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
        else:
            # TODO not tested
            k = (
                torch.gather(self.mamba_k, 1, _idx)
                .cuda(non_blocking=True)
                .split(sz, dim=2)
            )
            v = (
                torch.gather(self.mamba_v, 1, _idx)
                .cuda(non_blocking=True)
                .split(sz, dim=2)
            )
        for i in range(self.offload_start_id, self.num_total_layers):
            self.part_key[i] = k[i - self.offload_start_id].transpose(1, 2)
            self.part_value[i] = v[i - self.offload_start_id].transpose(1, 2)

        logger.info(f"index&to cuda used {time.time() - st}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        # logger.debug(f"in layer {layer_idx}")
        if layer_idx >= self.offload_start_id:
            if self.stage == "prefill":
                st = time.time()
                _key = (
                    key_states.transpose(1, 2)
                    .contiguous()
                    .to(
                        device="cpu",
                        non_blocking=(layer_idx != self.num_total_layers - 1),
                    )
                )
                _value = (
                    value_states.transpose(1, 2)
                    .contiguous()
                    .to(
                        device="cpu",
                        non_blocking=(layer_idx != self.num_total_layers - 1),
                    )
                )
                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(f"L{layer_idx} offload to cpu {time.time() - st}")
                if layer_idx == self.num_total_layers - 1:
                    self.mamba_k = torch.cat(
                        self.key_cache[self.offload_start_id :], dim=2
                    ).contiguous()
                    self.mamba_v = torch.cat(
                        self.value_cache[self.offload_start_id :], dim=2
                    ).contiguous()
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

            if layer_idx in self.idx:
                idx = (
                    self.idx[layer_idx]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(
                        -1,
                        self.key_cache[layer_idx].shape[1],
                        -1,
                        self.key_cache[layer_idx].shape[-1],
                    )
                )
                return self.key_cache[layer_idx].gather(2, idx), self.value_cache[
                    layer_idx
                ].gather(2, idx)
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        offload_start_id=12,
        num_total_layers=32,
    ):
        c = DynamicSubOffloadTrueCache(offload_start_id, num_total_layers)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class OmniKVMultiStageCache(transformers.cache_utils.DynamicCache):
    # 默认DenseMore=True
    def __init__(
        self,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        super().__init__()
        # self.selected_indices = {}
        self.stage = "prefill"
        self.full_attn_layers = full_attn_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_wait_layers = num_wait_load_layers
        self.layer_state = {}
        self.need_cat_layers = []
        self.device = "cpu" if real_offload else None
        # 设定层的状态
        for _l in self.full_attn_layers:
            _r = num_hidden_layers
            for i in range(_l + 1, self.num_hidden_layers):
                if i in self.full_attn_layers:
                    _r = i
                    break
            for i in range(_l, min(_r, _l + self.num_wait_layers + 1)):
                self.layer_state[i] = (False, _l, _r)
            for i in range(min(_r, _l + self.num_wait_layers + 1), _r):
                self.layer_state[i] = (True, _l, _r)  # 需要offload

        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = {}
        self.mamba_v = {}

    def set_idx_on_gpu(self, idx, sel_layer_idx):
        st = time.time()
        _idx = idx.view(-1)
        if self.device == "cpu":
            _idx = _idx.cpu()
        _r = self.layer_state[sel_layer_idx][2]
        sid = sel_layer_idx + self.num_wait_layers + 1
        for i in range(sid, _r):
            self.part_key[i] = torch.index_select(self.key_cache[i], 2, _idx).cuda(
                non_blocking=True
            )
            self.part_value[i] = torch.index_select(self.value_cache[i], 2, _idx).cuda(
                non_blocking=True
            )
        logger.info(
            f"index&to cuda for layer={sel_layer_idx} used={round(time.time() - st, 3)}s"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.layer_state[layer_idx][0]:  # 需要offload
            if self.stage == "prefill":
                st = time.time()
                is_last_layer = layer_idx == self.num_hidden_layers - 1
                _key = key_states.to(
                    device=self.device if self.device else key_states.device,
                    non_blocking=not is_last_layer,
                )
                _value = value_states.to(
                    device=self.device if self.device else value_states.device,
                    non_blocking=not is_last_layer,
                )

                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(
                    f"Layer={layer_idx} offload to cpu {round(time.time() - st, 3)}s"
                )
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        c = OmniKVMultiStageCache(
            full_attn_layers, num_hidden_layers, num_wait_load_layers, real_offload
        )
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class WOPackCache(transformers.cache_utils.DynamicCache):
    # 默认DenseMore=True
    def __init__(
        self,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        super().__init__()
        # self.selected_indices = {}
        self.stage = "prefill"
        self.full_attn_layers = full_attn_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_wait_layers = num_wait_load_layers
        self.layer_state = {}
        self.need_cat_layers = []
        self.device = "cpu" if real_offload else None
        # 设定层的状态
        for _l in self.full_attn_layers:
            _r = num_hidden_layers
            for i in range(_l + 1, self.num_hidden_layers):
                if i in self.full_attn_layers:
                    _r = i
                    break
            for i in range(_l, min(_r, _l + self.num_wait_layers + 1)):
                self.layer_state[i] = (False, _l, _r)
            for i in range(min(_r, _l + self.num_wait_layers + 1), _r):
                self.layer_state[i] = (True, _l, _r)  # 需要offload

        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = {}
        self.mamba_v = {}
        self.idx = None

    def set_idx_on_gpu(self, idx, sel_layer_idx):
        st = time.time()
        _idx = idx.cpu().view(-1)
        self.idx = _idx
        # _r = self.layer_state[sel_layer_idx][2]
        # sid = sel_layer_idx + self.num_wait_layers + 1
        # for i in range(sid, _r):
        #     self.part_key[i] = torch.index_select(self.key_cache[i], 2, _idx).cuda(non_blocking=True)
        #     self.part_value[i] = torch.index_select(self.value_cache[i], 2, _idx).cuda(non_blocking=True)
        # logger.info(f"index&to cuda for layer={sel_layer_idx} used={round(time.time() - st, 3)}s")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.layer_state[layer_idx][0]:  # 需要offload
            if self.stage == "prefill":
                st = time.time()
                is_last_layer = layer_idx == self.num_hidden_layers - 1
                _key = key_states.to(
                    device=self.device if self.device else key_states.device,
                    non_blocking=not is_last_layer,
                )
                _value = value_states.to(
                    device=self.device if self.device else value_states.device,
                    non_blocking=not is_last_layer,
                )

                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(
                    f"Layer={layer_idx} offload to cpu {round(time.time() - st, 3)}s"
                )
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )
                if torch.max(key_states) > -1e4:  # to ensure sync
                    self.part_key[layer_idx] = torch.index_select(
                        self.key_cache[layer_idx], 2, self.idx
                    ).cuda()
                    self.part_value[layer_idx] = torch.index_select(
                        self.value_cache[layer_idx], 2, self.idx
                    ).cuda()
                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        full_attn_layers: List,
        num_hidden_layers,
        num_wait_load_layers=2,
        real_offload=True,
    ):
        c = WOPackCache(
            full_attn_layers, num_hidden_layers, num_wait_load_layers, real_offload
        )
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class OmniKVLazyCache(DynamicSubOffloadTrueCache):

    def __init__(self, offload_start_id=12, num_total_layers=32, skip_threshold=0.4):
        super().__init__()
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = offload_start_id
        self.num_total_layers = num_total_layers
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.mamba_k = None
        self.mamba_v = None
        self.skip_threshold = skip_threshold

    def set_out_idx(self, idx, sid):
        assert idx.shape[0] == 1
        _idx = idx.cpu().view(-1)
        if sid in self.idx:
            sc = get_idx_iou_score(idx.view(-1), self.idx[sid].view(-1))
            logger.info(f"current score = {sc}")
            if sc > self.skip_threshold:
                return
        # 接下来进行更新
        st = time.time()
        for i in range(sid, self.num_total_layers):
            self.idx[i] = idx
        sz = self.mamba_k.shape[2] // (self.num_total_layers - self.offload_start_id)
        k = self.mamba_k[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
        v = self.mamba_v[:, _idx].cuda(non_blocking=True).split(sz, dim=2)
        for i in range(self.offload_start_id, self.num_total_layers):
            self.part_key[i] = k[i - self.offload_start_id].transpose(1, 2)
            self.part_value[i] = v[i - self.offload_start_id].transpose(1, 2)
        logger.info(f"index&to cuda used {time.time() - st}")

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        offload_start_id=12,
        num_total_layers=32,
        skip_threshold=0.4,
    ):
        c = OmniKVLazyCache(
            offload_start_id, num_total_layers, skip_threshold=skip_threshold
        )
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class OmniKVMoreEffCache(transformers.cache_utils.DynamicCache):
    def __init__(self, config):
        super().__init__()
        if cpu_cache is None:
            make_cpu_cache(config)
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = config.get("offload_sid", 12)
        self.num_total_layers = config.get("num_hidden_layers", 32)
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}

    def set_out_idx(self, idx, sid):
        st = time.time()
        for i in range(sid, self.num_total_layers):
            self.idx[i] = idx
        k, v = cpu_cache.get_cache(idx)
        for i in range(self.offload_start_id, self.num_total_layers):
            self.part_key[i] = k[i - self.offload_start_id].transpose(1, 2)
            self.part_value[i] = v[i - self.offload_start_id].transpose(1, 2)
        logger.debug(f"index&to cuda used {time.time() - st}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if layer_idx >= self.offload_start_id:
            if self.stage == "prefill":
                st = time.time()
                _key, _value = (
                    key_states.transpose(1, 2).contiguous(),
                    value_states.transpose(1, 2).contiguous(),
                )
                cpu_cache.set_prefilled_cache(_key, _value, layer_idx)
                logger.info(f"L{layer_idx} offload to cpu {time.time() - st}")
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                return (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

            if layer_idx in self.idx:
                idx = (
                    self.idx[layer_idx]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(
                        -1,
                        self.key_cache[layer_idx].shape[1],
                        -1,
                        self.key_cache[layer_idx].shape[-1],
                    )
                )
                return self.key_cache[layer_idx].gather(2, idx), self.value_cache[
                    layer_idx
                ].gather(2, idx)
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(cache: transformers.cache_utils.DynamicCache, config):
        c = OmniKVMoreEffCache(config)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class DynamicBrutalOffloadCache(transformers.cache_utils.DynamicCache):

    def __init__(self, offload_start_id=12, num_total_layers=32):
        super().__init__()
        self.idx = {}
        self.stage = "prefill"
        self.offload_start_id = offload_start_id
        self.num_total_layers = num_total_layers
        self.streams = {
            i: torch.cuda.Stream()
            for i in range(self.offload_start_id, self.num_total_layers)
        }
        self.part_key = {}
        self.part_value = {}
        self.tail_k = {}
        self.tail_v = {}
        self.offset = 4

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        # logger.debug(f"in layer {layer_idx}")
        if (
            self.stage == "decoding"
            and self.num_total_layers > layer_idx + self.offset >= self.offload_start_id
        ):
            self.part_key[layer_idx + self.offset] = self.key_cache[
                layer_idx + self.offset
            ].cuda(non_blocking=True)
            self.part_value[layer_idx + self.offset] = self.value_cache[
                layer_idx + self.offset
            ].cuda(non_blocking=True)
        if layer_idx >= self.offload_start_id:
            assert key_states.shape[0] == 1, "now only bsz==1"
            if self.stage == "prefill":
                st = time.time()
                _key = key_states.to(device="cpu", non_blocking=True)
                _value = value_states.to(device="cpu", non_blocking=True)
                # _key = key_states
                # _value = value_states
                self.key_cache.append(_key)
                self.value_cache.append(_value)
                logger.info(f"L{layer_idx} offload to cpu {time.time() - st}")
                return key_states, value_states
            else:
                if layer_idx not in self.tail_k:
                    self.tail_k[layer_idx] = key_states
                    self.tail_v[layer_idx] = value_states
                else:
                    self.tail_k[layer_idx] = torch.cat(
                        [self.tail_k[layer_idx], key_states], dim=-2
                    )
                    self.tail_v[layer_idx] = torch.cat(
                        [self.tail_v[layer_idx], value_states], dim=-2
                    )

                assert key_states.shape[-2] == 1
                temp = (
                    torch.cat(
                        [self.part_key[layer_idx], self.tail_k[layer_idx]], dim=-2
                    ),
                    torch.cat(
                        [self.part_value[layer_idx], self.tail_v[layer_idx]], dim=-2
                    ),
                )
                self.part_key[layer_idx] = None
                self.part_value[layer_idx] = None
                return temp

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def from_dynamic_cache(
        cache: transformers.cache_utils.DynamicCache,
        offload_start_id=12,
        num_total_layers=32,
    ):
        c = DynamicBrutalOffloadCache(offload_start_id, num_total_layers)
        c.key_cache = cache.key_cache
        c.value_cache = cache.value_cache
        c._seen_tokens = cache._seen_tokens
        return c


class SinkCache(transformers.cache_utils.DynamicCache):
    def __init__(self, window_length, num_sink_tokens):
        super().__init__()
        self.window_len = window_length
        self.num_sink_tokens = num_sink_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            assert key_states.shape[2] > 1
            if key_states.shape[-2] > self.window_len + self.num_sink_tokens:
                key_states = torch.cat(
                    [
                        key_states[:, :, : self.num_sink_tokens],
                        key_states[:, :, -self.window_len :],
                    ],
                    dim=-2,
                )  # noqa
                value_states = torch.cat(
                    [
                        value_states[:, :, : self.num_sink_tokens],
                        value_states[:, :, -self.window_len :],
                    ],
                    dim=-2,
                )  # noqa
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert key_states.shape[2] == 1
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def get_cache_cls(config):
    name2cls = {
        "lazy": OmniKVLazyCache,
        "default": DynamicSubOffloadTrueCache,
        "eff": OmniKVMoreEffCache,
        "multi": OmniKVMultiStageCache,
        "without_pack": WOPackCache,
    }
    return name2cls[config.get("cache_cls", "default")]
