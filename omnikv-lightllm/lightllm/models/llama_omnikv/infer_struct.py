import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager


class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None

        self.now_layer_idx = -1
        self.prefill_layer_idx = -1
        self.imp_seq_len = None
        self.max_imp_len_batch = None
        self.cache_cpu_kv = {}
        self.imp_cache = {}

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        self.now_layer_idx = -1
        self.imp_seq_len = None
        self.max_imp_len_batch = None
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            b_ready_cache_len_numpy = self.b_ready_cache_len.cpu().numpy()
            position_ids = torch.from_numpy(
                np.concatenate(
                    [
                        np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i])
                        for i in range(len(b_seq_len_numpy))
                    ],
                    axis=0,
                )
            ).cuda()
            self.position_cos = torch.index_select(
                model._cos_cached, 0, position_ids
            ).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(
                model._sin_cached, 0, position_ids
            ).view(position_ids.shape[0], -1)
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = torch.index_select(
                model._cos_cached, 0, position_ids
            ).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(
                model._sin_cached, 0, position_ids
            ).view(self.b_seq_len.shape[0], -1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[
                self.b_req_idx[0], 0
            ].item()
            # b_loc[0, max_len_in_batch - 1].item()
        return
