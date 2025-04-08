import torch
import hashlib
import pickle
import os


def dict_to_cuda(d, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)


def tensor_hash_stable(tensor):
    """
    Compute a stable hash value of a torch.int tensor using MD5 hash function.

    Args:
    tensor (torch.Tensor): An input tensor of type torch.int

    Returns:
    str: The hex digest of the MD5 hash value of the input tensor
    """
    # 确认输入是 torch.Tensor 类型
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # 获取张量的字节表示
    tensor_bytes = tensor.cpu().contiguous().view(-1).numpy().tobytes()

    # 创建 MD5 对象并计算哈希值
    md5 = hashlib.md5()
    md5.update(tensor_bytes)

    # 返回 MD5 哈希的十六进制摘要
    return md5.hexdigest()


class SelectedIndexTracer:
    def __init__(self):
        self.idx = {}
        self.layers_idx = {}
        self.outs = []
        self.num_samples = 0
        self.prefill_res = {}
        desc = os.environ.get('SAVE_SELECTED_IDX')
        self.save_path = f'debug_logs/selected_index_{desc}.pkl'

    def append_out(self, out):
        self.outs.append(out)

    def append(self, idx, v):
        if self.num_samples not in self.idx:
            self.idx[self.num_samples] = []
        self.idx[self.num_samples].append({
            "idx": idx.to(device='cpu', non_blocking=True),
            "value": v.to(device='cpu', non_blocking=True)
        })

    def append_prefill(self, idx, v, layer_idx):
        # assert layer_idx == 8
        self.prefill_res[self.num_samples] = {"idx": idx.cpu(), "value": v.cpu()}

    def append_spec_layer(self, idx, v, prefill_len, layer_idx):
        assert len(idx.shape) == 2 and len(v.shape) == 2
        if self.num_samples not in self.layers_idx:
            self.layers_idx[self.num_samples] = {}
        if layer_idx not in self.layers_idx[self.num_samples]:
            self.layers_idx[self.num_samples][layer_idx] = []
        self.layers_idx[self.num_samples][layer_idx].append({
            "idx": idx.to(device='cpu', non_blocking=True),
            "value": v.to(device='cpu', non_blocking=True),
            "prefill_len": prefill_len
        })

    def save_idx(self):
        torch.cuda.synchronize()
        with open(self.save_path, 'wb') as _out:
            pickle.dump({"idx": self.idx, "out": self.outs, "layers_idx": self.layers_idx,
                         "prefill_res": self.prefill_res}, _out)
        self.num_samples += 1

    def __del__(self):
        self.save_idx()


def read_idx_tracer(pkl_path):
    with open(pkl_path, 'rb') as _in:
        d = pickle.load(_in)
        idx = d['idx']
        out = d['out']
        layers_idx = d['layers_idx']
        prefill_res = d['prefill_res']
        return idx, out, layers_idx, prefill_res


idx_tracer = None
if os.environ.get("SAVE_SELECTED_IDX", False):
    idx_tracer = SelectedIndexTracer()
