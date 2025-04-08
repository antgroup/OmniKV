import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from tiny_tools.tensor_tools import read_idx_tracer
from fire import Fire
from tqdm import trange

torch.set_num_threads(2)


def get_attn_sparsity(sorted_score: torch.Tensor, prefill_len, threshold=0.99):
    cumulative_scores = sorted_score.cumsum(dim=-1)
    mask = (cumulative_scores >= threshold).to(dtype=torch.int)
    if mask.max() == 1:
        valid_indices = mask.argmax(dim=-1) + 1
    else:
        print("NOT Enough !")
        valid_indices = prefill_len
    sparsity_ratio = valid_indices.float() / prefill_len
    return torch.mean(sparsity_ratio).item()


def get_similarity_between_layer(a: torch.Tensor, b: torch.Tensor):
    a, b = a.view(-1), b.view(-1)
    assert a.dim() == 1 and b.dim() == 1, "Both tensors should be 1D"
    un = torch.cat([a, b], dim=0).unique().shape[0]
    intersection_len = a.shape[0] + b.shape[0] - un
    p = intersection_len / a.shape[0]
    return p


def get_sum_attn_score_by_spec_idx(spec_idx, idx, val):
    spec_idx = spec_idx.view(-1)
    idx = idx.view(-1)
    val = val.view(-1)
    assert spec_idx.dim() == idx.dim() == val.dim() == 1
    mask = torch.zeros_like(idx, dtype=torch.bool)
    mask[spec_idx] = True
    nv = torch.zeros_like(val)
    nv[idx] = val
    return torch.sum(nv[mask]).item()


def draw_attn_sparsity(layers_idx, task, threshold=0.99):
    num_samples = len(layers_idx)
    num_layers = len(layers_idx[0])
    layer_sparsity = [[] for _ in range(num_layers)]
    prefill_sparsity = {}
    step_sparsity = {}
    prefill_lengths = [1024, 2048, 4096, 6144, 8192]
    similarity_prefill_layers = {length: np.zeros((num_layers, num_layers)) for length in prefill_lengths}
    score_by_spec_idx_layers = {length: np.zeros((num_layers, num_layers)) for length in prefill_lengths}
    for i in trange(num_samples):
        idx_1sample = layers_idx[i]
        for j in range(num_layers):
            idx_1layer_all_steps = idx_1sample[j]
            num_steps = len(idx_1layer_all_steps)
            for k in range(num_steps):
                idx, val = idx_1layer_all_steps[k]['idx'], idx_1layer_all_steps[k]['value']
                prefill_len = idx_1layer_all_steps[k]['prefill_len']
                assert val.dtype == torch.float32
                sparsity = get_attn_sparsity(val, prefill_len, threshold)
                layer_sparsity[j].append(sparsity)
                # Prefill sparsity tracking
                if prefill_len not in prefill_sparsity:
                    prefill_sparsity[prefill_len] = []
                prefill_sparsity[prefill_len].append(sparsity)
                # Step sparsity tracking
                if k not in step_sparsity:
                    step_sparsity[k] = []
                step_sparsity[k].append(sparsity)
        for length in prefill_lengths:
            for l1 in range(num_layers):
                for l2 in range(num_layers):
                    _sample_len = min(4, len(idx_1sample[l1]))
                    _sample_idx = random.choices(list(range(len(idx_1sample[l1]))), k=_sample_len)
                    for k in _sample_idx:
                        idx1_sample = idx_1sample[l1][k]['idx'].view(-1)[:length]
                        idx2_sample = idx_1sample[l2][k]['idx'].view(-1)[:length]
                        similarity = get_similarity_between_layer(idx1_sample, idx2_sample)
                        similarity_prefill_layers[length][l1, l2] += similarity / _sample_len
                    # similarity_prefill_layers[length][l1, l2] /= len(idx_1sample[l1])
            for l1 in range(num_layers):
                for l2 in range(l1, num_layers):  # Ensure l1 < l2
                    _sample_len = min(4, len(idx_1sample[l1]))
                    _sample_idx = random.choices(list(range(len(idx_1sample[l1]))), k=_sample_len)
                    for k in _sample_idx:
                        idx1_sample = idx_1sample[l1][k]['idx'].view(-1)[:length]
                        idx2_sample = idx_1sample[l2][k]['idx'].view(-1)
                        idx2_val = idx_1sample[l2][k]['value'].view(-1)
                        cum_score = get_sum_attn_score_by_spec_idx(idx1_sample, idx2_sample, idx2_val)
                        score_by_spec_idx_layers[length][l1, l2] += cum_score / _sample_len
    for length in prefill_lengths:
        similarity_prefill_layers[length] /= num_samples
        score_by_spec_idx_layers[length] /= num_samples

    avg_layer_sparsity = [torch.tensor(layer).mean(dim=0).item() for layer in layer_sparsity]

    # 确保目标文件夹存在
    import os
    os.makedirs("debug_logs/detail_attn_analysis", exist_ok=True)

    # 子图1: 每层平均稀疏率
    plt.figure(figsize=(15, 10))
    plt.plot(avg_layer_sparsity)
    plt.title('Avg Sparsity Ratio Across Layers')
    plt.xlabel('Layer ID')
    plt.ylabel('Avg Sparsity Ratio')
    plt.savefig("debug_logs/detail_attn_analysis/avg_sparsity_across_layers.png")
    plt.close()

    # 子图2: 不同预填长度之间的稀疏性
    plt.figure(figsize=(15, 10))
    for prefill_len, sparsity_list in prefill_sparsity.items():
        avg_sparsity = torch.tensor(sparsity_list).mean(dim=0).item()
        plt.scatter([prefill_len], [avg_sparsity], label=f'Prefill Length {prefill_len}')
    plt.title('Avg Sparsity Ratio Across Different Prefill Lengths')
    plt.xlabel('Prefill Length')
    plt.ylabel('Avg Sparsity Ratio')
    plt.legend()
    plt.savefig("debug_logs/detail_attn_analysis/avg_sparsity_across_prefill_lengths.png")
    plt.close()

    # 子图3: 不同步数之间的稀疏性差异
    plt.figure(figsize=(15, 10))
    for step, sparsity_list in step_sparsity.items():
        avg_sparsity = torch.tensor(sparsity_list).mean(dim=0).item()
        plt.scatter([step], [avg_sparsity], label=f'Step {step}')
    plt.title('Avg Sparsity Ratio Across Different Steps')
    plt.xlabel('Step')
    plt.ylabel('Avg Sparsity Ratio')
    plt.legend()
    plt.savefig("debug_logs/detail_attn_analysis/avg_sparsity_across_steps.png")
    plt.close()

    # 子图4到8: 层与层之间相似度
    for i, length in enumerate(prefill_lengths):
        plt.figure(figsize=(15, 12))
        cax = plt.matshow(similarity_prefill_layers[length], cmap='coolwarm', fignum=False)
        plt.colorbar(cax)
        plt.title(f'Similarity Between Layers (Prefill Length {length})')
        plt.xlabel('Layer ID')
        plt.ylabel('Layer ID')
        for x in range(similarity_prefill_layers[length].shape[0]):
            for y in range(similarity_prefill_layers[length].shape[1]):
                plt.text(y, x, f'{similarity_prefill_layers[length][x, y]:.2f}',
                         ha="center", va="center", color="black", fontsize=6)
        plt.savefig(f"debug_logs/detail_attn_analysis/similarity_between_layers_prefill_{length}.png")
        plt.close()

    # 子图9到13: 累积attn分数的热力图 (根据指定idx)
    colormap = 'viridis'
    for i, length in enumerate(prefill_lengths):
        plt.figure(figsize=(10, 10))
        cax = plt.matshow(score_by_spec_idx_layers[length], cmap=colormap, fignum=False)
        plt.colorbar(cax, shrink=0.7)
        # plt.title(f'Cumulative Attention Score Specified Index (Prefill Length {length})')
        plt.xlabel('Layer ID')
        plt.ylabel('Layer ID')
        for x in range(score_by_spec_idx_layers[length].shape[0]):
            for y in range(score_by_spec_idx_layers[length].shape[1]):
                if x <= y:
                    plt.text(y, x, f'{score_by_spec_idx_layers[length][x, y]:.2f}',
                             ha="center", va="center", color="black", fontsize=5)
        plt.tight_layout()
        plt.savefig(f"debug_logs/detail_attn_analysis/cumulative_attn_score_prefill_{length}.png", bbox_inches='tight')
        plt.savefig(f"debug_logs/detail_attn_analysis/cumulative_attn_score_prefill_{length}.pdf", bbox_inches='tight')
        plt.close()

    # 保存全局标题信息
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, f'Attention Sparsity and Specified Index Cumulative Score (threshold={threshold})',
             horizontalalignment='center', verticalalignment='center', fontsize=14)
    plt.axis('off')
    plt.savefig("debug_logs/detail_attn_analysis/global_title.png")
    plt.close()


def main(model=''):
    tasks = (
        # "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
        # "dureader", "gov_report", "qmsum",
        # # "multi_news", # 找不到
        # "vcsum", "trec", "triviaqa",
        # # "samsum", # 找不到
        # "lsht",
        # "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
        # "longbook_qa_eng",
        # "longbook_choice_eng",
        # "longdialogue_qa_eng",
        # # "longbook_sum_eng",
        # "longbook_qa_chn",
        # "code_debug",
        # "math_find",
        # "passkey",
        # "number_string",
        # "kv_retrieval",

        # "llama8b_2wikimqa_cot",
        # "yi_2wikimqa_cot",
        # "llama70b_2wikimqa_cot",
        # "yi34b_2wikimqa_cot",
        "llama8b_hotpotqa",
    )
    tasks = [f"{model}{task}" for task in tasks]
    for task in tasks:
        _, __, layers_idx, ___ = read_idx_tracer(f"debug_logs/selected_index_{task}.pkl")
        print(task)
        draw_attn_sparsity(layers_idx, task, 0.9)


if __name__ == '__main__':
    Fire(main)
