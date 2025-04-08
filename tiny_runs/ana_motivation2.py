import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from tiny_tools.tensor_tools import read_idx_tracer
from fire import Fire
from tqdm import trange
import seaborn as sns

sns.set_theme()
torch.set_num_threads(4)


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


def draw_attn_sparsity(layers_idx, task, clear=True):
    num_samples = len(layers_idx)
    num_layers = len(layers_idx[0])
    print("num_layers", num_layers)
    topk_nums = [8192, 12800, 16384]
    score_by_spec_idx_layers = {length: np.zeros((num_layers, num_layers)) for length in topk_nums}
    for i in trange(num_samples):
        idx_1sample = layers_idx[i]
        for length in topk_nums:
            for l1 in range(num_layers):
                for l2 in range(l1, num_layers):  # Ensure l1 < l2
                    _sample_len = min(4, len(idx_1sample[l1]))
                    _sample_idx = random.choices(list(range(len(idx_1sample[l1]))), k=_sample_len)
                    for k in _sample_idx:
                        idx1_sample = idx_1sample[l1][k]['idx'].view(-1)[:length]
                        assert len(idx_1sample[l1]) == len(idx_1sample[l2]), f"{len(idx_1sample[l1])}," \
                                                                             f"{len(idx_1sample[l2])}"
                        idx2_sample = idx_1sample[l2][k]['idx'].view(-1)
                        idx2_val = idx_1sample[l2][k]['value'].view(-1)
                        cum_score = get_sum_attn_score_by_spec_idx(idx1_sample, idx2_sample, idx2_val)
                        score_by_spec_idx_layers[length][l1, l2] += cum_score / _sample_len
    for length in topk_nums:
        score_by_spec_idx_layers[length] /= num_samples

    #  开始画一个新图片
    if clear:
        plt.clf()
    #  定义平均的seg长度
    segs = [4, 8, 12, 16]
    lines = {l: [] for l in segs}
    for l in segs:
        for l1 in range(num_layers - l):
            lines[l].append(np.average(score_by_spec_idx_layers[8192][l1, l1:l1+l]))
    #  进行画图，保存到 debug_logs/attn_similar_sparsity_{task}.png
    scale_f = 1.5
    fig, ax = plt.subplots(figsize=(4 * scale_f, 3 * scale_f))

    for l, line_data in lines.items():
        ax.plot(line_data, label=f'Segment Length {l}')

    # ax.set_title('Average Cumulative Attention Scores by Segment Length')
    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Inter-Layer Attn Similarity')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"debug_logs/attn_similar_sparsity_{task}.png")
    plt.savefig(f"debug_logs/attn_similar_sparsity_{task}.pdf")

    # 清除当前图形，准备下一次绘图
    if clear:
        plt.clf()


def main(model=''):
    tasks = [
        # "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
        # "dureader", "gov_report", "qmsum",
        # # "multi_news", # 找不到
        # "vcsum", "trec", "triviaqa",
        # # "samsum", # 找不到
        # "lsht",
        # "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
        "longbook_qa_eng",
        "longbook_choice_eng",
        "longdialogue_qa_eng",
        # "longbook_sum_eng",
        "longbook_qa_chn",
        "code_debug",
        "math_find",
        "passkey",
        "number_string",
        "kv_retrieval",
    ]
    tasks = [f"{model}{task}" for task in tasks]

    for task in tasks:
        _, __, layers_idx = read_idx_tracer(f"debug_logs/selected_index_{task}.pkl")
        print(task)
        draw_attn_sparsity(layers_idx, task)


def main2():
    tasks = [
        # "llama8blongbook_choice_eng",
        # "yi9blongbook_choice_eng",
        # "llama8b_2wikimqa_cot",
        # "yi_2wikimqa_cot",
        # "llama70b_2wikimqa_cot"
        # 'yi9b_hotpotqa',
        # "llama70b_hotpotqa_cot",
        "llama8b_hotpotqa",
        "llama70b_hotpotqa",
        # "yi34b_hotpotqa_cot",
    ]
    scale_f = 1
    names = {
        'llama8blongbook_choice_eng': 'Llama3-8B',
        'yi9blongbook_choice_eng': 'Yi-9B',
        'llama70b_2wikimqa_cot': 'Llama3.1-70B',
        "yi9b_hotpotqa": 'Yi-9B',
        "llama70b_hotpotqa_cot": 'Llama3.1-70B',
        "llama70b_hotpotqa": 'Llama3.1-70B',
        'llama8b_hotpotqa': 'Llama3-8B',
        "yi34b_hotpotqa_cot": 'Yi-34B'
    }

    for ii, task in enumerate(tasks):
        plt.clf()
        plt.figure(figsize=(4 * scale_f, 3 * scale_f))

        _, __, layers_idx, _ = read_idx_tracer(f"debug_logs/selected_index_{task}.pkl")
        print(task)

        num_samples = len(layers_idx)
        num_layers = len(layers_idx[0])
        print("num_layers", num_layers)
        lenn = 4096
        if '70' in task:
            lenn = 4096
        topk_nums = [lenn]
        score_by_spec_idx_layers = {length: np.zeros((num_layers, num_layers)) for length in topk_nums}
        for i in trange(num_samples):
            idx_1sample = layers_idx[i]
            for length in topk_nums:
                for l1 in range(num_layers):
                    for l2 in range(l1, num_layers):  # Ensure l1 < l2
                        _sample_len = min(4, len(idx_1sample[l1]))
                        _sample_idx = random.choices(list(range(len(idx_1sample[l1]))), k=_sample_len)
                        for k in _sample_idx:
                            idx1_sample = idx_1sample[l1][k]['idx'].view(-1)[:length]
                            assert len(idx_1sample[l1]) == len(idx_1sample[l2]), f"{len(idx_1sample[l1])}," \
                                                                                 f"{len(idx_1sample[l2])}"
                            idx2_sample = idx_1sample[l2][k]['idx'].view(-1)
                            idx2_val = idx_1sample[l2][k]['value'].view(-1)
                            cum_score = get_sum_attn_score_by_spec_idx(idx1_sample, idx2_sample, idx2_val)
                            score_by_spec_idx_layers[length][l1, l2] += cum_score / _sample_len
        for length in topk_nums:
            score_by_spec_idx_layers[length] /= num_samples

        #  定义平均的seg长度
        segs = [8, 16]
        # segs = [4]
        lines = {l: [] for l in segs}
        for l in segs:
            for l1 in range(num_layers - l):
                lines[l].append(np.average(score_by_spec_idx_layers[lenn][l1, l1:l1 + l]))

        for l, line_data in lines.items():
            plt.plot(line_data, label=f'{names[task]} - n={l}')

        plt.xlabel('Transformer Layer ID')
        plt.ylabel('Inter-Layer Attn Similarity')
        plt.legend()
        # plt.tight_layout()
        plt.savefig(f"debug_logs/attn_similar_sparsity_last_{task}.png", bbox_inches='tight')
        plt.savefig(f"debug_logs/attn_similar_sparsity_last_{task}.pdf", bbox_inches='tight')


if __name__ == '__main__':
    # Fire(main)
    main2()
