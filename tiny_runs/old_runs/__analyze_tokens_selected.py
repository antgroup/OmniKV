import re
from fire import Fire
from transformers import AutoTokenizer
import pickle
import matplotlib.pyplot as plt
import torch


def visualize_sum_tensors(tensor_list, name):
    # Find the maximum index to determine the size of the resulting tensor
    max_index = max([idx.max().item() for idx, _ in tensor_list]) + 1
    # Find the maximum value among all tensors
    max_val = max([val.max().item() for _, val in tensor_list])

    # Set up plot settings before plotting any tensor
    plt.figure(figsize=(40, 30))  # Set figure size to 4000x3000, but matplotlib uses inches, so we use 40x30
    plt.ylim(0, max_val)  # Set y-axis limit to 0 and the max_val found

    for idx, val in tensor_list:
        # Initialize a tensor with zeros
        result_tensor = torch.zeros(max_index)
        result_tensor[idx] = val

        # Plot the result tensor
        plt.bar(range(len(result_tensor)), result_tensor)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Tensor Values by Index')

    # Save the figure with higher resolution for better quality when scaled up
    plt.savefig(f'debug_logs/{name}.png', dpi=100)  # Increase DPI for higher resolution


def visualize_heatmap(tensor_list, name, out_ids, prompt_ids):
    # Determine the size of the resulting heatmap
    max_index = max([idx.max().item() for idx, _ in tensor_list]) + 1
    num_tensors = len(tensor_list)

    # Initialize a tensor to hold the values for the heatmap
    heatmap_data = torch.zeros(num_tensors, max_index)

    # Fill the heatmap_data tensor with values from tensor_list
    for i, (idx, val) in enumerate(tensor_list):
        idx = idx.view(-1)
        val = val.view(-1)
        idx = idx[:256]
        val = val[:256]
        val /= val.max()
        val = val ** 0.1
        heatmap_data[i, idx] = val

    plt.figure(figsize=(80, 30))  # 增加宽度以适应更多的垂直标签
    # Draw the heatmap using imshow or pcolormesh
    mesh = plt.pcolormesh(heatmap_data, cmap='viridis')  # 'viridis' is a colormap
    # Add a color bar
    plt.colorbar(mesh)
    # Set axis labels
    plt.xlabel('Index')
    plt.ylabel('List Element Index')
    # Set the y-axis labels to be the elements of in_ids
    plt.yticks(range(0, len(out_ids)), out_ids)
    plt.xticks(range(0, len(prompt_ids)), prompt_ids, rotation=90)
    # Set title
    plt.title('Heatmap of Tensor Values')
    plt.savefig(f'debug_logs/{name}_heatmap.png', dpi=100)


def read_idx(pk_path):
    with open(pk_path, 'rb') as _in:
        d = pickle.load(_in)
    return d['idx'], d['out']


def main(pk_path="debug_logs/selected_index.pkl", log_path="debug_logs/main_2024-07-25-19-07_2c323.log",
         model_path="/input/jitai/huggingface/hub/Lourdle/Llama-3-8B-Instruct-262k"):
    tok = AutoTokenizer.from_pretrained(model_path)
    dict__id_list, outs = read_idx(pk_path)
    # with open(log_path, 'r', encoding='utf-8') as _in:
    #     contents = _in.readlines()
    # contents = '\n'.join(contents)
    # outs = re.findall(r"---\d+?:(.+)", contents)
    id_vocab = {v: k for k, v in tok.vocab.items()}
    for i, f in enumerate(dict__id_list.items()):
        k, v = f
        print(f"###sample {k}")
        print(f"decode len {len(v)}")
        v = [(d['idx'], d['value']) for d in v]
        # visualize_sum_tensors(v, f'sum_id{k}')
        out_ids = tok.encode(outs[i][1])
        prompt_ids = tok.encode(outs[i][0])
        prompt_ids = [(id_vocab[e] if e in id_vocab else '') for e in prompt_ids]
        print(outs[i][1])
        out_ids = [str(e) + ' ' + (id_vocab[e] if e in id_vocab else '') for e in out_ids]
        out_ids = out_ids[1:]  # 记录的score不包括prefill得到的那个
        visualize_heatmap(v, f'sum_id{k}', out_ids, prompt_ids)
        print('done...')


if __name__ == '__main__':
    Fire(main)
