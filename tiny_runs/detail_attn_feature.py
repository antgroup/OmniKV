import torch
import json
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM


class GetAttnMapLM(LlamaForCausalLM):
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if input_ids.shape[1] == 1:
            kwargs["output_attentions"] = True
        return super().forward(**kwargs)


def load_model_and_tokenizer(model_path):
    model = GetAttnMapLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tkn = AutoTokenizer.from_pretrained(model_path)
    return model, tkn


def greedy_decode(input_text, model, tkn, max_new_tokens=50):
    model.eval()  # Set model to evaluation mode

    inputs = tkn(input_text, return_tensors="pt").to(model.device)
    input_ids_prompt = inputs.input_ids

    current_generated_ids = input_ids_prompt  # Holds the full sequence including prompt and generated tokens

    collected_decode_attentions = []

    past_key_values = None

    if max_new_tokens == 0:
        output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
        return output_text, []

    with torch.no_grad():  # Disable gradient calculations for inference
        # --- Prefill Phase ---
        # Process the entire prompt to get initial past_key_values and logits for the first token.
        # GetAttnMapLM is specified to NOT output attentions in this prefill phase.
        prefill_outputs = model(
            input_ids=input_ids_prompt,
            use_cache=True
            # output_attentions is internally False for prefill by GetAttnMapLM
            # or we could explicitly pass output_attentions=False if GetAttnMapLM
            # respects user overrides for its special behavior.
            # Assuming GetAttnMapLM handles this correctly based on phase.
        )
        logits_prefill = prefill_outputs.logits
        past_key_values = prefill_outputs.past_key_values

        # Generate the first token based on the prompt
        next_token_logits = logits_prefill[:, -1, :]  # Logits for the last token of the prompt
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

        # If EOS is generated after the first token, finish early.
        # No decode attentions collected yet as this token came from prefill_outputs.
        if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
            output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
            return output_text, collected_decode_attentions  # Should be empty

        # --- Decode Phase ---
        # Loop to generate remaining (max_new_tokens - 1) tokens, as one is already generated.
        for _ in range(max_new_tokens - 1):
            # The input for this decode step is the token generated in the previous step.
            decode_input_ids = next_token_id

            outputs_decode_step = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True
                # output_attentions is internally True for decode by GetAttnMapLM
            )

            logits_decode_step = outputs_decode_step.logits
            past_key_values = outputs_decode_step.past_key_values  # Update KV cache

            # Collect attention scores from this decode step
            # GetAttnMapLM is expected to provide 'attentions' in outputs_decode_step.
            if hasattr(outputs_decode_step, 'attentions') and outputs_decode_step.attentions is not None:
                # attentions is a tuple (one for each layer) of torch.FloatTensor
                # of shape (batch_size, num_heads, sequence_length=1, key_sequence_length)
                # Detach and move to CPU to save GPU memory, especially for long generations.
                step_attentions = tuple(att.detach().cpu() for att in outputs_decode_step.attentions)
                collected_decode_attentions.append(step_attentions)
            # If GetAttnMapLM guarantees attentions in decode, no 'else' needed.
            # If not, one might log a warning or append a placeholder.

            # Determine the next token
            next_token_logits = logits_decode_step[:, -1, :]  # Logits for the newly generated token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token to the sequence
            current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

            # Check for EOS token
            if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
                break

    output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
    return output_text, collected_decode_attentions


def load_data(data_path, sample_num=10):
    data_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num:
                    break
                d = json.loads(line.strip())
                data_list.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {data_path} on line {i + 1}: {e}")
        # Optionally, continue loading other lines or return partially loaded data
        return data_list
    return data_list


def cumsum_attn_score_for_each_layer(attn_map_at_one_decode_step):
    processed_layers_attn = []
    if attn_map_at_one_decode_step is None:
        return []

    for layer_attn in attn_map_at_one_decode_step:
        # layer_attn expected shape: (batch_size, num_heads, 1, key_len)

        # Ensure it's 4D, though typical decode attention might be (batch, heads, 1, key_len)
        if layer_attn.ndim != 4:
            raise ValueError(f"Expected 4D attention tensor, got {layer_attn.ndim}D with shape {layer_attn.shape}")

        # Squeeze the query_length dimension (dim 2), assuming it's always 1 in decode.
        # If query_length > 1, this logic would need adjustment (e.g., mean over query_length too, or process each query pos).
        if layer_attn.shape[2] == 1:
            squeezed_attn = layer_attn.squeeze(2)  # Shape: (batch_size, num_heads, key_len)
        else:
            raise ValueError(f"attn shape: {layer_attn.shape}")

        # 1. Average over the head dimension (dim 1 of squeezed_attn)
        avg_head_attn = torch.mean(squeezed_attn, dim=1)  # Shape: (batch_size, key_len)
        # 2. Calculate cumulative sum along the key_sequence_length dimension (last dimension)
        cumsum_attn = torch.cumsum(avg_head_attn, dim=-1)  # Shape: (batch_size, key_len)
        processed_layers_attn.append(cumsum_attn)

    return processed_layers_attn


def vis_cumsum_attn_for_each_layer_each_step(step_i, layer_i, attn_cumsum_tensor, output_dir="visualizations"):
    if attn_cumsum_tensor is None or attn_cumsum_tensor.numel() == 0:
        print(f"  Visualization skipped for step {step_i + 1}, layer {layer_i + 1}: Empty tensor.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}. Visualizations will not be saved.")
            return

    if attn_cumsum_tensor.shape[0] > 1:
        print(
            f"  Warning: attn_cumsum_tensor has batch size {attn_cumsum_tensor.shape[0]} for step {step_i + 1}, layer {layer_i + 1}. Visualizing first batch element only.")

    data_to_plot = attn_cumsum_tensor[0].cpu().float().numpy()  # Shape: (key_sequence_length,)

    if data_to_plot.ndim != 1:
        print(
            f"  Visualization skipped for step {step_i + 1}, layer {layer_i + 1}: Expected 1D data after batch selection, got {data_to_plot.ndim}D.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(data_to_plot)
    plt.title(f"Cumulative Attention: Decode Step {step_i + 1}, Layer {layer_i + 1}")
    plt.xlabel("Key Sequence Position (Token Index in KV Cache)")
    plt.ylabel("Cumulative Attention Score (Averaged over Heads)")
    plt.grid(True)

    filename = f"cumsum_step_{step_i + 1}_layer_{layer_i + 1}.jpg"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=1000)
        print(f"    Saved visualization to {filepath}")
    except Exception as e:
        print(f"    Error saving plot to {filepath}: {e}")
    plt.close()  # Close the figure to free memory


def cal_attn_map_similarity(attn_cumsum_tensor_layer_i, attn_cumsum_tensor_layer_j):
    avg_head_attn_i = torch.mean(attn_cumsum_tensor_layer_i.squeeze(), dim=0)
    avg_head_attn_j = torch.mean(attn_cumsum_tensor_layer_j.squeeze(), dim=0)
    assert avg_head_attn_i.ndim == 1
    vi, idx_i = torch.topk(avg_head_attn_i, k=256, dim=-1)
    return torch.sum(avg_head_attn_j[idx_i]).item()


def vis_layer_similarity_matrix(
        similarity_matrix,
        output_dir="visualization",
        dpi=1000,
        show_values=True,  # 新增：是否在单元格中显示数值
        annot_fmt=".1f",  # 新增：数值的格式化字符串 (例如, ".2f" 表示两位小数)
        annot_fontsize=6,  # 新增：标注数值的字体大小
        text_color_threshold=0.5  # 新增：决定文字颜色的阈值 (基于归一化的单元格颜色)
):
    """Visualizes the layer-wise attention similarity matrix as a heatmap
       with optional display of similarity values on each cell."""
    plt.figure(figsize=(12, 10))

    # 确保 similarity_matrix 是 NumPy array 并且在 CPU 上
    if isinstance(similarity_matrix, torch.Tensor):
        matrix_np = similarity_matrix.detach().cpu().numpy()
    elif isinstance(similarity_matrix, np.ndarray):
        matrix_np = similarity_matrix
    else:
        raise TypeError("similarity_matrix must be a PyTorch Tensor or a NumPy array.")

    # 使用 aspect='auto' 并让 imshow 确定 vmin/vmax, 或者可以固定它们。
    # im 对象用于后续获取颜色范围
    im = plt.imshow(matrix_np, cmap='viridis', aspect='auto')

    plt.colorbar(label="Attention Similarity Score")
    plt.title(f"Layer Attention Similarity")
    plt.xlabel("Layer Index (j)")
    plt.ylabel("Layer Index (i)")

    num_layers = matrix_np.shape[0]
    # 根据层数调整刻度以提高可读性
    if num_layers > 0:
        ticks = list(range(num_layers))
        tick_labels = [str(t + 1) for t in ticks]  # 1-indexed 标签
        # 如果层数过多，可以考虑减少刻度标签数量或旋转
        tick_fontsize = 10
        if num_layers > 20:
            tick_fontsize = 8
        if num_layers > 30:  # 对于非常多的层，可能需要更智能的刻度处理
            step = max(1, num_layers // 15)  # 每隔 step 个显示一个刻度
            ticks = list(range(0, num_layers, step))
            tick_labels = [str(t) for t in ticks]

        plt.xticks(ticks, tick_labels, fontsize=tick_fontsize)
        plt.yticks(ticks, tick_labels, fontsize=tick_fontsize)

    # 在每个单元格上显示数值
    if show_values and num_layers > 0:
        # 获取 imshow 使用的颜色范围，用于决定文字颜色
        # vmin, vmax = im.get_clim() # 这是推荐的方式
        # 为了简化，如果你的 similarity_matrix 值总是在一个已知范围 (例如0-1),
        # 你可以直接使用那个范围。否则，get_clim() 更鲁棒。
        # 假设 similarity_matrix 的值主要在 0-1 之间，viridis 颜色映射从暗到亮
        # 如果 imshow 自动调整 vmin/vmax，我们需要用 get_clim()
        actual_vmin, actual_vmax = im.get_clim()

        for i in range(num_layers):
            for j in range(num_layers):
                value = matrix_np[i, j]

                # 根据单元格的值和颜色映射的范围决定文字颜色
                # 将当前值归一化到颜色映射的范围内
                if actual_vmax == actual_vmin:  # 避免除以零，如果所有值都相同
                    normalized_value = 0.5  # 可以是0或1，这里取中间
                else:
                    normalized_value = (value - actual_vmin) / (actual_vmax - actual_vmin)

                # Viridis: 0 (深紫) -> 0.5 (绿色) -> 1 (黄色)
                # 如果归一化后的值小于阈值 (例如0.5)，则背景较暗，用浅色文字
                # 否则背景较亮，用深色文字
                text_color = "white" if normalized_value < text_color_threshold else "black"

                plt.text(j, i, format(value, annot_fmt),
                         ha="center", va="center",
                         color=text_color, fontsize=annot_fontsize)

    filename = f"Layer Attention Similarity.jpg"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        # bbox_inches='tight' 确保标签不会被裁剪
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"    Saved layer similarity heatmap to {filepath}")
    except Exception as e:
        print(f"    Error saving layer similarity plot to {filepath}: {e}")
    finally:
        plt.close()


if __name__ == '__main__':
    MODEL_PATH = "../models/Llama-3-8B-Instruct-262k"
    DATA_PATH = "../OmniKV/benchmark/long_bench/data/hotpotqa.jsonl"
    SAMPLE_NUM = 10
    MAX_NEW_TOKENS_GENERATION = 20

    print(f"Loading model and tokenizer from: {MODEL_PATH}")
    model, tkn = load_model_and_tokenizer(MODEL_PATH)

    # 2. Load data
    print(f"Loading data from: {DATA_PATH} (first {SAMPLE_NUM} samples)")
    data_samples = load_data(DATA_PATH, SAMPLE_NUM)
    if not data_samples:
        print("No data loaded. Exiting.")
    else:
        print(f"Loaded {len(data_samples)} samples.")

    layers_sim = torch.zeros(model.config.num_hidden_layers, model.config.num_hidden_layers, device="cpu")
    sim_cal_step = 0
    for i, sample in enumerate(data_samples):
        input_text = sample

        print(f"Input text: {input_text}\n")

        # 3. Greedy decode and get attention scores
        output_text, collected_attentions = greedy_decode(input_text, model, tkn,
                                                          max_new_tokens=MAX_NEW_TOKENS_GENERATION)
        output_text = input_text + output_text

        print(f"Generated text: {output_text}")
        print(f"Number of decode steps for which attentions were collected: {len(collected_attentions)}")

        # 4. Process attention scores for each step (if any were collected)
        all_steps_processed_attns = []
        if collected_attentions:
            for step_idx, attn_map_one_step in enumerate(collected_attentions):
                print(f"  Processing attentions for decode step {step_idx + 1}:")
                # attn_map_one_step is a tuple of layer attentions for this decode step
                # Each layer_attn tensor is (batch_size, num_heads, 1, key_len_at_this_step)
                # processed_layer_attns_for_step = cumsum_attn_score_for_each_layer(attn_map_one_step)
                # all_steps_processed_attns.append(processed_layer_attns_for_step)
                #
                # # Visualize for this step
                # for layer_idx, cumsum_attn_tensor in enumerate(processed_layer_attns_for_step):
                #     print(f"    Layer {layer_idx + 1} processed cumsum attention shape: {cumsum_attn_tensor.shape}")
                #     vis_cumsum_attn_for_each_layer_each_step(
                #         step_idx, layer_idx, cumsum_attn_tensor
                #     )
                sim_cal_step += 1
                for layer_i in range(model.config.num_hidden_layers):
                    for layer_j in range(layer_i, model.config.num_hidden_layers):
                        sim_ij = cal_attn_map_similarity(attn_map_one_step[layer_i], attn_map_one_step[layer_j])
                        layers_sim[layer_i, layer_j] += sim_ij
                        # 可视化
        else:
            print("No attention scores were collected (e.g., max_new_tokens was small or EOS met early).")
        # `all_steps_processed_attns` is now a list (decode steps) of lists (layers)
        # where each inner element is a tensor of shape (batch_size, key_len_at_that_step)

    layers_sim /= sim_cal_step
    vis_layer_similarity_matrix(layers_sim)
