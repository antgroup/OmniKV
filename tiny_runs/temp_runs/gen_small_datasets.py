import json

# path = 'benchmark/infinite_bench/data/small_number_string'
path = 'benchmark/infinite_bench/data/small_kv_retrieval'
input_file = '/input/fangshuai.fs/huggingface/repo/InfiniteBench/kv_retrieval.jsonl'
output_file = f'{path}.jsonl'


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i % 20 == 0:  # 每隔10行取一行
                try:
                    # 解析JSON行
                    data = json.loads(line.strip())
                    # 将数据写入输出文件
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i + 1}. Skipping this line.")


# 执行处理
process_file(input_file, output_file)

print(f"Processing complete. Output written to {output_file}")
