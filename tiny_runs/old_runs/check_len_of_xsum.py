from tiny_tools.read_json import read_dataset_in_jsonl

x = read_dataset_in_jsonl('baselines/raw_h2o/data/summarization_data/xsum_5shot.jsonl')
print(x[0].keys())
# print(x[0]['article'])
lens = [len(t['article']) for t in x]
avg_len = sum(lens) / len(lens) / 4
print(avg_len)
