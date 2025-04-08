import json


def load_dataset(path, mode='r'):
    try:
        data = [json.loads(line) for line in open(path, mode, encoding="utf-8")]
    except FileNotFoundError:
        return []
    return data


datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader", "gov_report", "qmsum",
            # "multi_news",
            "vcsum", "trec", "triviaqa",
            # "samsum",
            "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset(f'benchmark/long_bench/data/{dataset}.jsonl', 'r')
    # data_all = [data_sample for data_sample in data]
    print(dataset, len(data))
print('-------')
for dataset in datasets:
    data = load_dataset(f'benchmark/long_bench/pred/my_model/configs/tests/0709.6.baseline.json/{dataset}.jsonl', 'r')
    # data_all = [data_sample for data_sample in data]
    print(dataset, len(data))