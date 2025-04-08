import json

with open("benchmark/long_bench/data/lsht.jsonl", 'r', encoding='utf-8') as _in:
    for line in _in:
        d = json.loads(line)
        print(d['input'][-50:])
        print(len(d['input']))
        print(d['input'][:50])
        print('---')
        print(d['context'][-50:])
        print(len(d['context']))
        print(d['context'][:50])
        break
