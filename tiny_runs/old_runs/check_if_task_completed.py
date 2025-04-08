import json
import os
import fnmatch
from pprint import pprint


def count_lines_in_jsonl_files(path):
    line_counts = {}
    other = []
    # 遍历指定路径下的所有文件
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jsonl'):
            file_path = os.path.join(root, filename)

            # 使用with语句确保文件正确关闭
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = 0
                for line in file:
                    try:
                        # 尝试将每一行解析为json，如果失败则跳过这一行
                        d = json.loads(line.strip())
                        d['pred'] = ''
                        other += [json.dumps(d, ensure_ascii=False)]
                        lines += 1
                    except json.JSONDecodeError:
                        pass

                line_counts[file_path] = lines

    return line_counts, other


print("First, Check Num of Samples")
_, o1 = count_lines_in_jsonl_files("benchmark/long_bench/pred/my_model/configs/tests/0712.2.json/")
pprint(_)
_, o2 = count_lines_in_jsonl_files("benchmark/long_bench/pred/my_model/configs/tests/0709.6.ours.json/")
pprint(_)
o1 = set(o1)
o2 = set(o2)
# if len(o1) > len(o2):
#     o1 = o1[:len(o1) // 2]
assert o1 == o2
_, o3 = count_lines_in_jsonl_files("benchmark/long_bench/pred/my_model/configs/tests/0709.6.baseline.json/")
pprint(_)
o3 = set(o3)
assert o2 == o3
