import re
from fire import Fire


def get_number(line):
    # print(line)
    obj = re.search(r"time (\d+\.\d+)", line)
    return float(obj.group(1))


def compute(log_path):
    prefill_timer = []
    decode_timer = []
    with open(log_path, 'r', encoding='utf-8') as _in:
        temp_sum = 0
        for line in _in:
            if 'prefill' in line:
                t = get_number(line)
                prefill_timer += [t]
                temp_sum += t
            elif 'decoding' in line:
                t = get_number(line)
                decode_timer += [get_number(line)]
                temp_sum += t
            elif 'chat time' in line:
                print(line)
                print("temp sum", temp_sum)
                temp_sum = 0

    print("prefill time", sum(prefill_timer) / len(prefill_timer))
    print("decode time", sum(decode_timer) / len(decode_timer))


if __name__ == '__main__':
    Fire(compute)
