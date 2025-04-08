import json
import re


def read_log(log_path, re_pat):
    with open(log_path, 'r', encoding='utf-8') as _input:
        for line in _input:
            if o := re.search(re_pat, line):
                yield o
