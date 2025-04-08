import json
import os
import random
from infer import get_any_chat_api
from tiny_tools.read_json import read_config
from fire import Fire
from tiny_tools.log import logger
from tiny_tools.tensor_tools import idx_tracer

colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "cyan",
          "magenta", "indigo", "violet", "beige", "maroon", "navy", "olive", "teal", "lime", "silver", "gold"]


def make_one(dict_len):
    prompt = "Let's play a game. You have a dict and a mathematical addition equation. " \
             "The keys of a dictionary can be any number. " \
             "You need to find the corresponding key value in the dictionary " \
             "after performing the addition and output the value corresponding to that key."

    d = {i: random.choice(colors) for i in range(dict_len)}
    final_answer = random.randint(0, dict_len - 1)
    num_a = random.randint(0, final_answer)
    num_b = final_answer - num_a
    prompt += f" The Dict is {d}\n"
    prompt += f"The equation is {num_a} + {num_b} = ?\n"
    prompt += f"MUST first output the result of the addition, " \
              f"and then answer the corresponding value based on the result."
    return prompt, d[final_answer], final_answer


def make_dataset(num_samples=150):
    data_lis = []
    lens = [40, 50, 60, 70, 80, 90, 100, 140, 180, 200]
    nl = len(lens)
    for i in range(num_samples):
        res = make_one(lens[i % nl])
        data_lis.append({'prompt': res[0], 'answers': [res[1]]})
    with open('benchmark/long_bench/data/2stage.jsonl', 'w', encoding='utf-8') as _out:
        for d in data_lis:
            # print(json.dumps(d).replace("'", ''), file=_out)
            print(json.dumps(d), file=_out)


def test(cfg_path, n_sample):
    chat, tok, max_len, o_dict = get_any_chat_api(cfg_path)
    idx = 0
    for n in [20, 100, 300, 500, 800, 1000, 10000]:
        find_cnt, cal_cnt = 0, 0
        for j in range(n_sample):
            prompt, answer, num = make_one(n)
            out = chat(prompt, max_new_tokens=30, skip_special_tokens=False)
            if os.environ.get('SAVE_SELECTED_IDX', False):
                idx_tracer.append_out((prompt, out))
                idx_tracer.save_idx()
            # logger.warning(f"--- len {len(tok.encode(p))}")
            logger.info(f"---ans: {answer} ---{idx}:{out}")
            if str(num) in out:
                cal_cnt += 1
            if answer in out:
                find_cnt += 1
                logger.warning(f"right in {idx}")
            idx += 1
        logger.warning(f"When n = {n}, acc = {find_cnt/n_sample}, cal acc = {cal_cnt/n_sample}")


if __name__ == '__main__':
    # Fire(test)
    make_dataset(100)
