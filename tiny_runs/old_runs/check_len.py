import json
from transformers import AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained("/input/jitai/huggingface/hub/Lourdle/Llama-3-8B-Instruct-262k")
    base_path = '/'
    files = ['small', 'medium', 'large']
    for f in files:
        with open(f"{f}.jsonl", 'r', encoding='utf-8') as _in:
            avg_context_len, avg_ref_len = 0, 0
            cnt = 0
            for line in _in:
                cnt += 1
                d = json.loads(line)
                ctx_ids = tok.encode(d['context'])
                ref_ids = tok.encode(d['refered_chunk'])
                avg_context_len += len(ctx_ids)
                avg_ref_len += len(ref_ids)
            print(f, avg_context_len / cnt, avg_ref_len / cnt, cnt)


main()
