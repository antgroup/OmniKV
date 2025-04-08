import json
import os
import time
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig
from benchmark.infinite_bench.src.eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from benchmark.infinite_bench.src.args import parse_args
from infer import get_any_chat_api
from tiny_tools.read_json import read_config
from tiny_tools.tensor_tools import idx_tracer

MAX_POSITION_ID = 50000  # 后面会被覆盖掉
TRUNCATE_LEN = 50000
GENERATION_CONFIG = None


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(
        model,
        tok: AutoTokenizer,
        input_text: str,
        max_tokens: int,
        verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    output = model(input_text, generation_config=GENERATION_CONFIG,
                   skip_special_tokens=True)

    print("Chunked generation:", output)
    return output


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    config = read_config(args.config_path)
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    if config.get("cot", False):
        max_tokens *= 2

    chat, tok, max_len, o_dict = get_any_chat_api(args.config_path)
    if max_len is not None:
        # global TRUNCATE_LEN, MAX_POSITION_ID
        TRUNCATE_LEN = max_len
        MAX_POSITION_ID = max_len

    if (__ := o_dict.get('eos_token_id', None)) is not None:
        tok.eos_token_id = __
    print("EOS is", tok.eos_token)

    if 'sum' in args.task:
        GENERATION_CONFIG = GenerationConfig(
            temperature=0.8, top_p=0.95, max_new_tokens=max_tokens, do_sample=True, eos_token_id=tok.eos_token_id
        )
    else:
        GENERATION_CONFIG = GenerationConfig(
            max_new_tokens=max_tokens, do_sample=False, eos_token_id=tok.eos_token_id
        )

    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    st = time.time()
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== {data_name} Example {i}/{args.stop_idx} ======")
        pred = get_pred(
            chat, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        if os.environ.get('SAVE_SELECTED_IDX', False):
            idx_tracer.append_out((pred, preds[-1]['ground_truth']))
            idx_tracer.save_idx()
        print(f"#### std_answer {get_answer(eg, data_name)}")
        dump_jsonl(preds, output_path)

    print("final total time", time.time() - st)
