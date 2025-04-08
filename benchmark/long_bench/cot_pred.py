from fire import Fire
from tiny_tools.log import create_logger
from tiny_tools.read_json import read_config
from infer import get_any_chat_api
from benchmark.long_bench.pred import load_dataset
from transformers import GenerationConfig, set_seed
from tqdm import tqdm, trange
from configs import template_for_chat

set_seed(42)
dataset_list = ["2wikimqa", "hotpotqa", "2stage"]
data_format = "Answer the question based on the given passages. " \
              "\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. " \
              "\n\nQuestion: {input}\n"


def truncate_prompt(prompt, tokenizer, max_length):
    # truncate to fit max_length (we suggest truncate in the middle,
    # since the left and right side may contain crucial instructions)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                  tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))
    return prompt


def main(cfg_path, test_num=0, base_data_path='benchmark/long_bench/data/', verbose=False, task=None):
    global dataset_list
    logger = create_logger("cot_pred", f"debug_logs/cot_pred_{cfg_path.replace('/', '--')}_{task}.log")
    cfg = read_config(cfg_path)
    cot = cfg['cot']
    use_sink = cfg.get('use_sink_cache', False)
    if use_sink:
        template_for_chat.prompt_template['llama_cot'] += ' First,'
    chat, tkn, max_len, o_dict = get_any_chat_api(cfg_path)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=200)
    acc_lis = []
    if task is not None:
        task = task.split(',')
        for t in task:
            assert t in dataset_list
        dataset_list = task

    for data in dataset_list:
        if data == '2stage':
            if 'yi' in cfg['model_name'].lower():
                gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=50)
            if not cot:
                print("修改内置prompt")
                # 让模型直接回答
                template_for_chat.prompt_template['llama'] += (
                    "The equation result should be used as a key to find the value in the dictionary.\n"
                    "Then, i will directly give the color. The corresponding color is")
                # print(template_for_chat.prompt_template['llama'])
                template_for_chat.prompt_template['yi'] += (
                    "The equation result should be used as a key to find the value in the dictionary.\n"
                    "Then, i will directly give the color. The corresponding color is")
            else:
                template_for_chat.prompt_template['llama_cot'] = template_for_chat.prompt_template['llama_cot_2stage']
                template_for_chat.prompt_template['qwen_cot'] = template_for_chat.prompt_template['qwen_cot_2stage']

        data_lis = load_dataset(f"{base_data_path}{data}.jsonl")
        data_lis = data_lis[-test_num:]
        correct = 0
        for sp in tqdm(data_lis):
            if data != '2stage':
                prompt = data_format.format(**sp)
                prompt = truncate_prompt(prompt, tkn, max_len)
            else:
                prompt = sp['prompt']
                if not cot:
                    # 因为2stage自带cot，所以去掉自带的
                    prompt = prompt.replace(
                        "MUST first output the result of the addition, "
                        "and then answer the corresponding value based on the result.",
                        "Directly answer the corresponding color based on the equation result.")

            llm_out = chat(prompt, generation_config=gen_cfg)
            if verbose:
                if 'input' in sp:
                    logger.info(f"{sp['input']} {'-' * 5} {sp['answers']}")
                else:
                    logger.info(f"... {prompt[-30:]} {'-' * 5} {sp['answers']}")
                logger.info(llm_out)
                logger.info('=' * 10)
            for ans in sp['answers']:
                if ans.lower() in llm_out.lower():
                    correct += 1
                    break

        print(f"For {data} with cot={cot}, Acc = {correct / len(data_lis)}")
        logger.warning(f"For {data} with cot={cot}, Acc = {correct / len(data_lis)}")
        acc_lis += [correct / len(data_lis)]
    logger.warning(f"final acc lis : {acc_lis}")
    return acc_lis


if __name__ == '__main__':
    Fire(main)
