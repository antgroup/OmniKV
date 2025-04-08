import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 加载模型和分词器，这里使用的是LLaMA模型的假设名称
# 请确保你替换成你所需要的LLaMA模型名称
model_name = "/input/jitai/huggingface/hub/shakechen/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             attn_implementation='flash_attention_2',
                                             torch_dtype=torch.float16)

# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_response(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=max_length, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def chat():
    print("开始聊天! 输入'exit'来退出。")
    while True:
        prompt = input("你: ")
        if prompt.lower() == 'exit':
            break
        response = generate_response(prompt)
        print(f"LLaMA: {response}")


if __name__ == "__main__":
    chat()
