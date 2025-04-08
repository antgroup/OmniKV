from infer import get_ntk_llama_chat_api_with_tokenizer_bs1

r = get_ntk_llama_chat_api_with_tokenizer_bs1("configs/tests/0717.2.json")
chat = r[0]
while x := input("your turn:"):
    print(chat(x, max_new_tokens=50))
