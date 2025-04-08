from transformers import AutoTokenizer

tkn = AutoTokenizer.from_pretrained("/input/jitai/huggingface/hub/Lourdle/Llama-3-8B-Instruct-262k")
out_ids = tkn(["you are </s><|eot_id|><|eot_id|>"], return_tensors='pt')['input_ids']
print(out_ids)
print(tkn.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
