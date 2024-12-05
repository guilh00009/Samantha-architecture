from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('k050506koch/GPT3-dev-125M', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('k050506koch/GPT3-dev-125M')
tokenizer.pad_token_id = tokenizer.eos_token_id
print("\n", tokenizer.decode(model.generate(tokenizer.encode("What is the capital of France?", return_tensors='pt'),
    max_length=128, temperature=0.7, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3,
    num_return_sequences=1, do_sample=True)[0], skip_special_tokens=True))
