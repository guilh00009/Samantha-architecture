from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import *

#model_path = "k050506koch/GPT3-dev"
model_path = "your_local_model_path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = CustomGPT2LMHeadModel.from_pretrained(model_path)
text = (model.generate(tokenizer.encode("Hello, my name is", return_tensors="pt")))
text2 = tokenizer.decode(text[0], skip_special_tokens=False)
print(text2)