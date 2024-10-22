from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model import *

#model_path = "k050506koch/GPT3-dev"
model_path = "gpt3-small-fineweb"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = CustomGPT2LMHeadModel.from_pretrained(model_path)
print(model)