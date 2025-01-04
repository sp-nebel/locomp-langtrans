from transformers import AutoModelForCausalLM
from torchinfo import summary

llama_1b = "meta-llama/Llama-3.2-1B-Instruct"
llama_3b = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(llama_1b)
with open('model_dims.txt', 'a') as f:
    f.write(summary(model, verbose=1))
    f.write(summary(model, verbose=2))
    f.write('\n')

model = AutoModelForCausalLM.from_pretrained(llama_3b)
with open('model_dims.txt', 'a') as f:
    f.write(summary(model, verbose=1))
    f.write(summary(model, verbose=2))
    f.write('\n')
