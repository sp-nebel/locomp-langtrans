from evaluate import load
from datasets import load_dataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

xnli = load("xnli")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def preprocess_function(examples):
    return [f"Premise: {p} Hypothesis: {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]

xnli_dataset = load_dataset("xnli", split="test[:10%]")

results = xnli.compute()

