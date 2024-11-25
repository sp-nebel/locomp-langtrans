from huggingface_hub import login
from evaluate import load, evaluator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

login()

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

xnli_metric = load("xnli")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
xnli_evaluator = evaluator("xnli")



def preprocess_function(examples):
    return [f"Premise: {p} Hypothesis: {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]

xnli_dataset = load_dataset("xnli", split="test[:10%]")

def eval(model, tokenizer, dataset, metric, evaluator):
    results = evaluator.compute(model, tokenizer, dataset, metric, input_column="premise", label_column="label", hypothesis_column="hypothesis", split="test")
    return results

results = eval(model, tokenizer, xnli_dataset, xnli_metric, xnli_evaluator)

with open('./results.txt', 'w') as f:
    f.write(results)
