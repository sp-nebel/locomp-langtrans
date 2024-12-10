from huggingface_hub import login
from datasets import load_dataset
from evaluate import load
from transformers import AutoModel, AutoTokenizer, pipeline

prompt = '''{premise} Question : {hypothesis} True, False, or Neither?
Answer:'''

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

xnli_dataset = load_dataset("xnli", 'en', split="test", streaming=True)
xnli_metric = load("xnli")


classifier = pipeline("text-generation", model=model_name, tokenizer=tokenizer, device=1, top_k=None)


def compute_metric(dataset):
    predictions = []
    references = []

    for example in dataset:

        output = classifier(
            prompt.format(premise=example['premise'], hypothesis=example['hypothesis']),
            max_new_tokens=5,
            return_full_text=False
        )
                
        predictions.append(compute_label(output[0]['generated_text']))
        references.append(example['label'])
    return xnli_metric.compute(predictions=predictions, references=references), predictions, references

def compute_label(input: str):
    if 'True' in input:
        return 0
    elif 'Neither' in input:
        return 1
    elif 'False' in input:
        return 2
    else:
        return None

accuracy, predictions, references = compute_metric(xnli_dataset)

with open('output.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write('Predictions: \n')
    for pred in predictions:
        f.write(f'{pred}\n')
    f.write('References: \n')
    for ref in references:
        f.write(f'{ref}\n')
