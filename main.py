from huggingface_hub import login
from datasets import load_dataset
from evaluate import load
from transformers import AutoModel, AutoTokenizer, pipeline

prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Task: Classify the relationship between two sentences (premise and hypothesis).

Rules:
- You MUST respond with EXACTLY ONE of these labels: "entailment", "contradiction", "neutral"
- DO NOT include any other words, punctuation, or explanations
- DO NOT add newlines or spaces before/after the label

Example 1:
Premise: The cat is sleeping.
Hypothesis: The animal is resting.
Output: entailment

Example 2:
Premise: The sky is blue.
Hypothesis: The sky is red.
Output: contradiction

Example 3:
Premise: The man is walking.
Hypothesis: He is wearing a hat.
Output: neutral
<|eot_id|><|start_header_id|>user<|end_header_id|>

Now classify:
Premise: {premise}
Hypothesis: {hypothesis}
Output: <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

xnli_dataset = load_dataset("xnli", 'en', split="test")
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
        
        print(output)
        
        predictions.append(compute_label(output[0]['generated_text']))
        references.append(example['label'])
    return xnli_metric.compute(predictions=predictions, references=references), predictions, references

def compute_label(input: str):
    if 'entailment' in input:
        return 0
    elif 'neutral' in input:
        return 1
    elif 'contradiction' in input:
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
