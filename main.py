from huggingface_hub import login
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

embedding_size = model.model.embed_tokens.weight.shape[1]
print(f'Embedding size: {embedding_size}')

xnli_dataset = load_dataset("xnli", 'en', split="test[:5]")
xnli_metric = load("xnli")


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}


def preprocess_function(examples):
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
            examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128
        )

encoded_dataset = xnli_dataset.map(preprocess_function, batched=False)


def compute_metric(dataset):
    predictions = []
    references = []

    for example in dataset:

        output = classifier(
            f"{example['premise']} {tokenizer.sep_token} {example['hypothesis']}"
        )
        
        predicted_label_obj = max(output[0], key=lambda x: x['score'])
        print('debug')
        
        
        predicted_label_id = {v: k for k, v in label_mapping.items()}[predicted_label_obj['label']]
        predictions.append(predicted_label_id)
        references.append(example["label"])
    return xnli_metric.compute(predictions=predictions, references=references)

accuracy = compute_metric(encoded_dataset)