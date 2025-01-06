from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

system_prompt = '''Task: Classify the relationship between two sentences (premise and hypothesis).

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
Output: neutral'''

prompt_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Now classify:
Premise: {premise}
Hypothesis: {hypothesis}
Output: <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

model_name = "meta-llama/Llama-3.2-1B-Instruct"

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='lora_only',
    use_rslora=True,
    modules_to_save=["decode_head"]
)

training_args = TrainingArguments(
    output_dir=f'{model_name}_xnli_lora',
    learning_rate=5e-4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    save_total_limit=3,
    eval_strategy='epoch',
    logging_steps=5,
    remove_unused_columns=True,
    logging_dir='./logs',
    use_cpu=True
)

xnli_metric = load('xnli')

def prepare_tokenized_xnlis(tokenizer):
    xnli = load_dataset('xnli', 'en', split='train[:10]', streaming=False)
    xnli = preprocess_dataset(xnli, tokenizer)
    xnlis = xnli.train_test_split(test_size=0.1)
    return xnlis

def preprocess_dataset(dataset, tokenizer):
    def apply_prompt(example):
        return {'prompt': create_prompt(example['premise'], example['hypothesis'])}

    prompt_dataset = dataset.map(
        apply_prompt,
        remove_columns=['premise', 'hypothesis'],
        batched=False
    )

    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding='max_length', truncation=True)
    
    tokenized_dataset = prompt_dataset.map(tokenize_function, batched=True)

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return tokenized_dataset

def create_prompt(premise, hypothesis):
    return prompt_template.format(system_prompt=system_prompt, premise=premise, hypothesis=hypothesis)

def setup_peft_model(model_name, config):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_model = get_peft_model(model, config)
    return lora_model

def compute_metrics(eval_pred):
    #TODO: Implement this
    pass

def run_training_experiment():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    xnlis = prepare_tokenized_xnlis(tokenizer)
    model = setup_peft_model(model_name, lora_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=xnlis['train'],
        eval_dataset=xnlis['test']
    )

    trainer.train()

    model.save_pretrained(f'{model_name}_xnli_lora')

if __name__ == '__main__':
    run_training_experiment()