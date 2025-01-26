from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

system_prompt = '''Task: Classify the relationship between two sentences (premise and hypothesis).

Rules:
- You MUST respond with EXACTLY ONE of these labels: "entailment", "contradiction", "neutral"
- DO NOT include any other words, punctuation, or explanations
- DO NOT add newlines or spaces before/after the label
'''

prompt_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Premise: {premise}
Hypothesis: {hypothesis}
Output: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}'''

model_name = "meta-llama/Llama-3.2-3B-Instruct"

entailment_id = 0
contradiction_id = 2
neutral_id = 1

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='lora_only',
    use_rslora=True,
    modules_to_save=["decode_head"],
    task_type='CAUSAL_LM',
)

training_args = TrainingArguments(
    output_dir=f'./xnli_lora_output',
    learning_rate=5e-4,
    num_train_epochs=5,
    save_total_limit=3,
    eval_strategy='epoch',
    logging_steps=5,
    remove_unused_columns=True,
    logging_dir='./logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

def prepare_tokenized_xnlis(tokenizer):
    xnli_train = load_dataset('xnli', 'en', split='train+test', streaming=False)
    xnli_train = preprocess_dataset(xnli_train, tokenizer)
    return xnli_train.train_test_split(train_size=392702, shuffle=False)

def preprocess_dataset(dataset, tokenizer):
    def create_prompt_dict(example):
        prompt = create_label_prompt(example['premise'], example['hypothesis'], example['label'])
        return {'prompt': prompt}

    def tokenize_function(examples):
      model_inputs = tokenizer(
          examples['prompt'],
          max_length=512,
          truncation=True,
          padding=True
      )

      labels = tokenizer(
          examples['prompt'],
          max_length=512,
          truncation=True,
          padding=True
      )
      model_inputs['labels'] = labels['input_ids']
      return model_inputs

    prompt_dataset = dataset.map(
        create_prompt_dict,
        remove_columns=['premise', 'hypothesis', 'label']
    )
    tokenized_dataset = prompt_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['prompt'])

    return tokenized_dataset

def create_label_prompt(premise, hypothesis, label):
    if label == entailment_id:
        label = 'entailment'
    elif label == contradiction_id:
        label = 'contradiction'
    elif label == neutral_id:
        label = 'neutral'
    else:
        raise ValueError(f'Invalid label: {label}')
    return prompt_template.format(system_prompt=system_prompt, premise=premise, hypothesis=hypothesis, label=label)

def setup_peft_model(model_name, config):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    lora_model = get_peft_model(model, config)
    return lora_model

def run_training_experiment():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' 

    xnlis = prepare_tokenized_xnlis(tokenizer)
    model = setup_peft_model(model_name, lora_config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=xnlis['train'],
        eval_dataset=xnlis['test'],
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(f'./{model_name}_xnli_lora')

if __name__ == '__main__':
    run_training_experiment()