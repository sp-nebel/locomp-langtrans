import copy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
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

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='lora_only',
    use_rslora=True,
    task_type='CAUSAL_LM',
)

training_args = TrainingArguments(
    output_dir=f'./xnli_lora_output',
    learning_rate=5e-4, # research optimal learning rate for lora training
    num_train_epochs=1,
    save_total_limit=3,
    eval_strategy='epoch',
    logging_steps=10,
    remove_unused_columns=True,
    logging_dir='./logs',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2, # research optimal gradient accumulation steps for lora training
    # consider fp16 training, or check if it is default
    eval_on_start=True,
)

def prepare_tokenized_xnlis(tokenizer):
    xnli_train = load_dataset('xnli', 'en', split='train[:635]+validation[:50]', streaming=False)
    xnli_train = preprocess_dataset(xnli_train, tokenizer)
    return xnli_train.train_test_split(train_size=635, shuffle=False)

def preprocess_dataset(dataset, tokenizer):
    def create_prompt_dict(example):
        prompt = create_label_prompt(example['premise'], example['hypothesis'], example['label'])
        return {'prompt': prompt}

    def tokenize_function(examples):
        model_inputs = tokenizer(
          examples['prompt'],
          return_tensors=None,
          padding=False,
          truncation=True,
          max_length=1024,
      )

        labels = copy.deepcopy(model_inputs['input_ids'])

        model_inputs['labels'] = labels
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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            } 
    )

    xnlis = prepare_tokenized_xnlis(tokenizer)
    model = setup_peft_model(model_name, lora_config)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors='pt',
        padding=True,
        label_pad_token_id=IGNORE_INDEX
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=xnlis['train'],
        eval_dataset=xnlis['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(f'./{model_name}_xnli_lora')

if __name__ == '__main__':
    run_training_experiment()