from datasets import load_dataset, interleave_datasets
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from itertools import islice

prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Translate English to German. Maintain the original meaning, tone, and formatting. Provide only the translation without additional text or explanations.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

model_name = "meta-llama/Llama-3.2-1B-Instruct"

def load_datasets():
    flores = load_dataset('openlanguagedata/flores_plus', split='devtest', streaming=True)
    flores_redux = flores.select_columns(['id', 'iso_639_3', 'text'])
    eng_flores = flores_redux.filter(lambda x: x['iso_639_3'] in ['eng'])
    deu_flores = flores_redux.filter(lambda x: x['iso_639_3'] in ['deu'])
    return eng_flores, deu_flores

def setup_pipeline():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_pipeline = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )
    return translation_pipeline

def preprocess_function(examples):
    return prompt.format(examples['text'])

def preprocess_dataset_with_prompt(dataset):
    return dataset.map(preprocess_function, batched=True)

def run_translation_eval():
    eng_flores, deu_flores = load_datasets()
    translation_pipeline = setup_pipeline()
    eng_prompts = preprocess_dataset_with_prompt(eng_flores)

    translations = translation_pipeline(eng_prompts, return_full_text=False)
    
    
    translation_metric = load('sacrebleu')


if __name__ == '__main__':
    run_translation_eval()