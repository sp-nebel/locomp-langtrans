from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def create_prompt(text: str) -> str:
    """Creates a formatted prompt for translation."""
    system_prompt = "Translate the following text from English to German. Maintain the original meaning, tone, and formatting. Provide only the translation without additional text or explanations. Now translate:"
    return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

model_name = "meta-llama/Llama-3.2-3B-Instruct"

def load_datasets():
    flores = load_dataset('openlanguagedata/flores_plus', split='devtest', streaming=False)
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
        return_full_text=False,
        device=0,
        max_new_tokens=20
    )
    return translation_pipeline

def preprocess_dataset_with_prompt(dataset):
    """Adds translation prompts to dataset examples."""
    def apply_prompt(example):
        if not example['text']:
            return {'processed_input': ''}
        return {'processed_input': create_prompt(example['text'])}
    
    return dataset.map(
        apply_prompt,
        remove_columns=dataset.column_names,
        batched=False
    )

def make_reference_list(dataset):
    returnlist = []
    for text in dataset['text']:
        returnlist.append([text])
    return returnlist

def save_results(result):
    with open('translation_results.txt', 'w') as f:
        f.write(str(result))

def run_translation_eval():
    eng_flores, deu_flores = load_datasets()
    translation_pipeline = setup_pipeline()
    eng_prompts = preprocess_dataset_with_prompt(eng_flores)

    translations = translation_pipeline(eng_prompts['processed_input'], return_full_text=False)
    generated_texts = [translation[0]['generated_text'] for translation in translations]
    
    translation_metric = load('sacrebleu')
    references = make_reference_list(deu_flores)
    result = translation_metric.compute(predictions=generated_texts, references=references)
    save_results(result)


if __name__ == '__main__':
    run_translation_eval()
