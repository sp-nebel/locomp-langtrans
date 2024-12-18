from datasets import load_dataset, interleave_datasets
from evaluate import load
from transformers import AutoModel, AutoTokenizer, pipeline

prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
'''

model_name = "meta-llama/Llama-3.2-3B-Instruct"

def run_translation_eval():
    flores = load_dataset('openlanguagedata/flores_plus', split='devtest', streaming=True)
    eng_flores = flores.filter(lambda x: x['iso_639_3'] in ['eng'])
    deu_flores = flores.filter(lambda x: x['iso_639_3'] in ['deu'])
    eng_deu_flores = interleave_datasets([eng_flores, deu_flores]) #also works with non-streaming datasets
    for obj in eng_deu_flores.take(4):
        print(obj)
    translation_metric = load('translation')


if __name__ == '__main__':
    run_translation_eval()