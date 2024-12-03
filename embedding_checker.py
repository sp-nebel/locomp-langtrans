from transformers import AutoModel

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModel.from_pretrained(model_name)

#write embedding size to file
embedding_size = model.embed_tokens.weight.shape[1]
with open('output.txt', 'w') as f:
    f.write(f'Model: {model_name} Embedding size: {embedding_size}\n')