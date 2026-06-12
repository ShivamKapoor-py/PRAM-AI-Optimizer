from transformers import AutoModel

AutoModel.from_pretrained("distilbert-base-uncased", cache_dir="./hf_model")