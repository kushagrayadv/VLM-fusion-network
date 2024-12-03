from transformers import DebertaV2Tokenizer
from datasets import Dataset

class TextTokenizer:
    def __init__(self):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
        

    def preprocess_text(self,text):      
        encoding = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)