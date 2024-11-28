from transformers import BertTokenizer
from datasets import Dataset
class TextTokenizer:
    def __init__(self,text,label):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text=text
        self.label = label
    def tokenize(self,batch):
        return self.tokenizer(batch["text"], return_tensors='pt', padding=True)
    def get_dataset(self):
        dataset = Dataset.from_dict({'text': self.text, 'labels': self.label})
        dataset = dataset.map(self.tokenize, batched=True)
        return dataset