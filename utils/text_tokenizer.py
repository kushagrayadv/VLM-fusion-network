from transformers import BertTokenizer
from utils.data_processor import TextDataProcessor
from utils.customDatasets import Dataset
class TextTokenizer:
    def __init__(self,text_path,label_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_path=text_path
        self.label_path = label_path
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt', padding=True)
    def get_dataset(self):
        texts,labels = TextDataProcessor(self.label_path,self.text_path).get_text_with_same_labels()
        dataset = Dataset.from_dict({'text': texts, 'labels': labels})
        dataset = dataset.map(self.tokenize, batched=True)
        return dataset