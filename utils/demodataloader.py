from utils.data_processor import TextDataProcessor,ImageDataProcessor
from utils.text_tokenizer import TextTokenizer
from utils.dataloaders import MVSADataLoaders

label_path="/Users/uttamsingh/Documents/Graduate/Fall2024/MVSA_Single/labelResultAll.csv"
text_path= "/Users/uttamsingh/Documents/Graduate/Fall2024/MVSA_Single/data/texts"

texts,labels = TextDataProcessor(label_path,text_path).get_text_with_same_labels()
dataset = TextTokenizer(texts,labels).get_dataset()
_train_loader,test_loader = MVSADataLoaders().get_text_dataloader(dataset,labels)