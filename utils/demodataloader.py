from data_processor import TextDataProcessor,ImageDataProcessor
from dataloaders import MVSADataLoaders


label_path="/Users/uttamsingh/Documents/Graduate/Fall2024/MVSA_Single/labelResultAll.csv"
text_path= "/Users/uttamsingh/Documents/Graduate/Fall2024/MVSA_Single/data/texts"
image_path = "/Users/uttamsingh/Documents/Graduate/Fall2024/MVSA_Single/data/images"
texts,labels = TextDataProcessor(label_path,text_path).get_text_with_same_labels()
text_train_loader,text_test_loader = MVSADataLoaders().get_text_dataloader(texts,labels)
image_list,labels = ImageDataProcessor(label_path,image_path).get_image_with_same_labels()
image_train_loader,image_test_loader = MVSADataLoaders().get_image_dataloader(image_list,labels)
# print(text_train_loader)
# for batch_idx, batch_data in enumerate(text_train_loader):
#     print(f"Batch {batch_idx + 1}: {batch_data}")