from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.customDatasets import ImageDataset,TextDataset
from utils.data_processor import TextDataProcessor,ImageDataProcessor

class MVSADataLoaders:
    def __init__(self):
        pass  

    def get_image_dataloader(self,image_paths,labels):
        train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)
        train_dataset = ImageDataset(train_paths, train_labels)
        test_dataset = ImageDataset(test_paths, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return train_loader, test_loader
    def get_text_dataloader(self,tokenized_data,labels):
        train_data, test_data, train_labels, test_labels = train_test_split(
    tokenized_data, labels, test_size=0.2, random_state=42
)
        train_dataset = TextDataset(train_data, train_labels)

        test_dataset = TextDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return train_loader, test_loader