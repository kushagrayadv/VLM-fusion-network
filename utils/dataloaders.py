from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.customDatasets import ImageDataset, TextDataset
from utils.data_processor import TextDataProcessor, ImageDataProcessor


class MVSADataLoaders:
  def __init__(self):
    pass

    def get_image_dataloader(self, image_paths, labels):
        train_size = int(0.8 * len(image_paths))  # 80% train, 20% test

        train_paths = image_paths[:train_size]
        train_labels = labels[:train_size]

        test_paths = image_paths[train_size:]
        test_labels = labels[train_size:]

        train_dataset = ImageDataset(train_paths, train_labels)
        test_dataset = ImageDataset(test_paths, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        return train_loader, test_loader

    def get_text_dataloader(self, tokenized_data, labels):
        train_size = int(0.8 * len(labels))  # 80% train, 20% test

        train_data = tokenized_data[:train_size]
        train_labels = labels[:train_size]

        test_data = tokenized_data[train_size:]
        test_labels = labels[train_size:]

        train_dataset = TextDataset(train_data, train_labels)
        test_dataset = TextDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        return train_loader, test_loader