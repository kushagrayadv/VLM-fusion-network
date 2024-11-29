from torch.utils.data import DataLoader
from .text_tokenizer import TextTokenizer
from .customDatasets import ImageDataset, TextDataset

class MVSADataLoaders:
    def __init__(self):
        pass
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

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def get_text_dataloader(self, text, labels):
        train_size = int(0.8 * len(labels))  # 80% train, 20% test

        train_data = text[:train_size]
        train_labels = labels[:train_size]

        test_data = text[train_size:]
        test_labels = labels[train_size:]

        train_dataset = TextTokenizer(train_data, train_labels).get_dataset()
        test_dataset = TextTokenizer(test_data, test_labels).get_dataset()

        train_dataset = TextDataset(train_dataset, train_labels)
        test_dataset = TextDataset(test_dataset, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader