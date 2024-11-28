import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor


class TextDataset(Dataset):
    def __init__(self, tokenized_data, labels):
       data_dict = tokenized_data.to_dict()  
       self.input_ids = data_dict['input_ids']
       self.token_type_ids = data_dict['token_type_ids']
       self.attention_mask = data_dict['attention_mask']
       self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text_inputs': self.input_ids[idx],
            'text_masks': self.attention_mask[idx],
            "targets": {
                "M": torch.tensor(self.labels[idx], dtype=torch.long),
                "T": torch.tensor(self.labels[idx], dtype=torch.long),
                "I": torch.tensor(self.labels[idx], dtype=torch.long)
            }
        }

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_processor(image)

        return {
            "img_inputs": image,
            "targets": {
                "M": torch.tensor(self.labels[idx], dtype=torch.long),
                "T": torch.tensor(self.labels[idx], dtype=torch.long),
                "I": torch.tensor(self.labels[idx], dtype=torch.long)
            }
        }

