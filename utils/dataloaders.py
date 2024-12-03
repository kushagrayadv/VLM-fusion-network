from torch.utils.data import DataLoader
from utils.text_tokenizer import TextTokenizer
import torch
from sklearn.model_selection import train_test_split
from utils.text_tokenizer import TextTokenizer
from utils.image_tokenizer import ImageTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
class PrepData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_input, attention_mask = TextTokenizer().preprocess_text(row["Content"])
        image_input = ImageTokenizer().preprocess_image(row.get("Image_Name", None))
        label = torch.tensor(int(row["Label"]), dtype=torch.long)
        return {'input_ids': text_input, 'attention_mask': attention_mask, 'pixel_values': image_input}, label

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    pixel_values = torch.stack([item['pixel_values'] for item in inputs])
    input_ids = pad_sequence([item['input_ids'] for item in inputs], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in inputs], batch_first=True)
    labels = torch.stack(labels)
    return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}, labels
class MVSADataLoaders:
    def __init__(self):
        pass
    def get_dataloaders(self,data):
        train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

        train_dataset = PrepData(train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        val_dataset = PrepData(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        test_dataset = PrepData(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        return train_dataloader, val_dataloader, test_dataloader