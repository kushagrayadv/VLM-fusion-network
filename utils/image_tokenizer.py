from transformers import DebertaV2Tokenizer
from torchvision import transforms
import pandas as pd
import torch
import os
from PIL import Image


class ImageTokenizer:
    def __init__(self):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    def preprocess_image(self,image_name):
        if pd.isna(image_name) or not isinstance(image_name, str):
            return torch.zeros(3, 224, 224)
        image_path = os.path.join("/content/data/MVSA_Single/data/images/", image_name)  # Update path accordingly
        try:
            image = Image.open(image_path).convert("RGB")
            return self.image_transform(image)
        except FileNotFoundError:
            return torch.zeros(3, 224, 224)
