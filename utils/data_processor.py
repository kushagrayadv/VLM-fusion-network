import os
import torch
import pandas as pd

class TextDataProcessor:
    
    def __init__(self,label_path,text_store_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_path = label_path
        self.text_store_path = text_store_path
    

    def read_text_dataset(self):
        texts = []
        for filename in os.listdir(self.text_store_path):
            extenstion = filename 
            if( filename.strip('.txt') in list(map(str, self.ids.values))):
                filepath = os.path.join(self.text_store_path,extenstion)
                with open(filepath, 'r',errors="replace") as file:
                    content = file.read()
                    texts.append(content) # Process each line as a text sample
        return texts

    def get_text_with_same_labels(self): 
        df_labels = pd.read_csv(self.labels_path)
        df_labels_with_same_annotation = df_labels[(df_labels['text']== df_labels['image'])]
        self.labels = df_labels_with_same_annotation
        self.ids = df_labels_with_same_annotation['ID']
        texts = self.read_text_dataset(self.text_store_path)
        return texts,self.labels
class ImageDataProcessor:   
    def __init__(self,label_path,image_store_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_path = label_path
        self.image_store_path = image_store_path
    def load_image_dataset(self):
        images = []
        for filename in os.listdir(self.image_store_path):
            extenstion = filename 
            if( filename.strip('.jpg') in list(map(str, self.ids.values))):
                filepath = os.path.join(self.image_store_path,extenstion)
                images.append(filepath) # Process each line as a text sample
        return images
    def get_image_with_same_labels(self): 
        df_labels = pd.read_csv(self.labels_path)
        df_labels_with_same_annotation = df_labels[(df_labels['text']== df_labels['image'])]
        self.labels = df_labels_with_same_annotation
        self.ids = df_labels_with_same_annotation['ID']
        images= self.load_image_dataset(self.image_store_path)
        return images,self.labels
