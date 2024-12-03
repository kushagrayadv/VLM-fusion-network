import pandas as pd
import numpy as np
import torch
import os
class TextDataProcessor:

  def __init__(self, label_path, text_store_path):
    self.ids = None
    self.labels = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.labels_path = label_path
    self.text_store_path = text_store_path

  def read_text_dataset(self):
    texts = []
    for text_id in self.ids.values:
      filepath = os.path.join(self.text_store_path, f"{text_id}.txt")
      with open(filepath, 'r', errors="replace") as file:
        content = file.read()
        texts.append(content)
    return texts

  def get_text_with_same_labels(self):
    df_labels = pd.read_csv(self.labels_path)
    df_labels_with_same_annotation = df_labels[(df_labels['text'] == df_labels['image'])]
    self.labels = df_labels_with_same_annotation['text']
    self.ids = df_labels_with_same_annotation['ID']
    texts = self.read_text_dataset()
    return texts, self.labels


class ImageDataProcessor:
  def __init__(self, label_path, image_store_path):
    self.ids = None
    self.labels = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.labels_path = label_path
    self.image_store_path = image_store_path

  def load_image_dataset(self):
    images = [ f"{img_id}.jpg" for img_id in self.ids.values]
    return images

  def get_image_with_same_labels(self):
    df_labels = pd.read_csv(self.labels_path)
    df_labels_with_same_annotation = df_labels[(df_labels['text'] == df_labels['image'])]
    self.labels = df_labels_with_same_annotation['image']
    self.ids = df_labels_with_same_annotation['ID']
    images = self.load_image_dataset()
    return images, self.labels


class MultiModalDataProcessor:
  def __init__(self, texts, imgs, labels):
    self.texts = texts
    self.imgs = imgs
    self.labels = labels

  def shuffle(self):
    texts_array = np.array(self.texts)
    img_array = np.array(self.imgs)
    labels = self.labels.map({"positive": 2, "negative": 0, "neutral": 1}).to_numpy()

    concatenated_array = np.column_stack((labels, texts_array, img_array))
    # np.random.shuffle(concatenated_array)

    return pd.DataFrame(concatenated_array, columns=['Label', 'Content', 'Image_Name'])