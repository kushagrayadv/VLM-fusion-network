from tqdm.auto import tqdm
import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import json
from utils.text_sub_model import TextEncoder
from utils.image_sub_model import ImageEncoder
from utils.multi_modal import MultimodalTransformerModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.model_config import Config
from utils.data_processor import TextDataProcessor, ImageDataProcessor, MultiModalDataProcessor
from utils.dataloaders import MVSADataLoaders

# Set device
class Trainer:
  def __init__(self,config):
    self.config = config
  def train(self):
    label_path = self.config.label_path
    text_path = self.config.text_path
    image_path = self.config.image_path

    texts, labels = TextDataProcessor(label_path, text_path).get_text_with_same_labels()
    image_list, labels = ImageDataProcessor(label_path, image_path).get_image_with_same_labels()
    data = MultiModalDataProcessor(texts, image_list, labels).shuffle()
    train_loader, val_loader, test_loader = MVSADataLoaders().get_dataloaders(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Initialize Models, Optimizer, and Loss
    text_encoder = TextEncoder().to(device)
    image_encoder = ImageEncoder().to(device)
    model = MultimodalTransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    class_weights = torch.tensor([1.0,1.75,0.5], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    for param in image_encoder.features.parameters():
      param.requires_grad = True

    for param in text_encoder.embedding_model.parameters():
      param.requires_grad = True
    # Directories to save the best models
    highest_acc_dir = "/Users/uttamsingh/Documents/Graduate/Fall2024/highest_acc"
    highest_f1_dir = "/Users/uttamsingh/Documents/Graduate/Fall2024/highest_f1"
    os.makedirs(highest_acc_dir, exist_ok=True)
    os.makedirs(highest_f1_dir, exist_ok=True)

    # model.load_state_dict(torch.load(os.path.join(highest_acc_dir, "best_accuracy_model.pth")))

    # Initialize tracking variables for highest metrics
    best_accuracy = 0.0

    # Training loop with model-saving logic
    history = {'epoch': [],
              'train_loss': [],
              'train_accuracy': [],
              'train_f1_score': [],
              'val_loss': [],
              'val_accuracy': [],
              'val_f1_score': []
              }

    for epoch in range(30):  # Adjust epochs as needed
        print(f"\n=================== Epoch {epoch + 1} ====================")
        model.train()
        total_loss, correct_predictions, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

        # Train step
        for batch in tqdm(train_loader):
            inputs, labels = batch
            input_ids, attention_mask, pixel_values = [inputs[key].to(device) for key in ['input_ids', 'attention_mask', 'pixel_values']]
            labels = labels.to(device)

            text_features = text_encoder.forward(input_ids, attention_mask)
            image_features = image_encoder.forward(pixel_values)

            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        train_f1 = f1_score(all_labels, all_preds, average="weighted")

        scheduler.step(train_f1)

        model.eval()
        text_encoder.eval()
        image_encoder.eval()
        total_loss, correct_predictions, total_samples = 0, 0, 0

        with torch.inference_mode():
          for batch in tqdm(val_loader):
            inputs, labels = batch
            input_ids, attention_mask, pixel_values = [inputs[key].to(device) for key in ['input_ids', 'attention_mask', 'pixel_values']]
            labels = labels.to(device)

            text_features = text_encoder(input_ids, attention_mask)
            image_features = image_encoder(pixel_values)

            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        val_loss = total_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples
        val_f1 = f1_score(all_labels, all_preds, average="weighted")

        # Save to history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_f1_score'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_score'].append(val_f1)

        with open("training_history.json", "w") as f:
          json.dump(history, f)

        # Check for new highest accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            shutil.rmtree(highest_acc_dir)
            os.makedirs(highest_acc_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(highest_acc_dir, "best_accuracy_model.pth"))
            print(f"New best accuracy model saved with accuracy: {best_accuracy:.4f}")

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Train F1 Score: {train_f1:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | Validation F1 Score: {val_f1:.4f}")

    return model
  def test(self):
    pass