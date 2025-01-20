import json
import random

import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import MVSADataLoaders
from utils.data_processor import TextDataProcessor, ImageDataProcessor, MultiModalDataProcessor
from utils.metrics import Metrics
from utils.model import MSAModel
from utils.model_config import Config


def dict_to_str(src_dict):
  dst_str = ""
  for key in src_dict.keys():
    dst_str += " %s: %.4f " % (key, src_dict[key])
  return dst_str

class Trainer(object):
  def __init__(self, config: Config):

    self.config = config
    self.loss_fn = nn.CrossEntropyLoss()
    self.metrics = Metrics()
    self.tasks = ['M', 'T', 'I']              # M -> Multimodal task, T -> Text modality, I -> Image modality

  def train_step(self, model: MSAModel, text_dataloader: DataLoader, image_dataloader: DataLoader):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

    total_loss = 0.0
    total_accuracy = 0.0

    for (batch_text, batch_img) in tqdm(zip(text_dataloader, image_dataloader)):
      text_inputs = batch_text["text_inputs"].to(self.config.device)
      text_mask = batch_text["text_masks"].to(self.config.device)

      img_inputs = batch_img["img_inputs"].to(self.config.device)

      targets = batch_text["targets"].to(self.config.device)

      optimizer.zero_grad()

      outputs = model(text_inputs, text_mask, img_inputs)
      # pred_probs = torch.softmax(outputs['M'], dim=1)
      # pred_labels = torch.argmax(pred_probs, dim=1)
      # print(pred_labels, targets)

      loss = 0.0
      for task in self.tasks:
        sub_loss = self.config.loss_weights[task] * self.loss_fn(outputs[task], targets)
        loss += sub_loss

      # loss = self.loss_fn(outputs['M'], targets)

      train_results = self.metrics.evaluate(outputs['M'], targets)
      accuracy = train_results['accuracy']

      total_loss += loss.item() * text_inputs.size(0)
      total_accuracy += accuracy * text_inputs.size(0)

      loss.backward()
      optimizer.step()

    total_loss = total_loss / len(text_dataloader.dataset)
    total_accuracy = total_accuracy / len(text_dataloader.dataset)

    return total_loss, total_accuracy

  def test_step(self, model: MSAModel, text_dataloader: DataLoader, image_dataloader: DataLoader, mode: str):
    model.eval()
    y_pred = {'M': [], 'T': [], 'I': []}
    y_true = {'M': [], 'T': [], 'I': []}
    total_loss = 0
    val_loss = {
      'M': 0,
      'T': 0,
      'I': 0
    }

    with torch.inference_mode():
      for (batch_text, batch_img) in tqdm(zip(text_dataloader, image_dataloader)):
        text_inputs = batch_text["text_inputs"].to(self.config.device)
        text_mask = batch_text["text_masks"].to(self.config.device)

        img_inputs = batch_img["img_inputs"].to(self.config.device)

        outputs = model(text_inputs, text_mask, img_inputs)

        targets = batch_text["targets"].to(self.config.device)

        loss = 0.0
        for task in self.tasks:
          sub_loss = self.config.loss_weights[task] * self.loss_fn(outputs[task], targets)
          loss += sub_loss
          val_loss[task] += sub_loss.item() * text_inputs.size(0)

          y_pred[task].append(outputs[task].cpu())
          y_true[task].append(targets.cpu())

        total_loss += loss.item() * text_inputs.size(0)

      for task in self.tasks:
        val_loss[task] = val_loss[task] / len(text_dataloader.dataset)

      total_loss = total_loss / len(text_dataloader.dataset)

      print(f"{mode} loss: {total_loss:.4f} | Multimodal loss: {val_loss['M']:.4f} | Text loss: {val_loss['T']:.4f} | Image loss: {val_loss['I']:.4f}")

      eval_results = {}
      for task in self.tasks:
        pred, true = torch.cat(y_pred[task]), torch.cat(y_true[task])
        results = self.metrics.evaluate(pred, true)
        print(f"'{task}' results: "  + dict_to_str(results))
        eval_results[task] = results

      eval_results = eval_results[self.tasks[0]]
      eval_results['loss'] = total_loss

    return eval_results

  def train(self):
    random.seed(self.config.random_seed)
    torch.manual_seed(self.config.random_seed)
    torch.cuda.manual_seed(self.config.random_seed)
    np.random.seed(self.config.random_seed)
    torch.backends.cudnn.deterministic = True

    texts, labels = TextDataProcessor(self.config.label_path, self.config.text_path).get_text_with_same_labels()
    image_list, labels = ImageDataProcessor(self.config.label_path, self.config.image_path).get_image_with_same_labels()
    texts, image_list, labels = MultiModalDataProcessor(texts, image_list, labels).shuffle()
    text_train_loader, text_val_loader = MVSADataLoaders().get_text_dataloader(texts, labels)
    image_train_loader, image_val_loader = MVSADataLoaders().get_image_dataloader(image_list, labels)

    model = MSAModel(self.config).to(self.config.device)

    # Freezing the pretrained embedding models
    for param in model.text_sub_model.embedding_model.parameters():
      param.requires_grad = False

    for param in model.img_sub_model.embedding_model.parameters():
      param.requires_grad = False

    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    best_model = None

    history = {
      "train_losses": [],
      "train_accs": [],
      "val_losses": [],
      "val_accs": []
    }

    for epoch in tqdm(range(self.config.epochs)):
      print(f"\n=================== Epoch {epoch + 1} ====================")

      train_loss, train_acc = self.train_step(model, text_train_loader, image_train_loader)
      print(f"Train loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

      eval_results = self.test_step(model, text_val_loader, image_val_loader, mode='Validation')
      val_loss, val_acc = eval_results['loss'], eval_results['accuracy']

      if eval_results["accuracy"] >= highest_eval_acc:
        highest_eval_acc = eval_results["accuracy"]
        torch.save(model.state_dict(), self.config.model_save_path + f'msa_model_{highest_eval_acc}_ckpt.pth')
        best_epoch = epoch
        best_model = model.state_dict()

      history["train_losses"].append(train_loss)
      history["train_accs"].append(train_acc)
      history["val_losses"].append(val_loss)
      history["val_accs"].append(val_acc)

      with open("training_history.json", "w") as f:
        json.dump(history, f)

      if epoch - best_epoch >= self.config.early_stop:
        break


    torch.save(best_model, self.config.model_save_path + f'msa_model_best_ckpt.pth')


  def test(self):
    texts, labels = TextDataProcessor(self.config.label_path, self.config.text_path).get_text_with_same_labels()
    image_list, labels = ImageDataProcessor(self.config.label_path, self.config.image_path).get_image_with_same_labels()
    texts, image_list, labels = MultiModalDataProcessor(texts, image_list, labels).shuffle()
    _, text_test_loader = MVSADataLoaders().get_text_dataloader(texts, labels, batch_size=self.config.batch_size)
    _, image_test_loader = MVSADataLoaders().get_image_dataloader(image_list, batch_size=self.config.batch_size)

    model = MSAModel(self.config).to(self.config.device)
    model.eval()

    with torch.inference_mode():
      model.load_state_dict(torch.load(self.config.model_save_path + f'msa_model_best_ckpt.pth'))
      test_results = self.test_step(model, text_test_loader, image_test_loader, mode='Test')
      print(f"\nTest results: {test_results['accuracy']:.4f}")
