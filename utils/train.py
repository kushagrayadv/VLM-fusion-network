import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.metrics import Metrics
from utils.model import MSAModel
from utils.model_config import Config


class Trainer(object):
  def __init__(self, config: Config):

    self.config = config
    self.loss_fn = nn.CrossEntropyLoss()
    self.metrics = Metrics()
    self.tasks = ['M', 'T', 'I']              # M -> Multimodal task, T -> Text modality, I -> Image modality

  def train_step(self, model: MSAModel, dataloader: DataLoader):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

    total_loss = 0.0

    for batch in tqdm(dataloader):
      text_inputs = batch["text_inputs"].to(self.config.device)
      text_mask = batch["text_masks"].to(self.config.device)

      img_inputs = batch["img_inputs"].to(self.config.device)
      img_mask = batch["img_masks"].to(self.config.device)

      targets = batch["targets"].to(self.config.device).view(-1, 1)

      optimizer.zero_grad()

      outputs = model(text_inputs, text_mask, img_inputs, img_mask)

      loss = 0.0
      for task in self.tasks:
        sub_loss = self.config.loss_weights[task] * self.loss_fn(outputs[task], targets[task])
        loss += sub_loss

      total_loss += loss.item() * text_inputs.size(0)

      loss.backward()
      optimizer.step()

    total_loss = round(total_loss / len(dataloader.dataset), 4)

  def test_step(self, model: MSAModel, dataloader: DataLoader):
    model.eval()
    y_pred = {'M': [], 'T': [], 'I': []}
    y_true = {'M': [], 'T': [], 'I': []}
    total_loss = 0
    val_loss = {
      'M': 0,
      'T': 0,
      'I': 0
    }