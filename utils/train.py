import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

  def train_step(self, model: MSAModel, dataloader: DataLoader):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

    total_loss = 0.0

    for batch in tqdm(dataloader):
      text_inputs = batch["text_inputs"].to(self.config.device)
      text_mask = batch["text_masks"].to(self.config.device)

      img_inputs = batch["img_inputs"].to(self.config.device)
      img_mask = batch["img_masks"].to(self.config.device)

      optimizer.zero_grad()

      outputs = model(text_inputs, text_mask, img_inputs, img_mask)

      loss = 0.0
      for task in self.tasks:
        targets = batch["targets"].to(self.config.device).view(-1, 1)

        sub_loss = self.config.loss_weights[task] * self.loss_fn(outputs[task], targets[task])
        loss += sub_loss

      total_loss += loss.item() * text_inputs.size(0)

      loss.backward()
      optimizer.step()

    total_loss = round(total_loss / len(dataloader.dataset), 4)

    return total_loss

  def test_step(self, model: MSAModel, dataloader: DataLoader, mode: str):
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
      for batch in tqdm(dataloader):
        text_inputs = batch["text_inputs"].to(self.config.device)
        text_mask = batch["text_masks"].to(self.config.device)

        img_inputs = batch["img_inputs"].to(self.config.device)
        img_mask = batch["img_masks"].to(self.config.device)

        outputs = model(text_inputs, text_mask, img_inputs, img_mask)

        loss = 0.0
        for task in self.tasks:
          targets = batch["targets"][task].to(self.config.device).view(-1, 1)

          sub_loss = self.config.loss_weights[task] * self.loss_fn(outputs[task], targets)
          loss += sub_loss
          val_loss[task] += sub_loss.item() * text_inputs.size(0)

          y_pred[task].append(outputs[task].cpu())
          y_true[task].append(targets.cpu())

        total_loss += loss.item() * text_inputs.size(0)

      for task in self.tasks:
        val_loss[task] = val_loss[task] / len(dataloader.dataset)

      total_loss = total_loss / len(dataloader.dataset)

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