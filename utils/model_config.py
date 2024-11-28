from typing import Dict, Literal
import torch
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Config(object):
  def __init__(self,
               loss_weights: Dict = { 'M': 1, 'T': 1, 'I': 1},
               model_save_path: str = 'checkpoint/',
               learning_rate: float = 1e-5,
               epochs: int = 20,
               early_stop: int = 8,
               random_seed: int = 42,
               dropout = 0.3,
               model: Literal["cc", "cme"] = 'cc',
               batch_size: int = 64,
               device = device,
               num_attention_layers: int = 1):

    self.loss_weights = loss_weights
    self.model_save_path = model_save_path
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.early_stop = early_stop
    self.random_seed = random_seed
    self.dropout = dropout
    self.model = model
    self.batch_size = batch_size
    self.device = device
    self.num_attention_layers = num_attention_layers
    self.base_data_path = Path("data/MVSA_Single")
    self.label_path = self.base_data_path / "labelResultAll.csv"
    self.text_path = self.base_data_path / "data/texts"
    self.image_path = self.base_data_path / "data/images"
