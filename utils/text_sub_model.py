from transformers import DebertaV2Model
from utils.model_config import Config
from torch import Tensor
from torch import torch
import torch.nn as nn
# from utils.base_model import BaseModel


class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding_model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")
    self.proj = nn.Linear(1024, 512)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  def forward(self, input_ids, attention_mask):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(outputs[:, 0, :])
