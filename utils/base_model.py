from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils.attention_encoder import AttentionEncoder
from utils.bert_attention_model import BertConfig
from utils.model_config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.config = config

    self.embedding_model = None                     # It will be Roberta/Data2VecVision for respective modalities
    self.output_layers = nn.Sequential(             # These will be for individual modality
        nn.Dropout(),
        nn.Linear(768, 1)       # Embedding size is 768
    )

    self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)            # class embedding layer

    # Attention layers
    bert_config = BertConfig(num_hidden_layers=config.num_attention_layers)
    self.attention_encoder_layers = nn.ModuleList(
        [AttentionEncoder(bert_config) for _ in range(config.num_attention_layers)]
    )

  def prepend_class(self, inputs: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
      index = torch.LongTensor([0]).to(device=inputs.device)
      cls_emb = self.cls_emb(index)
      cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
      outputs = torch.cat((cls_emb, inputs), dim=1)

      cls_mask = torch.ones(inputs.size(0), 1, inputs.size(2)).to(device=inputs.device)
      masks = torch.cat([cls_mask, masks], dim=1)

      return outputs, masks