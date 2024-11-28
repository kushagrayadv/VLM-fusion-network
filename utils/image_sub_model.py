from typing import Tuple

import torch
from torch import Tensor

from transformers import Data2VecVisionModel

from utils.base_model import BaseModel
from utils.model_config import Config


class ImageSubModel(BaseModel):
  def __init__(self, config: Config) -> None:
    super().__init__(config=config)
    self.device = config.device
    self.embedding_model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base")

  def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    embeddings = self.embedding_model(inputs, output_attentions=True)
    hidden_states = embeddings.hidden_state

    features = []
    mask_idx_new = []
    for batch in range(hidden_states.shape[0]):
      layer = 0
      while layer < 12:
        try:
          padding_idx = sum(embeddings.attentions[layer][batch][0][0] != 0)
          mask_idx_new.append(padding_idx)
        except:
          layer = layer + 1

      truncated_feature = torch.mean(hidden_states[batch][:padding_idx], dim=0)
      features.append(truncated_feature)

    features = torch.stack(features, dim=0).to(self.device)

    mask_new = torch.zeros(hidden_states.shape[0], hidden_states.shape[1]).to(self.device)
    for batch in range(mask_new.shape[0]):
      mask_new[batch][:mask_idx_new[batch]] = 1

    output = self.output_layers(features)

    attention_encoder_inputs, attention_mask = self.prepend_class(hidden_states, mask_new)

    for layer_module in self.attention_encoder_layers:
      attention_encoder_inputs = layer_module(attention_encoder_inputs, attention_mask)

    return output, attention_encoder_inputs
