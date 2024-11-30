from typing import Tuple

import torch
from torch import Tensor


from utils.base_model import BaseModel
from utils.model_config import Config


class ImageSubModel(BaseModel):
  def __init__(self, config: Config) -> None:
    super().__init__(config=config)
    self.device = config.device

  def forward(self, inputs: Tensor, ctx_inputs: Tensor, ctx_attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
    output = self.output_layers(inputs)

    attention_encoder_inputs = inputs
    for layer_module in self.attention_encoder_layers:
      attention_encoder_inputs = layer_module(attention_encoder_inputs, ctx_inputs, None, ctx_attention_mask)

    return output, attention_encoder_inputs
