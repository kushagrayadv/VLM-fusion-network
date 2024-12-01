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
    self.embedding_model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base", add_pooling_layer=True)

  def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    embeddings = self.embedding_model(inputs)

    features = embeddings.pooler_output

    output = self.output_layers(features)

    attention_encoder_inputs = features
    # for layer_module in self.attention_encoder_layers:
    #   attention_encoder_inputs = layer_module(attention_encoder_inputs, attention_mask = None)

    return output, attention_encoder_inputs
