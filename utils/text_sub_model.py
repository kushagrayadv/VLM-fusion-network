from transformers import RobertaModel
from utils.model_config import Config
from torch import Tensor
from utils.base_model import BaseModel


class TextSubModel(BaseModel):
  def __init__(self, config: Config):
    super().__init__(config)
    self.device = config.device

  def forward(self, inputs: Tensor, ctx_inputs: Tensor, mask: Tensor):
    output = self.output_layers(inputs)

    attention_encoder_inputs = inputs
    for layer_module in self.attention_encoder_layers:
      attention_encoder_inputs = layer_module(attention_encoder_inputs, ctx_inputs, mask, None)

    return output, attention_encoder_inputs
