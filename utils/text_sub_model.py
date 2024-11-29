from transformers import RobertaModel
from utils.model_config import Config
from torch import Tensor
from utils.base_model import BaseModel


class TextSubModel(BaseModel):
  def __init__(self, config: Config):
    super().__init__(config)
    self.embedding_model = RobertaModel.from_pretrained('roberta-base')
    self.device = config.device

  def forward(self, inputs: Tensor, mask: Tensor):
    pretrained_outputs = self.embedding_model(inputs)  # change arguments
    features = pretrained_outputs.pooler_output

    output = self.output_layers(features)

    attention_encoder_inputs = features
    for layer_module in self.attention_encoder_layers:
      attention_encoder_inputs = layer_module(attention_encoder_inputs, attention_mask=None)

    return output, attention_encoder_inputs
