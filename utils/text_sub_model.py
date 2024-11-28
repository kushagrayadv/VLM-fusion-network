from transformers import AutoModel
from utils.model_config import Config
from torch import Tensor
from base_model import BaseModel


class TextSubModel(BaseModel):
  def __init__(self, config: Config):
    super().__init__(config)
    self.roberta_model = AutoModel.from_pretrained('roberta-large')
    self.device = config.device

  def forward(self, inputs: Tensor, mask: Tensor):
    pretrained_outputs = self.roberta_model(inputs, attention_mask=mask)  # change arguments
    hidden_states = pretrained_outputs.last_hidden_state

    features = pretrained_outputs["pooler_output"]

    output = self.output_layers(features)

    attention_encoder_inputs, attention_mask = self.prepend_class(hidden_states, mask)

    for layer_module in self.attention_encoder_layers:
      attention_encoder_inputs = layer_module(attention_encoder_inputs, attention_mask)

    return output, attention_encoder_inputs
