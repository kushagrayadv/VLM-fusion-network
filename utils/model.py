from typing import Dict

import torch
from torch import nn, Tensor

from utils.model_config import Config
from utils.image_sub_model import ImageSubModel
from utils.text_sub_model import TextSubModel


class MSAModel(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.img_sub_model = ImageSubModel(config)
    self.text_sub_model = TextSubModel(config)   

    self.fused_output_layers = nn.Sequential(
      nn.Dropout(config.dropout),
      nn.Linear(in_features=768 * 2, out_features=768),
      nn.ReLU(),
      nn.Linear(in_features=768, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=1),
    )

  def forward(self, text_inputs: Tensor,
              text_mask: Tensor,
              img_inputs: Tensor) -> Dict[str, Tensor]:

    text_output, attention_enc_text_output = self.text_sub_model(text_inputs, text_mask)
    img_output, attention_enc_img_output = self.img_sub_model(img_inputs)

    concatenated_hidden_states = torch.cat((attention_enc_text_output[:, 0, :], attention_enc_img_output[:, 0, :]), dim=1)

    fused_output = self.fused_output_layers(concatenated_hidden_states)

    return {
      'T': text_output,
      'I': img_output,
      'M': fused_output,
    }