from typing import Dict

import torch
from torch import nn, Tensor

from transformers import Data2VecVisionModel, RobertaModel

from utils.model_config import Config
from utils.image_sub_model import ImageSubModel
from utils.text_sub_model import TextSubModel


class MSAModel(nn.Module):
  def __init__(self, config: Config) -> None:
    super().__init__()
    self.img_sub_model = ImageSubModel(config)
    self.text_sub_model = TextSubModel(config)

    self.img_embedding_model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base", add_pooling_layer=True)
    self.text_embedding_model = RobertaModel.from_pretrained('roberta-base')

    self.fused_output_layers = nn.Sequential(
      nn.Dropout(config.dropout),
      nn.Linear(in_features=768 * 2, out_features=768),
      nn.ReLU(),
      nn.Linear(in_features=768, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=3),
    )

    # self.fusion_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=config.num_attention_layers)

  def forward(self, text_inputs: Tensor,
              text_mask: Tensor,
              img_inputs: Tensor) -> Dict[str, Tensor]:

    img_embeddings = self.img_embedding_model(img_inputs)
    img_features = img_embeddings.pooler_output

    text_embeddings = self.text_embedding_model(text_inputs, attention_mask=text_mask)
    text_features = text_embeddings.pooler_output

    img_output, attention_enc_img_output = self.img_sub_model(img_features, text_features, text_mask)
    text_output, attention_enc_text_output = self.text_sub_model(text_features, img_features, text_mask)

    fused_features = torch.stack((attention_enc_text_output, attention_enc_img_output), dim=1)

    # fused_features  = self.fusion_model(fused_features).mean(dim=1)

    fused_output = self.fused_output_layers(fused_features)

    return {
      'T': text_output,
      'I': img_output,
      'M': fused_output,
    }