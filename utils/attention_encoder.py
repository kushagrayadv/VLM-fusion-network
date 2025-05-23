from torch import nn, Tensor

from utils.bert_attention_model import BertConfig, BertSelfAttentionLayer, BertIntermediateLayer, BertOutputLayer, \
  BertCrossAttentionLayer


class AttentionEncoder(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super().__init__()

    self.bert_cross_attention_layer = BertCrossAttentionLayer(config)
    self.bert_self_attention_layer = BertSelfAttentionLayer(config)
    self.bert_intermediate_layer = BertIntermediateLayer(config)
    self.bert_output_layer = BertOutputLayer(config)

  def cross_attention(self, input_1: Tensor, input_2: Tensor, mask: Tensor) -> Tensor:
    return self.bert_cross_attention_layer(input_1, input_2, mask)

  def self_attention(self, input_tensor: Tensor, mask: Tensor) -> Tensor:
    return self.bert_self_attention_layer(input_tensor, mask)

  def feed_forward(self, input_tensor: Tensor) -> Tensor:
    intermediate_output = self.bert_intermediate_layer(input_tensor)
    output_tensor = self.bert_output_layer(intermediate_output, input_tensor)

    return output_tensor

  def forward(self, features: Tensor, ctx_features: Tensor, attention_mask: Tensor, ctx_attention_mask: Tensor) -> Tensor:
    output_tensor = self.cross_attention(features, ctx_features, ctx_attention_mask)
    output_tensor = self.self_attention(output_tensor, attention_mask)
    output_tensor = self.feed_forward(output_tensor)

    return output_tensor