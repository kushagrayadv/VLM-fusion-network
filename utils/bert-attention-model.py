import math

import torch
from torch import nn, Tensor

"""
---------------------------------------------------------------------------------------
      Below modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""

class BertConfig(object):
  """Configuration class to store the configuration of a `BertModel`.
  """

  def __init__(self,
               hidden_size=768,
               num_hidden_layers=3,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="relu",
               hidden_dropout_prob=0.3,
               attention_probs_dropout_prob=0.3,
               max_position_embeddings=512):
    """Constructs BertConfig.
    Args:
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        add_abs_pos_emb: absolute positional embeddings
        add_pos_enc: positional encoding
    """

    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings


BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
  """Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
  """Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
  """

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return gelu(x)


def swish(x):
  return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": nn.functional.relu, "swish": swish}


class BertAttention(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super().__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x: Tensor) -> Tensor:
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)  # (b, s,h, d) -> (b, h, s, d)

  def forward(self, hidden_states: Tensor, context: Tensor, attention_mask: Tensor = None) -> Tensor:
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(context)
    mixed_value_layer = self.value(context)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # shape is (b, h, s_q, s_k)

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Apply the attention mask
    if attention_mask is not None:
      attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      attention_mask = attention_mask.expand((-1, attention_scores.size(1), attention_scores.size(2), -1))
      attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
      attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
      attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)  # shape is (b, h, s_q, d)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # shape is (b, s_q, h, d)
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


class BertAttentionOutput(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super(BertAttentionOutput, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class BertSelfAttentionLayer(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super(BertSelfAttentionLayer, self).__init__()
    self.self = BertAttention(config)
    self.output = BertAttentionOutput(config)

  def forward(self, input_tensor: Tensor, attention_mask: Tensor) -> Tensor:
    # Self attention attends to itself, thus keys and queries are the same (input_tensor).
    self_output = self.self(input_tensor, input_tensor, attention_mask)
    attention_output = self.output(self_output, input_tensor)
    return attention_output


class BertIntermediateLayer(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super(BertIntermediateLayer, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    if config.hidden_act in ACT2FN:
      self.intermediate_act_fn = ACT2FN[config.hidden_act]
    else:
      self.intermediate_act_fn = nn.functional.relu

  def forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


class BertOutputLayer(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super(BertOutputLayer, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


class BertLayer(nn.Module):
  def __init__(self, config: BertConfig) -> None:
    super(BertLayer, self).__init__()
    self.attention = BertSelfAttentionLayer(config)
    self.intermediate = BertIntermediateLayer(config)
    self.output = BertOutputLayer(config)

  def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    attention_output = self.attention(hidden_states, attention_mask)
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output