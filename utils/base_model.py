# from abc import abstractmethod, ABC
# from typing import Tuple

# import torch
# import torch.nn as nn
# from torch import Tensor

# from utils.attention_encoder import AttentionEncoder
# from utils.bert_attention_model import BertConfig
# from utils.model_config import Config

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class BaseModel(nn.Module, ABC):
#   def __init__(self, config: Config) -> None:
#     super().__init__()
#     self.config = config

#     self.embedding_model = None                     # It will be Roberta/Data2VecVision for respective modalities
#     self.output_layers = nn.Sequential(             # These will be for individual modality
#         nn.Dropout(),
#         nn.Linear(768, 3)       # Embedding size is 768
#     )

#     # Attention layers
#     bert_config = BertConfig(num_hidden_layers=config.num_attention_layers)
#     self.attention_encoder_layers = nn.ModuleList(
#         [AttentionEncoder(bert_config) for _ in range(config.num_attention_layers)]
#     )


#   @abstractmethod
#   def forward(self, inputs: Tensor, *args, **kwargs) -> Tensor:
#     """ This method must be implemented by subclasses. """
#     pass