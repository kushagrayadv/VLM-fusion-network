from transformer import AutoModel
from utils.model_config import Config
from attention_encoder import AttentionEncoder
from torch import Tensor
from base_model import BaseModel
class TextSubModel(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.roberta_model = AutoModel.from_pretrained('roberta-large') 
        self.device = config.device

    def forward(self, input_ids: Tensor, mask: Tensor):
        pretrained_outputs = self.roberta_model(input_ids, attention_mask=mask) # change arguments
        output = AttentionEncoder(self.roberta_model.config)
        return output


