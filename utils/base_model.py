import torch
import torch.nn as nn

from utils.model_config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    def __init__(self, config: Config):
        super(BaseModel, self).__init__()
        self.config = config

        self.embedding_model = None                     # It will be Roberta/Data2VecVision for respective modalities
        self.output_layers = nn.Sequential(             # These will be for individual modality
            nn.Dropout(),
            nn.Linear(768, 1)       # Embedding size is 768
        )

        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)            # class embedding layer
