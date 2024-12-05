# from transformers import Data2VecVisionModel
# from utils.base_model import BaseModel
from utils.model_config import Config
from torch import nn
from torchvision import models
import torchvision


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        densenet = models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1024, 512)  # Reduce image focus with DenseNet-121 and project to 512-dim

    def forward(self, images):
        features = self.features(images)
        pooled = self.avgpool(features).view(features.size(0), -1)
        return self.proj(pooled)  # Project to 512-dim