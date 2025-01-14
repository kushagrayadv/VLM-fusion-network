import torch.nn as nn
import torch

class MultimodalTransformerModel(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=4):
        super(MultimodalTransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_features, image_features):
        combined_features = torch.stack([text_features, image_features], dim=1)
        fused_features = self.transformer_encoder(combined_features).mean(dim=1)
        return self.classifier(fused_features)