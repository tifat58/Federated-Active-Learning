import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(EfficientNetB0, self).__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.embedding_dim = in_features  # Needed for LoGo gradient embedding

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        out = self.backbone.classifier(features)
        return out, features  # (logits, embedding)

    def get_embedding_dim(self):
        return self.embedding_dim