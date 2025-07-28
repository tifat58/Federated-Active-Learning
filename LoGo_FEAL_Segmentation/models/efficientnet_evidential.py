import torch
import torch.nn as nn
import timm

class EvidentialEfficientNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(EvidentialEfficientNet, self).__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0, features_only=False)

        # Get feature dimension from backbone
        self.feat_dim = self.backbone.num_features

        # Evidential head for Dirichlet parameters
        self.evidential_head = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
            nn.Softplus()
        )

    def forward(self, x):
        features = self.backbone(x)
        alpha = self.evidential_head(features) + 1  # ensure alpha > 1
        return alpha

    def extract_features(self, x):
        return self.backbone(x)

def get_model(config):
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    return EvidentialEfficientNet(num_classes=num_classes, pretrained=pretrained)