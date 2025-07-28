import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50

class EvidentialUNet(nn.Module):
    def __init__(self, num_classes):
        super(EvidentialUNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = fcn_resnet50(pretrained_backbone=True, num_classes=num_classes)

    def forward(self, x):
        """
        Returns:
            alpha: Dirichlet parameters (B, C, H, W)
        """
        output = self.backbone(x)['out']  # logits, shape: [B, C, H, W]
        evidence = F.relu(output)         # non-negative evidence
        alpha = evidence + 1              # Dirichlet alpha
        return alpha