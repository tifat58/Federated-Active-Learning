from sklearn.metrics import balanced_accuracy_score
import torch
import numpy as np

def dice_score(pred, target, epsilon=1e-6):
    """
    Compute mean Dice score over batch.
    pred: logits or probabilities with shape [B, C, H, W]
    target: one-hot ground truth mask with shape [B, C, H, W]
    """
    pred = torch.argmax(pred, dim=1)
    target = torch.argmax(target, dim=1)

    intersection = (pred * target).float().sum(dim=(1, 2))
    union = pred.float().sum(dim=(1, 2)) + target.float().sum(dim=(1, 2))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def evaluate(model, dataloader, device, task='segmentation'):
    model.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            alpha = model(images)  # shape: [B, C, H, W]

            if task == 'segmentation':
                if masks.dim() == 3:
                    num_classes = alpha.shape[1]
                    masks = torch.nn.functional.one_hot(masks.long(), num_classes=num_classes)
                    masks = masks.permute(0, 3, 1, 2).float()
                
                total_dice += dice_score(alpha, masks)
                count += 1

    return total_dice / count