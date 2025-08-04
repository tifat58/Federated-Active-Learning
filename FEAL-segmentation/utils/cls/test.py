import logging
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def dice_score(pred, target, smooth=1e-5):
    """
    Compute Dice score between predicted mask and ground truth.
    Assumes pred and target are tensors of shape [B, H, W]
    """
    pred = pred.float()
    target = target.squeeze(1).float() if target.dim() == 4 else target.float()

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.mean()  # average over batch


def test(dataset, model, dataloader, client_idx):
    model.eval()

    pred_list = []
    label_list = []
    dice_scores = []

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image, label = data['image'], data['label']
            image = image.cuda()
            label = label.cuda()

            logit = model(image)[0]  # output: [B, C, H, W]

            if dataset == 'FedISIC':
                pred = torch.argmax(logit, dim=1)  # [B, H, W]
                pred_list.append(pred.view(-1))
                label_list.append(label.view(-1))

            elif dataset == 'FedPolyp':
                pred = torch.argmax(logit, dim=1)  # [B, H, W]
                dice_scores.append(dice_score(pred, label))

    if dataset == 'FedISIC':
        preds = torch.cat(pred_list)
        labels = torch.cat(label_list)
        return balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    
    elif dataset == 'FedPolyp':
        return torch.tensor(dice_scores).mean().item() if dice_scores else 0.0


# import logging
# import torch
# import numpy as np
# from sklearn.metrics import balanced_accuracy_score, accuracy_score

# def test(dataset, model, dataloader, client_idx):
#     model.eval()

#     pred_list = torch.tensor([]).cuda()
#     label_list = torch.tensor([]).cuda()

#     with torch.no_grad():
#         for _, (_, data) in enumerate(dataloader):
#             image, label = data['image'], data['label']

#             image = image.cuda()
#             label = label.cuda()

#             logit = model(image)[0]    

#             pred_list = torch.cat((pred_list, torch.argmax(logit, dim=1)))
#             label_list = torch.cat((label_list, label))

#     if dataset == 'FedISIC':
#         return balanced_accuracy_score(label_list.cpu().numpy(), pred_list.cpu().numpy())
