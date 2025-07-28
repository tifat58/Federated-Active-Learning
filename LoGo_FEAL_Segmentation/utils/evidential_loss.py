import torch
import torch.nn.functional as F
import torch.nn as nn

def edl_mse_loss(output, target, epoch_num, num_classes, kl_weight=0.01):
    """
    Evidential Deep Learning (EDL) Mean Squared Error Loss.
    output: alpha (Dirichlet parameters) from model
    target: one-hot encoded labels
    """
    S = torch.sum(output, dim=1, keepdim=True)
    E = output - 1

    # Prediction loss
    pred_loss = torch.sum((target - (output / S)) ** 2, dim=1, keepdim=True)

    # Variance loss
    var_loss = torch.sum(output * (S - output) / (S * S * (S + 1)), dim=1, keepdim=True)

    # Total loss
    loss = pred_loss + var_loss

    # KL divergence
    annealing_coef = min(1.0, epoch_num / 10)
    alpha_0 = E + 1
    kl_div = kl_dirichlet(alpha_0, num_classes)

    return loss.mean() + kl_weight * annealing_coef * kl_div.mean()

def kl_dirichlet(alpha, num_classes, prior_strength=1.0):
    """
    KL divergence between Dir(alpha) and Dir(1,...,1)
    """
    beta = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB_alpha = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(S_alpha).squeeze()
    lnB_beta = torch.sum(torch.lgamma(beta), dim=1) - torch.lgamma(S_beta).squeeze()

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1) + lnB_alpha - lnB_beta
    return kl

def bayes_dice_loss(alpha, y_true):
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    num = 2 * (probs * y_true).sum(dim=(2, 3))
    denom = (y_true**2).sum(dim=(2, 3)) + (probs**2).sum(dim=(2, 3)) + ((probs * (1 - probs)) / (S + 1)).sum(dim=(2, 3))
    return 1 - (num / denom).mean()

def evidential_segmentation_loss(alpha, y_true, lam=0.001):
    loss_task = bayes_dice_loss(alpha, y_true)
    alpha_tilde = y_true + (1 - y_true) * alpha
    S = alpha.sum(dim=1, keepdim=True)
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    reg = torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
    reg = reg.mean()
    return loss_task + lam * reg