# global_aggregator.py

import copy
import torch


def aggregate_models(models, uncertainties=None, beta=1.0):
    """
    Federated aggregation of model weights with optional uncertainty-aware weighting.

    Args:
        models (list of OrderedDict): Local models' state_dicts.
        uncertainties (list of float): Average uncertainties from clients.
        beta (float): Weighting coefficient for uncertainty-based aggregation.

    Returns:
        Aggregated global model weights (OrderedDict).
    """
    global_model = copy.deepcopy(models[0])

    if uncertainties is None:
        # Simple FedAvg
        for key in global_model:
            global_model[key] = torch.stack([model[key] for model in models], dim=0).mean(dim=0)
    else:
        weights = [1 / (1 + beta * u) for u in uncertainties]
        norm_factor = sum(weights)
        for key in global_model:
            weighted_sum = sum(w * model[key] for w, model in zip(weights, models))
            global_model[key] = weighted_sum / norm_factor

    return global_model