import torch
import numpy as np
from collections import defaultdict
from torch.nn.functional import normalize
from sklearn.metrics.pairwise import cosine_distances

def compute_uncertainty(alpha):
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True)  # shape: [B, 1, H, W]

    epistemic = alpha.shape[1] / (alpha_0 + 1e-8)  # shape: [B, 1, H, W]
    aleatoric = torch.sum(alpha * (alpha_0 - alpha), dim=1, keepdim=True) / ((alpha_0 ** 2) * (alpha_0 + 1))  # [B, 1, H, W]

    total_uncertainty = epistemic + aleatoric  # [B, 1, H, W]

    #Average over spatial dimensions to get per-image uncertainty
    total_uncertainty = total_uncertainty.mean(dim=(2, 3))  # shape: [B, 1]

    return total_uncertainty.view(-1)  # shape: [B]

def select_uncertain_samples(model, unlabeled_loader, device, query_budget, apply_cosine_filter=False):
    model.eval()
    uncertainties = []
    all_indices = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            inputs, _, indices = batch
            inputs = inputs.to(device)
            alpha = model(inputs)
            uncertainty = compute_uncertainty(alpha)

            # Dirichlet mean = expected class probabilities
            probs = alpha / alpha.sum(dim=1, keepdim=True)
            preds = torch.argmax(probs, dim=1)

            uncertainties.append(uncertainty)
            all_indices.extend(indices)
            all_preds.extend(preds.cpu().numpy())
            
            # For cosine diversity: extract features before final layer
            features = model.backbone(inputs)
            if isinstance(features, dict) and 'out' in features:
                features = features['out']
            features = normalize(features.view(features.size(0), -1), dim=1)  # L2 normalize
            all_features.append(features.cpu())

    if len(uncertainties) == 0 or len(all_indices) == 0:
        return []

    uncertainties = torch.cat(uncertainties)
    all_features = torch.cat(all_features)
    all_preds = np.array(all_preds)

    # Step 1: Get top-2k uncertain samples
    candidate_top_k = min(query_budget * 2, len(all_indices))
    if len(all_indices) != len(uncertainties):
        raise ValueError(f"[ERROR] Mismatch: len(all_indices)={len(all_indices)} vs len(uncertainties)={len(uncertainties)}")

    topk_indices = torch.argsort(uncertainties, descending=True)[:candidate_top_k]

    top_uncertainties = uncertainties[topk_indices]
    top_indices = [all_indices[i] for i in topk_indices.tolist()]
    top_preds = all_preds[topk_indices.cpu().numpy()]
    top_features = all_features[topk_indices.cpu()]

    # Step 2: Group by predicted class
    class_to_samples = defaultdict(list)
    for i, idx in enumerate(top_indices):
        value = top_preds[i]
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else int(value.flatten()[0])
        elif hasattr(value, 'item'):
            value = value.item()
        class_label = int(value)  # Ensure it's a native Python int
        class_to_samples[class_label].append((top_uncertainties[i].item(), idx, top_features[i]))

    # Step 3: Sample proportionally from each class
    selected_indices = []
    per_class_quota = max(1, query_budget // max(1, len(class_to_samples)))

    for samples in class_to_samples.values():
        samples.sort(reverse=True)  # by uncertainty
        for _, idx, feat in samples[:per_class_quota]:
            selected_indices.append(idx)
            if len(selected_indices) >= query_budget:
                break
        if len(selected_indices) >= query_budget:
            break

    # Step 4: Apply cosine diversity filtering
    if apply_cosine_filter and len(selected_indices) > 1:
        filtered = []
        selected_feats = []
        for idx in selected_indices:
            idx_pos = top_indices.index(idx)
            feat = top_features[idx_pos].unsqueeze(0)
            if not selected_feats:
                filtered.append(idx)
                selected_feats.append(feat)
            else:
                dists = cosine_distances(torch.cat(selected_feats).numpy(), feat.numpy())
                if np.min(dists) > 0.1:  # 0.1 = diversity threshold
                    filtered.append(idx)
                    selected_feats.append(feat)
        selected_indices = filtered[:query_budget]
        selected_feats = torch.stack([top_features[top_indices.index(idx)] for idx in selected_indices])
        diverse_ids = greedy_cosine_diverse_selection(selected_feats, query_budget)
        selected_indices = [selected_indices[i] for i in diverse_ids]

    return selected_indices

def extract_features(model, data_loader, device, selected_indices=None):
    features = []
    index_map = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            inputs, _, indices = batch
            inputs = inputs.to(device)
            if selected_indices is not None:
                mask = [i.item() in selected_indices for i in indices]
                if not any(mask): continue
                inputs = inputs[mask]
                indices = [i for i, m in zip(indices, mask) if m]
            feats = model.extract_features(inputs)
            features.append(feats.cpu())
            index_map.extend(indices)

    if features:
        return torch.cat(features), index_map
    else:
        return torch.empty(0), []

def greedy_cosine_diverse_selection(features, k):
    selected = []
    remaining = list(range(features.size(0)))

    selected.append(remaining.pop(0))

    while len(selected) < k and remaining:
        max_dist = -float('inf')
        next_idx = -1
        for idx in remaining:
            dists = torch.nn.functional.cosine_similarity(
                features[idx].unsqueeze(0), features[selected], dim=1)
            min_sim = dists.min()
            if min_sim > max_dist:
                max_dist = min_sim
                next_idx = idx
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected