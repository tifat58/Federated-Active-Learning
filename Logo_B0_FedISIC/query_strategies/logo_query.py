import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm


def compute_gradient_embedding(model, data_loader, device):
    model.eval()
    embeddings = []

    for x, y, idx in data_loader:
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()
        #output = model(x)
        output, _ = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        grad_vector = torch.cat(grads)
        embeddings.append(grad_vector.cpu().numpy())

    return np.stack(embeddings, axis=0)

def compute_entropy(model, data_loader, device):
    model.eval()
    entropy_scores = []

    with torch.no_grad():
        for x, y, idx in data_loader:
            x = x.to(device)
            # outputs = model(x)
            # probs = F.softmax(outputs, dim=1)
            outputs, _ = model(x)
            probs = F.softmax(outputs, dim=1)   
            entropy = -(probs * probs.log()).sum(dim=1)
            entropy_scores.extend(entropy.cpu().numpy())

    return np.array(entropy_scores)

def logo_query_samples(dataset_query, dict_users_unlabeled, model_global, model_local, args):
    budget = args.budget_per_round
    device = args.device
    query_dict = {}

    for client_id in range(args.num_users):
        unlabeled_idxs = dict_users_unlabeled[client_id]
        if len(unlabeled_idxs) == 0:
            query_dict[client_id] = []
            continue

        subset = Subset(dataset_query, unlabeled_idxs)
        loader = DataLoader(subset, batch_size=1, shuffle=False)

        # Step 1: Gradient embedding (local model)
        grads = compute_gradient_embedding(model_local, loader, device)

        # Step 2: K-means clustering
        n_clusters = min(budget, len(unlabeled_idxs))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(grads)
        cluster_indices = [[] for _ in range(n_clusters)]

        for i, label in enumerate(kmeans.labels_):
            cluster_indices[label].append(i)

        # Step 3: Entropy (global model)
        entropy_scores = compute_entropy(model_global, loader, device)

        # Step 4: Select top-entropy from each cluster
        selected_indices = []
        for group in cluster_indices:
            if not group:
                continue
            group_scores = entropy_scores[group]
            top_idx = group[np.argmax(group_scores)]
            selected_indices.append(unlabeled_idxs[top_idx])

        query_dict[client_id] = selected_indices

    return query_dict