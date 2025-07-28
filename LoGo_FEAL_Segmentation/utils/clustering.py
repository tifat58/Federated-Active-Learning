import numpy as np
from sklearn.cluster import KMeans
import torch
from sklearn.metrics import silhouette_score

def get_client_clusters(client_models, client_loaders, device, max_k=10):
    client_features = []

    for cid in client_models:
        model = client_models[cid]
        loader = client_loaders[cid]

        model.eval()
        with torch.no_grad():
            features = []
            count = 0
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                elif isinstance(batch, dict):
                    images = batch['image']
                else:
                    raise ValueError("Unexpected batch format.")

                images = images.to(device)
                feats = model.backbone(images)['out']
                feats = feats.view(feats.size(0), -1)
                features.append(feats.mean(dim=0))
                count += 1
                if count >= 2:
                    break

            mean_feat = torch.stack(features).mean(dim=0)
            client_features.append(mean_feat.cpu().numpy())

    embeddings = np.stack(client_features)

    if len(embeddings) < 3:
        return [list(range(len(embeddings)))]

    best_k = 2
    best_score = -1
    try:
        for k in range(2, min(max_k, len(embeddings)) + 1):
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
    except ValueError as e:
        print("Silhouette score failed:", e)
        return [list(range(len(embeddings)))]

    if best_score <= 0:
        print("Silhouette score inconclusive. Defaulting to k=2")
        best_k = 2

    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(embeddings)
    clusters = [[] for _ in range(best_k)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(idx)

    return clusters