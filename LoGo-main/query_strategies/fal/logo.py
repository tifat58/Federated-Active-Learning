import copy
import math
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

import torch
import torch.nn as nn

from ..strategy import Strategy


class LoGo(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)

        # Handle case when there are no labeled indices initially
        if len(label_idxs) == 0:
            print(f"[Client {user_idx}] No labeled data available, performing random selection.")
            if len(unlabel_idxs) == 0:
                print(f"[Client {user_idx}] No unlabeled data to sample from either! Skipping query.")
                return []
            n_sample = min(n_query, len(unlabel_idxs))
            return np.random.choice(unlabel_idxs, n_sample, replace=False).tolist()

        g_net = self.net
        l_net = self.training_local_only(label_idxs)
        
        # Cluster with uncertain samples using local model embeddings
        embedding = self.get_grad_embedding_maxInd(list(unlabel_idxs), net=l_net)

        if embedding.shape[0] == 0:
            print(f"[Client {user_idx}] No embedding to perform KMeans. Skipping query.")
            return []

        print("Macro Step: K-Means EM algorithm with local-only model")
        kmeans = KMeans(n_clusters=min(n_query, len(embedding)))  # prevent error if fewer samples
        kmeans.fit(embedding)
        cluster_pred = kmeans.labels_
        
        cluster_dict = {i: [] for i in range(n_query)}        
        for u_idx, c in zip(unlabel_idxs, cluster_pred):
            cluster_dict[c].append(u_idx)

        print("Micro Step: 1-step EM algorithm with global model")
        query_idx = []
        for c_i in cluster_dict.keys():
            cluster_idxs = np.array(cluster_dict[c_i])
            
            probs = self.predict_prob(cluster_idxs, g_net)
            log_probs = torch.log(probs)
            
            log_probs[log_probs == float('-inf')] = 0
            log_probs[log_probs == float('inf')] = 0
            
            U = (probs * log_probs).sum(1)
            U = U.numpy()
            
            if len(U) == 0:
                continue  # Skip empty clusters
            
            chosen = np.argsort(U)[0]                
            query_idx.append(cluster_idxs[chosen])
                    
        query_idx = list(set(query_idx))

        # Handle fewer clusters than query budget
        if len(query_idx) != n_query:
            print('Cluster centroids number is different from query budget.')
            
            num = math.ceil((n_query - len(query_idx)) / len(np.unique(cluster_pred)))
            idx, skip = 0, []

            query_idx = set(query_idx)
            U_dict = {c_i: None for c_i in cluster_dict.keys()}
            max_retries = 10
            retry_count = 0
            while len(query_idx) < n_query and retry_count < max_retries:
                for c_i in cluster_dict.keys():
                    if c_i in skip: continue

                    cluster_idxs = np.array(cluster_dict[c_i])
                    
                    if len(cluster_idxs) < idx+1:
                        skip.append(c_i)
                    else:
                        if U_dict[c_i] is None:
                            probs = self.predict_prob(cluster_idxs, g_net)
                            log_probs = torch.log(probs)
                            
                            log_probs[log_probs == float('-inf')] = 0
                            log_probs[log_probs == float('inf')] = 0
                            
                            U = (probs*log_probs).sum(1).numpy()
                            U_dict[c_i] = deepcopy(U)
                        else:
                            U = U_dict[c_i]

                        chosen = np.argsort(U)[idx+1:idx+1+num]
                        query_idx.update(cluster_idxs[chosen])
                idx += num
                retry_count += 1
            
            query_idx = list(query_idx)[:n_query]

        print(f"[LoGo] Finished querying client {user_idx} — Total queried: {len(query_idx)}")
        return query_idx

# import copy
# import math
# import numpy as np
# from copy import deepcopy
# from sklearn.cluster import KMeans

# import torch
# import torch.nn as nn

# from ..strategy import Strategy


# class LoGo(Strategy):
#     def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
#         unlabel_idxs = np.array(unlabel_idxs)

#         g_net = self.net
#         l_net = self.training_local_only(label_idxs)
        
#         # cluster with uncertain samples by local net
#         embedding = self.get_grad_embedding_maxInd(list(unlabel_idxs), net=l_net)

#         print("Macro Step: K-Means EM algorithm with local-only model")
#         kmeans = KMeans(n_clusters=n_query)
#         kmeans.fit(embedding)
#         cluster_pred = kmeans.labels_
        
#         cluster_dict = {i: [] for i in range(n_query)}        
#         for u_idx, c in zip(unlabel_idxs, cluster_pred):
#             cluster_dict[c].append(u_idx)

#         print("Micro Step: 1 step of EM algorithm with global model")
#         # query with uncertain samples by global net via predefined cluster
#         query_idx = []
#         for c_i in cluster_dict.keys():
#             cluster_idxs = np.array(cluster_dict[c_i])
            
#             probs = self.predict_prob(cluster_idxs, g_net)
#             log_probs = torch.log(probs)
            
#             # inf to zero
#             log_probs[log_probs == float('-inf')] = 0
#             log_probs[log_probs == float('inf')] = 0
            
#             U = (probs*log_probs).sum(1)
#             U = U.numpy()
            
#             try:
#                 chosen = np.argsort(U)[0]                
#                 query_idx.append(cluster_idxs[chosen])
#             except:
#                 # IndexError: index 0 is out of bounds for axis 0 with size 0 with ConvergenceWarning
#                 continue
                
#         query_idx = list(set(query_idx))

#         # sometimes k-means clustering output smaller amount of centroids due to convergence errors
#         if len(query_idx) != n_query:
#             print('cluster centroids number is different from the number of query budget')
            
#             num = math.ceil((n_query - len(query_idx)) / len(np.unique(cluster_pred)))
#             idx, skip = 0, []

#             query_idx = set(query_idx)
#             U_dict = {c_i: None for c_i in cluster_dict.keys()}
#             while len(query_idx) < n_query:
#                 for c_i in cluster_dict.keys():
#                     if c_i in skip: continue

#                     cluster_idxs = np.array(cluster_dict[c_i])
                    
#                     if len(cluster_idxs) < idx+1:
#                         skip.append(c_i)
#                     else:
#                         if U_dict[c_i] is None:
#                             # store uncertainty
#                             probs = self.predict_prob(cluster_idxs, g_net)
#                             log_probs = torch.log(probs)
                            
#                             log_probs[log_probs == float('-inf')] = 0
#                             log_probs[log_probs == float('inf')] = 0
                            
#                             U = (probs*log_probs).sum(1)
#                             U = U.numpy()
#                             U_dict[c_i] = deepcopy(U)
#                         else:
#                             U = U_dict[c_i]

#                         chosen = np.argsort(U)[idx+1:idx+1+num]
#                         try:
#                             query_idx = query_idx.union(set(cluster_idxs[chosen]))
#                         except TypeError:
#                             query_idx = query_idx.union(set([cluster_idxs[chosen]]))
#                 idx += num
            
#             query_idx = list(query_idx)[:n_query]
#         return query_idx