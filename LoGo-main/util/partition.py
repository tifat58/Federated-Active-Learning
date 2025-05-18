import numpy as np
from collections import defaultdict

def partition_data_dirichlet(dataset_train, dataset_test, args):
    num_classes = len(np.unique([label for _, label in dataset_train]))
    num_users = args.num_users
    beta = args.dd_beta

    def partition(dataset):
        labels = np.array([label for _, label in dataset]).flatten()
        dict_users = defaultdict(list)
        idxs = np.arange(len(labels))

        # Shuffle indices initially for randomness
        np.random.shuffle(idxs)

        proportions = np.random.dirichlet(np.repeat(beta, num_users))

        proportions = (proportions * len(labels)).astype(int)

        # Adjust to ensure all indices are used
        proportions[-1] = len(labels) - np.sum(proportions[:-1])

        start_idx = 0
        for user in range(num_users):
            end_idx = start_idx + proportions[user]
            dict_users[user] = idxs[start_idx:end_idx].tolist()
            start_idx = end_idx

        return dict_users

    dict_users_train_total = partition(dataset_train)
    dict_users_test_total = partition(dataset_test)

    return dict_users_train_total, dict_users_test_total