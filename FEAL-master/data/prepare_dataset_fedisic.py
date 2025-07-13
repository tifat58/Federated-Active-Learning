
import os
import pandas as pd
from collections import defaultdict

def load_fedisic_splits(csv_split_dir):
    train_df = pd.read_csv(os.path.join(csv_split_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(csv_split_dir, 'test.csv'))

    dict_clients_train = defaultdict(list)
    dict_clients_test = defaultdict(list)

    for idx, row in train_df.iterrows():
        dict_clients_train[row['client']].append(idx)

    for idx, row in test_df.iterrows():
        dict_clients_test[row['client']].append(idx)

    return train_df, test_df, dict_clients_train, dict_clients_test
