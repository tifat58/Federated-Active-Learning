import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# def load_csv_splits(csv_dir):
#     """
#     Load train, val, and test CSVs into dictionary of DataFrames.
#     Each CSV should have: image_id,label
#     """
#     splits = {}
#     for split in ['train', 'val', 'test']:
#         csv_path = os.path.join(csv_dir, f'{split}.csv')
#         if not os.path.exists(csv_path):
#             raise FileNotFoundError(f"Missing CSV file: {csv_path}")
#         df = pd.read_csv(csv_path)
#         splits[split] = df.values.tolist()  # list of [image_id, label]
#     return splits


class FedISICDataset(Dataset):
    def __init__(self, data_list, data_root, transform=None):
        self.data_list = data_list
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_id, label = self.data_list[idx]
        img_path = os.path.join(self.data_root, f"{image_id}.npy")

        image = np.load(img_path)
        image = image.astype(np.uint8)

        # Convert HWC → CHW
        if image.shape[-1] == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected image shape: {image.shape} at {img_path}")

        # Normalize and augment
        if self.transform:
            image = self.transform(image)

        return image, int(label), idx  # idx is useful for querying


def get_transforms():
    """
    Return training and test transforms.
    EfficientNet-B0 expects 224x224 and ImageNet normalization.
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def get_fedisic_loaders(config, csv_splits):
    train_data = csv_splits['train']  # list of [image_id, label, client_id]
    test_data = csv_splits['test']    # same format

    num_clients = config['num_clients']
    data_root = config.get('dataset', {}).get('data_root')
    if data_root is None:
        raise ValueError("Missing 'data_root' in config['dataset']")
    
    train_transform, test_transform = get_transforms()
    
    # Build per-client train datasets and dict_users_train_total
    dict_users_train_total = {}
    for client_id in range(num_clients):
        client_samples = [[img_id, label] for img_id, label, cid in train_data if cid == client_id]
        dataset = FedISICDataset(client_samples, data_root, transform=train_transform)
        indices = list(range(len(dataset)))
        dict_users_train_total[client_id] = indices

    # Build one combined training dataset (for index mapping)
    all_train_samples = [[img_id, label] for img_id, label, _ in train_data]
    dataset_train = FedISICDataset(all_train_samples, data_root, transform=train_transform)

    # Build test set
    test_samples = [[img_id, label] for img_id, label, _ in test_data]
    test_dataset = FedISICDataset(test_samples, data_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return dataset_train, dict_users_train_total, test_loader

def load_csv_splits_feal_style(csv_path):
    """
    Load FEAL-style split from TSV, including client_id from 'center' column.
    Returns a dictionary: {'train': [...], 'test': [...]}
    Each element is [image_id, label, client_id]
    """
    df = pd.read_csv(csv_path, sep='\t')
    df.rename(columns={'image': 'image_id', 'target': 'label'}, inplace=True)

    splits = {'train': [], 'test': []}

    for client_id in sorted(df['center'].unique()):
        train_subset = df[(df['center'] == client_id) & (df['fold'] == 'train')]
        test_subset = df[(df['center'] == client_id) & (df['fold'] == 'test')]

        for _, row in train_subset.iterrows():
            splits['train'].append([row['image_id'], row['label'], client_id])
        for _, row in test_subset.iterrows():
            splits['test'].append([row['image_id'], row['label'], client_id])
    
    return splits