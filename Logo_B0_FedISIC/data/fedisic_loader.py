import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_csv_splits(csv_dir):
    """
    Load train, val, and test CSVs into dictionary of DataFrames.
    Each CSV should have: image_id,label
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(csv_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        splits[split] = df.values.tolist()  # list of [image_id, label]
    return splits


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
    """
    Load full dataset and return:
    - full_train_dataset
    - dict_users (client partitioning)
    - test_loader (for evaluation)
    """
    data_root = config['dataset']['data_root']
    initial_label_rate = config['initial_label_rate']
    batch_size = config['batch_size']
    num_clients = config['num_clients']

    train_data = csv_splits['train']
    test_data = csv_splits['test']

    train_transform, test_transform = get_transforms()

    full_train_dataset = FedISICDataset(train_data, data_root, transform=train_transform)
    test_dataset = FedISICDataset(test_data, data_root, transform=test_transform)

    # Split train_data indices among clients
    data_indices = np.arange(len(train_data))
    client_splits = np.array_split(data_indices, num_clients)
    user_groups = {cid: list(split) for cid, split in enumerate(client_splits)}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return full_train_dataset, user_groups, test_loader