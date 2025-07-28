import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import ast

class FedPolypDataset(Dataset):
    def __init__(self, data_paths, image_size=(224, 224)):
        self.data_paths = data_paths
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size)
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sample_path = self.data_paths[idx]
        sample = np.load(sample_path, allow_pickle=True).item()
        image = sample['image']  # shape: (H, W, 3)
        mask = sample['mask']    # shape: (H, W)

        # Resize and convert image
        image = Image.fromarray(image.astype(np.uint8))
        image = self.transform(image)  # result: [3, 224, 224]

        # Resize mask to 224x224 (nearest to preserve class labels)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()  # result: [224, 224]

        return image, mask, idx

def load_txt_splits(txt_dir):
    splits = {}
    for client_id in range(1, 5):
        for split in ['train', 'test']:
            file_path = os.path.join(txt_dir, f'client{client_id}_{split}.txt')
            with open(file_path, 'r') as f:
                line = f.read().strip()
                paths = ast.literal_eval(line)
            splits[f'client{client_id}_{split}'] = paths
    return splits

def get_fedpolyp_loaders(config, splits):
    # Create dataset instance
    train_paths = []
    user_groups = {}
    for cid in range(1, config['num_clients'] + 1):
        key = f'client{cid}_train'
        paths = splits[key]
        user_groups[cid] = list(range(len(train_paths), len(train_paths) + len(paths)))
        train_paths.extend(paths)

    train_dataset = FedPolypDataset(train_paths)

    test_paths = []
    for cid in range(1, config['num_clients'] + 1):
        key = f'client{cid}_test'
        test_paths.extend(splits[key])
    test_dataset = FedPolypDataset(test_paths)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_dataset, user_groups, test_loader

def load_csv_splits(_):
    raise NotImplementedError("Use load_txt_splits instead for FedPolyp.")