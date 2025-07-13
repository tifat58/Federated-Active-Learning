
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FedISICDataset(Dataset):
    def __init__(self, csv_file, data_root, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        image_name = self.data_info.iloc[idx]['image']
        image_path = os.path.join(self.data_root, image_name)
        image = np.load(image_path).astype(np.float32)
        label = int(self.data_info.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image=image)['image']

        return idx, {'image': torch.from_numpy(image.transpose(2, 0, 1)), 'label': torch.tensor(label)}
