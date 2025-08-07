# import pandas as pd
# import os

# csv_path = 'train_test_split.csv'
# df = pd.read_csv(csv_path, sep='\t')
# df.rename(columns={'image': 'image_id', 'target': 'label'}, inplace=True)

# output_dir = './splits'
# os.makedirs(output_dir, exist_ok=True)

# for client_id in sorted(df['center'].unique()):
#     train_df = df[(df['center'] == client_id) & (df['fold'] == 'train')][['image_id', 'label']]
#     test_df = df[(df['center'] == client_id) & (df['fold'] == 'test')][['image_id', 'label']]

#     train_df.to_csv(os.path.join(output_dir, f'train_client{client_id}.csv'), index=False)
#     test_df.to_csv(os.path.join(output_dir, f'test_client{client_id}.csv'), index=False)

import os
import pandas as pd
import numpy as np

def validate_fedisic(data_dir='data/fedisic', npy_dir='data/FedISIC_npy'):
    print("Validating FedISIC setup...\n")

    # 1. Check split CSVs
    split_dir = os.path.join(data_dir, 'splits')
    required_splits = ['train.csv', 'test.csv']
    for fname in required_splits:
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            return False
        df = pd.read_csv(path)
        if 'image_id' not in df.columns or 'label' not in df.columns:
            print(f"[ERROR] CSV missing required columns: {path}")
            return False
        print(f"[OK] Found and validated: {path} ({len(df)} samples)")

    # 2. Check some .npy files
    sample_df = pd.read_csv(os.path.join(split_dir, 'train.csv'))
    sample_ids = sample_df['image_id'].values[:5]

    for img_id in sample_ids:
        npy_path = os.path.join(npy_dir, f"{img_id}.npy")
        if not os.path.exists(npy_path):
            print(f"[ERROR] Missing image file: {npy_path}")
            return False
        try:
            img = np.load(npy_path)
            assert img.ndim == 3 and img.shape[-1] == 3
        except Exception as e:
            print(f"[ERROR] Failed loading image: {npy_path} - {e}")
            return False
        print(f"[OK] Image check passed: {npy_path}")

    print("\n✅ All checks passed. Ready to run LoGo on FedISIC.\n")
    return True

if __name__ == "__main__":
    validate_fedisic()