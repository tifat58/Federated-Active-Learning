import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==== FILE PATHS ====
metadata_path = '/home/student_account/Desktop/Aditya/FEAL-master/data/ISIC_2019_Training_Metadata.csv'
gt_path = '/home/student_account/Desktop/Aditya/FEAL-master/data/ISIC_2019_Training_GroundTruth.csv'
save_dir = '/home/student_account/Desktop/Aditya/FEAL-master/data/data_split/splits'

os.makedirs(save_dir, exist_ok=True)

# ==== LOAD DATA ====
metadata = pd.read_csv(metadata_path)
gt = pd.read_csv(gt_path)

# Standardize column
metadata['image'] = metadata['image'].apply(lambda x: f"{x}.npy" if not str(x).endswith('.npy') else x)
gt['image'] = gt['image'].apply(lambda x: f"{x}.npy" if not str(x).endswith('.npy') else x)

# Merge both files
merged = pd.merge(metadata, gt, on='image')

# One-hot → class index
diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
merged['diagnosis'] = merged[diagnosis_cols].values.argmax(axis=1)

# Extract prefix from lesion_id
merged['prefix'] = merged['lesion_id'].str.extract(r'^([A-Z]+)')

# ==== CLIENT DEFINITIONS (based on lesion_id prefix and fixed sizes) ====
clients = {
    'client_0': merged[merged['prefix'] == 'BCN'].sample(n=12413, random_state=42),  # Full BCN
    'client_1': merged[(merged['prefix'] == 'HAM')].iloc[0:3954],  # HAM → molemax
    'client_2': merged[(merged['prefix'] == 'HAM')].iloc[3954:7317],  # HAM → modern
    'client_3': merged[(merged['prefix'] == 'HAM')].iloc[7317:10015],  # HAM → rosendahl
}

splits = {'train': [], 'test': [], 'val': []}

# ==== PER-CLIENT SPLITTING ====
split_sizes = {
    'client_0': (9930, 2483),
    'client_1': (3163, 791),
    'client_2': (2691, 672),
    'client_3': (1807, 452),
}

for client_name, df in clients.items():
    train_size, test_size = split_sizes[client_name]
    train_df, test_df = train_test_split(df, stratify=df['diagnosis'], test_size=test_size, train_size=train_size, random_state=42)

    for _, row in train_df.iterrows():
        splits['train'].append({'client': client_name, 'image': row['image'], 'label': int(row['diagnosis'])})

    for _, row in test_df.iterrows():
        splits['test'].append({'client': client_name, 'image': row['image'], 'label': int(row['diagnosis'])})

# ==== SAVE TO CSV ====
for split_name, data in splits.items():
    out_df = pd.DataFrame(data)
    out_path = os.path.join(save_dir, f'{split_name}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"[Saved] {split_name}.csv — {len(out_df)} samples")

print("FedISIC dataset split complete according to the paper.")