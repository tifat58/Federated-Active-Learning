# import pandas as pd

# metadata_path = "/home/student_account/Desktop/Aditya/FEAL-master/data/ISIC_2019_Training_Metadata.csv"
# metadata = pd.read_csv(metadata_path)

# print("Column names in metadata file:")
# print(metadata.columns.tolist())

# gt_path = "/home/student_account/Desktop/Aditya/FEAL-master/data/ISIC_2019_Training_GroundTruth.csv"
# labels = pd.read_csv(gt_path)
# print("Ground Truth columns:", labels.columns.tolist())

# import pandas as pd

# # Path to metadata
# metadata_path = '/home/student_account/Desktop/Aditya/FEAL-master/data/ISIC_2019_Training_Metadata.csv'

# # Load metadata
# metadata = pd.read_csv(metadata_path)

# # Drop missing lesion_id
# metadata = metadata.dropna(subset=['lesion_id'])

# # Extract prefix before underscore
# metadata['prefix'] = metadata['lesion_id'].astype(str).str.split('_').str[0]

# # Show unique prefixes
# print("Unique lesion_id prefixes:")
# print(metadata['prefix'].value_counts())

import pandas as pd

# Path to your test.csv file
test_csv_path = '/home/student_account/Desktop/Aditya/FEAL-master/data/data_split/splits/train.csv'

# Load CSV
df = pd.read_csv(test_csv_path)

# Count samples per client
client_distribution = df['client'].value_counts()

print(client_distribution)