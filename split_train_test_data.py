import os
import pandas as pd
from sklearn.model_selection import train_test_split

PREPROCESSED_DATA_DIR = "data/preprocessed"
OUTPUT_DIR = "data/dataset"
DISTANCES = ["1m", "2m", "3m"]

global_test_list = []

for distance in DISTANCES:
    file_path = os.path.join(PREPROCESSED_DATA_DIR, f"node_{distance}_data.csv")
    df = pd.read_csv(file_path)

    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['Class'], random_state=42)
    train_file_path = os.path.join(OUTPUT_DIR, f"node_{distance}_train.csv")
    train_df.to_csv(train_file_path, index=False)
    global_test_list.append(test_df)

global_test_df = pd.concat(global_test_list)
global_test_df.to_csv(os.path.join(OUTPUT_DIR, "global_test.csv"), index=False)
