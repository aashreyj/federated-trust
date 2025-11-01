import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from constants import (
    FEATURES,
    PROCESSED_DATA_DIR,
    OUTPUT_DATASET_DIR,
    SCALER_PATH,
)


def split_and_scale_datasets():
    """
    Scales the features of the datasets using Min-Max scaling
    """

    df_1m = pd.read_csv(f"{PROCESSED_DATA_DIR}/node_1m_data.csv")
    df_2m = pd.read_csv(f"{PROCESSED_DATA_DIR}/node_2m_data.csv")
    df_3m = pd.read_csv(f"{PROCESSED_DATA_DIR}/node_3m_data.csv")

    class0_mask = df_2m['Class'] == 0
    class0_indices = df_2m[class0_mask].sample(frac=0.5, random_state=42).index
    other_indices = df_2m[~class0_mask].index
    df_2m = df_2m.loc[class0_indices.union(other_indices)].reset_index(drop=True)

    all_data = pd.concat([df for df in [df_1m, df_2m, df_3m]])
    all_train, all_test = train_test_split(
        all_data, test_size=0.3, random_state=42, stratify=all_data["Class"]
    )

    global_scaler = MinMaxScaler()
    global_scaler.fit(all_train[FEATURES])

    joblib.dump(global_scaler, SCALER_PATH)

    train_splits = np.array_split(all_train, 3)
    for i, split_df in enumerate(train_splits):
        split_df.to_csv(
            os.path.join(OUTPUT_DATASET_DIR, f"node_{i}_train.csv"), index=False
        )

    all_test.to_csv(
        os.path.join(OUTPUT_DATASET_DIR, "global_test.csv"), index=False
    )


split_and_scale_datasets()
print("Train-test split and scaling completed.")
