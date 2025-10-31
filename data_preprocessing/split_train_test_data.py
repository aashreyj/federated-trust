import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from constants import (
    DISTANCES,
    FEATURES,
    PROCESSED_DATA_DIR,
    OUTPUT_DATASET_DIR,
    SCALER_PATH,
)


def perform_test_train_split():
    """
    Splits the preprocessed data into training and testing sets for each distance
    """
    global_test_list = []

    for idx, distance in enumerate(DISTANCES):
        file_path = os.path.join(PROCESSED_DATA_DIR, f"node_{distance}_data.csv")
        df = pd.read_csv(file_path)

        train_df, test_df = train_test_split(
            df, test_size=0.3, stratify=df["Class"], random_state=42
        )
        train_file_path = os.path.join(OUTPUT_DATASET_DIR, f"node_{idx}_train.csv")
        train_df.to_csv(train_file_path, index=False)
        global_test_list.append(test_df)

    global_test_df = pd.concat(global_test_list)
    global_test_df.to_csv(
        os.path.join(OUTPUT_DATASET_DIR, "global_test.csv"), index=False
    )


def scale_datasets():
    """
    Scales the features of the datasets using Min-Max scaling
    """

    df_1m = pd.read_csv(f"{OUTPUT_DATASET_DIR}/node_0_train.csv")
    df_2m = pd.read_csv(f"{OUTPUT_DATASET_DIR}/node_1_train.csv")
    df_3m = pd.read_csv(f"{OUTPUT_DATASET_DIR}/node_2_train.csv")

    all_data = pd.concat([df_1m[FEATURES], df_2m[FEATURES], df_3m[FEATURES]])

    global_scaler = MinMaxScaler()
    global_scaler.fit(all_data)

    joblib.dump(global_scaler, SCALER_PATH)


perform_test_train_split()
scale_datasets()
print("Train-test split and scaling completed.")
