import os

import pandas as pd

from constants import COLUMNS, RAW_DATA_DIR


for filename in os.listdir(RAW_DATA_DIR):
    print("Processing file:", filename)
    if not filename.startswith("DataCap"):
        continue

    # set input and output file paths
    file_path = os.path.join(RAW_DATA_DIR, filename)
    split_filename = filename.split("-")
    antenna_angle, distance = split_filename[-2], split_filename[-1].split(".")[0]
    output_file_path = os.path.join(RAW_DATA_DIR, f"node_{distance}_data_raw.csv")
    does_output_file_exist = os.path.exists(output_file_path)

    df = pd.read_csv(file_path, header=None)

    # randomly undersample rows of node placed at 2m with antenna angle 180 to balance dataset
    if distance == "2m" and antenna_angle == "180":
        df = df.sample(frac=0.4, random_state=42).reset_index(drop=True)

    # select relevant features and save to output file
    output_df = pd.DataFrame(columns=COLUMNS)
    output_df["RSSI"] = df[17].str.split().str[0]
    output_df["LQI"] = df[18]
    output_df["BatteryLevel"] = df[9]
    output_df["BatteryDiff"] = output_df["BatteryLevel"].diff().fillna(0)
    output_df["AntennaAngle"] = int(antenna_angle)

    output_df.to_csv(
        output_file_path, mode="a", header=not does_output_file_exist, index=False
    )

print("Dataset generation completed.")
