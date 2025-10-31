from collections import deque
import os

import numpy as np
import pandas as pd

from constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES, DISTANCES, WINDOW_SIZE


class TrustModel:
    """
    Implements the stateful, sliding-window logic of Algorithm 1
    """

    def __init__(self, window_size, features, phi=3.5):
        """
        Initializes the model
        """
        self.window_size = window_size
        self.features = features
        self.phi = phi

        self.history = deque(maxlen=window_size)

        self.thresh_strict = phi - 2
        self.thresh_medium = phi - 1
        self.thresh_loose = phi

    def _calculate_stats(self):
        """
        Calculates the median and MAD from the current history buffer.
        Returns: (medians, mads) as pandas Series.
        """
        hist_df = pd.DataFrame(list(self.history))

        # calculate medians
        medians = hist_df[self.features].median()

        # calculate Median Absolute Deviation (MAD)
        errors = np.abs(hist_df[self.features] - medians)
        mads = errors.median()

        return medians, mads

    def predict_one(self, packet):
        """
        Analyzes a single new packet and returns a trust score
        """

        # --- Priming Phase ---
        if len(self.history) < self.window_size:
            self.history.append(packet)
            return 1.0

        # --- Calculation Phase ---
        medians, mads = self._calculate_stats()

        # Calculate Z-scores for the new packet
        z_scores = []
        for f in self.features:
            median = medians[f]
            mad = mads[f]

            if mad == 0:
                z = 0.0 if packet[f] == median else np.inf
            else:
                z = (packet[f] - median) / mad

            z_scores.append(np.abs(z))

        # --- Trust Assignment ---
        max_z = np.max(z_scores)

        if max_z <= self.thresh_strict:
            trust_score = 1.00
        elif max_z <= self.thresh_medium:
            trust_score = 0.85
        elif max_z <= self.thresh_loose:
            trust_score = 0.70
        else:
            trust_score = 0.00

        # --- History Update ---
        if trust_score > 0:
            self.history.append(packet)

        return trust_score


def generate_trust_labels(df_device, features, model_instance):
    """
    Runner function to apply the TrustModel to an entire DataFrame.
    """

    trust_labels = []

    for _, row in df_device.iterrows():
        packet = row[features].to_dict()
        trust = model_instance.predict_one(packet)

        trust_labels.append(trust)

    return trust_labels


for distance in DISTANCES:
    input_file_path = os.path.join(RAW_DATA_DIR, f"node_{distance}_data_raw.csv")
    output_file_path = os.path.join(PROCESSED_DATA_DIR, f"node_{distance}_data.csv")

    print("Processing file:", input_file_path)
    does_output_file_exist = os.path.exists(output_file_path)

    df = pd.read_csv(input_file_path)
    df_normal = df[df["AntennaAngle"] == 90].copy()
    df_attack = df[df["AntennaAngle"] != 90].copy()

    trust_model = TrustModel(window_size=WINDOW_SIZE, features=FEATURES)

    normal_labels = generate_trust_labels(
        df_device=df_normal, features=FEATURES, model_instance=trust_model
    )
    df_normal["Trust_Factor"] = normal_labels

    attack_labels = generate_trust_labels(
        df_device=df_attack, features=FEATURES, model_instance=trust_model
    )
    df_attack["Trust_Factor"] = attack_labels

    df_final = pd.concat([df_normal, df_attack]).sort_index()

    label_map = {1.0: 3, 0.85: 2, 0.70: 1, 0.0: 0}
    df_final["Class"] = df_final["Trust_Factor"].map(label_map)

    print("Label generation complete.")
    print(df_final["Class"].value_counts())
    print()

    df_final.to_csv(output_file_path, mode="w", header=True, index=False)
