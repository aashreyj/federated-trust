import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import SCALER_PATH, FEATURES


class TrustDataset(Dataset):
    """
    PyTorch Dataset for loading the trust data.
    """

    def __init__(self, data, labels, scaler_path=SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        self.data = self.scaler.transform(pd.DataFrame(data, columns=FEATURES))

        self.features = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_scaler(self):
        return self.scaler
