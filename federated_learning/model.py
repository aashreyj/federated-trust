from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=4):
        """
        Initialize the BaseModel Architecture
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Forward pass of the model"""
        return self.net(x)

    def get_weights(self):
        """Return model parameters as a list of NumPy arrays"""
        weights = []
        for k, v in self.state_dict().items():
            arr = v.detach().cpu().numpy()
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            weights.append(arr)
        return weights

    def set_weights(self, weights):
        """
        Set model parameters from a list of NumPy arrays
        """
        state_dict = self.state_dict()
        if len(weights) != len(state_dict):
            raise ValueError(
                f"weights length {len(weights)} != state_dict keys {len(state_dict)}"
            )

        new_state = OrderedDict()
        for (key, ref_tensor), arr in zip(state_dict.items(), weights):
            arr = np.array(arr)
            if tuple(arr.shape) != tuple(ref_tensor.shape):
                raise ValueError(
                    f"Shape mismatch for key '{key}': expected {tuple(ref_tensor.shape)}, got {arr.shape}"
                )
            tensor = torch.tensor(arr, dtype=ref_tensor.dtype)
            new_state[key] = tensor
        self.load_state_dict(new_state)
