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
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_weights(self, weights):
        """
        Set model parameters from a list of NumPy arrays
        """
        state_dict = self.state_dict()
        for key, value in zip(state_dict.keys(), weights):
            state_dict[key] = torch.tensor(value)
        self.load_state_dict(state_dict)
