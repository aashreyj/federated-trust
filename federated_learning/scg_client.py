import sys

import flwr as fl
import torch

from constants import SERVER_SOCKET
from federated_learning.base_client import PyTorchClient

CLIENT_ID = sys.argv[1]
PARAMS = {
    "lr": 0.001,
    "epochs": 40,
    "momentum": 0.8,
    "step_size": 10,
    "gamma": 0.9,
}


# PARAMS = {
#     "lr": 0.0005,
#     "epochs": 50,
#     "momentum": 0.8,
#     "step_size": 15,
#     "gamma": 0.9,
# }


class SCG_Client(PyTorchClient):
    def __init__(self, client_id, params):
        super().__init__(client_id, params)

    def fit(self, parameters, config):
        """
        Train the model on the local dataset and evaluate on validation set.
        Plots and saves accuracy and loss curves at the end of training.
        """

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.params["lr"],
            momentum=self.params["momentum"],
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.params["step_size"],
            gamma=self.params["gamma"],
        )

        return super().fit(parameters, config)


scg_client = SCG_Client(client_id=CLIENT_ID, params=PARAMS).to_client()
fl.client.start_client(server_address=SERVER_SOCKET, client=scg_client)
