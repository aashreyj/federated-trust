import sys

import flwr as fl
import torch

from constants import SERVER_SOCKET
from federated_learning.base_client import PyTorchClient

CLIENT_ID = sys.argv[1]
PARAMS = {
    "lr": 0.001,
    "epochs": 40,
}


# PARAMS = {
#     "lr": 0.0005,
#     "epochs": 50,
# }


class LM_Client(PyTorchClient):
    def __init__(self, client_id, params):
        super().__init__(client_id, params)

    def fit(self, parameters, config):
        """
        Train the model on the local dataset and evaluate on validation set.
        Plots and saves accuracy and loss curves at the end of training.
        """

        self.model.set_weights(parameters)
        self.round += 1
        if self.trainloader is None or self.valloader is None:
            self.load_data()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        return super().fit(parameters, config)


lm_client = LM_Client(client_id=CLIENT_ID, params=PARAMS).to_client()
fl.client.start_client(server_address=SERVER_SOCKET, client=lm_client)
