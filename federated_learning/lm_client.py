import sys

import flwr as fl
import torch
import torch.nn as nn

from constants import SERVER_SOCKET
from federated_learning.base_client import PyTorchClient

CLIENT_ID = sys.argv[1]
PARAMS = {
    "lr": 0.0001,
    "epochs": 50,
}


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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        criterion = nn.CrossEntropyLoss()

        # Training
        self.model.train()
        self.train_accuracies, self.train_losses = [], []
        self.val_accuracies, self.val_losses = [], []

        for _ in range(self.params["epochs"]):
            correct, total, running_loss = 0, 0, 0.0
            for features, labels in self.trainloader:
                features = features.float().to(self.device)
                labels = labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = torch.max(outputs.data, 1)[1]
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    running_loss += loss.item() * labels.size(0)

            train_accuracy = correct / total if total > 0 else 0.0
            train_loss = running_loss / total if total > 0 else 0.0
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            self.evaluate_model(criterion=criterion)
            self.model.train()

        # Plot and save graphs
        self.plot_graphs()

        avg_val_accuracy = sum(self.val_accuracies) / self.params["epochs"]
        avg_train_accuracy = sum(self.train_accuracies) / self.params["epochs"]

        return (
            self.model.get_weights(),
            len(self.trainloader.dataset),
            {
                "val_accuracy": avg_val_accuracy,
                "train_accuracy": avg_train_accuracy,
            },
        )


lm_client = LM_Client(client_id=CLIENT_ID, params=PARAMS).to_client()
fl.client.start_client(server_address=SERVER_SOCKET, client=lm_client)
