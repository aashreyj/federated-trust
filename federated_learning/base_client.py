import os
import logging

import flwr as fl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import ARTIFACTS_DIR, BATCH_SIZE, FEATURES, FL_DIR, OUTPUT_DATASET_DIR
from federated_learning.dataset import TrustDataset
from federated_learning.model import BaseModel

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.addFilter(lambda record: record.levelno in (logging.INFO, logging.ERROR))


class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, client_id, params):
        """
        Initialize the PyTorchClient
        """
        self.client_id = client_id
        self.data_file = f"{OUTPUT_DATASET_DIR}/node_{client_id}_train.csv"
        self.output_dir = f"{ARTIFACTS_DIR}/client_{client_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = BaseModel().to(self.device)
        self.trainloader = None
        self.valloader = None
        self.scaler = None
        self.round = 0
        self.params = params

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        return self.model.get_weights()

    def load_data(self):
        """Loads data, creates Datasets, and Dataloaders."""
        df = pd.read_csv(self.data_file)

        df_train, df_val = train_test_split(
            df, test_size=0.2, stratify=df["Class"], random_state=42
        )

        X_train = df_train[FEATURES].values
        y_train = df_train["Class"].values

        train_dataset = TrustDataset(X_train, y_train)
        self.trainloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        X_val = df_val[FEATURES].values
        y_val = df_val["Class"].values

        val_dataset = TrustDataset(X_val, y_val)
        self.valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def evaluate(self, parameters, config):
        """
        Evaluate model on validation set
        """
        self.model.set_weights(parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        with torch.no_grad():
            for features, labels in self.valloader:
                if isinstance(features, torch.Tensor):
                    features = features.float().to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.long().to(self.device)

                outputs = self.model(features)
                batch_loss = criterion(outputs, labels).item()
                loss += batch_loss * labels.size(0)
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss / total if total > 0 else 0.0
        return (
            avg_loss,
            total,
            {"accuracy": accuracy},
        )

    def evaluate_model(self, criterion):
        """
        Evaluate model on validation set
        """
        val_correct, val_total, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for features, labels in self.valloader:
                features = features.float().to(self.device)
                labels = labels.long().to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                preds = torch.max(outputs.data, 1)[1]
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_running_loss += loss.item() * labels.size(0)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_running_loss / val_total if val_total > 0 else 0.0
        self.val_accuracies.append(val_accuracy)
        self.val_losses.append(val_loss)

    def plot_graphs(self):
        """
        Plot and save training and validation accuracy/loss curves.
        """
        epochs = range(1, self.params["epochs"] + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_accuracies, label="Train Accuracy")
        plt.plot(epochs, self.val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy per Epoch")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss per Epoch")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/training_curves_client_{self.client_id}_round_{self.round}.png"
        )
        plt.close()
