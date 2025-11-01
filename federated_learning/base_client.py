import os
import logging

import flwr as fl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from flwr.common.logger import log
from logging import INFO
from pprint import pprint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    NUM_CLASSES,
    FEATURES,
    OUTPUT_DATASET_DIR,
    CLASS_WEIGHTS,
)
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
        self.params = params
        self.data_file = f"{OUTPUT_DATASET_DIR}/node_{client_id}_train.csv"
        self.output_dir = f"{ARTIFACTS_DIR}/client_{client_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = BaseModel().to(self.device)
        self.round = 0
        self.optimizer = None
        self.scheduler = None

        class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.trainloader = None
        self.valloader = None
        self.scaler = None

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

    def fit(self, parameters, config):
        """
        Train the model on the local dataset.
        """
        self.round += 1

        if self.trainloader is None or self.valloader is None:
            self.load_data()

        num_classes = NUM_CLASSES

        self.model.train()
        self.train_accuracies, self.train_losses = [], []
        self.val_accuracies, self.val_losses = [], []

        for epoch in range(self.params["epochs"]):
            # Training
            correct, total, running_loss = 0, 0, 0.0
            for features, labels in self.trainloader:
                features = features.float().to(self.device)
                labels = labels.long().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss_for_backward = loss.mean() if loss.ndim > 0 else loss
                loss_for_backward.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    running_loss += (
                        loss.sum().item()
                        if loss.ndim > 0
                        else loss.item() * labels.size(0)
                    )

            if self.scheduler:
                self.scheduler.step()

            train_accuracy = correct / total if total > 0 else 0.0
            train_loss = running_loss / total if total > 0 else 0.0
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_correct, val_total, val_running_loss = 0, 0, 0.0
            class_val_correct = [0] * num_classes
            class_val_total = [0] * num_classes

            with torch.no_grad():
                for features, labels in self.valloader:
                    features = features.float().to(self.device)
                    labels = labels.long().to(self.device)
                    outputs = self.model(features)
                    losses = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)

                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    val_running_loss += (
                        losses.sum().item()
                        if losses.ndim > 0
                        else losses.item() * labels.size(0)
                    )

                    for i in range(labels.size(0)):
                        label = labels[i].item()
                        class_val_total[label] += 1
                        if preds[i].item() == label:
                            class_val_correct[label] += 1

            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            val_loss = val_running_loss / val_total if val_total > 0 else 0.0
            self.val_accuracies.append(val_accuracy)
            self.val_losses.append(val_loss)
            self.model.train()

            last_class_correct = class_val_correct
            last_class_total = class_val_total

        avg_val_accuracy = (
            sum(self.val_accuracies) / len(self.val_accuracies)
            if self.val_accuracies
            else 0.0
        )
        avg_train_accuracy = (
            sum(self.train_accuracies) / len(self.train_accuracies)
            if self.train_accuracies
            else 0.0
        )

        class_metrics = {}
        for cls in range(num_classes):
            total = last_class_total[cls]
            correct = last_class_correct[cls]
            class_metrics[f"class_{cls}_accuracy"] = round(
                correct / total if total > 0 else 0.0, 4
            )

        self.plot_graphs()

        log(INFO, f"Client {self.client_id} - Round {self.round} results:")
        pprint(
            {
                "train_accuracy": avg_train_accuracy,
                "val_accuracy": avg_val_accuracy,
                "train_loss": self.train_losses[-1],
                "val_loss": self.val_losses[-1],
                **class_metrics,
            },
        )

        return (
            self.model.get_weights(),
            len(self.trainloader.dataset),
            {
                "train_accuracy": avg_train_accuracy,
                "val_accuracy": avg_val_accuracy
            },
        )

    def evaluate(self, parameters, config):
        """
        Evaluate model on validation set for Flower server
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
