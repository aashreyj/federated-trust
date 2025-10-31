import logging

import flwr as fl
from flwr.common.logger import log
from logging import INFO
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

from constants import (
    SERVER_SOCKET,
    NUM_ROUNDS,
    CLIENTS_REQUIRED,
    ARTIFACTS_DIR,
    OUTPUT_DATASET_DIR,
    FEATURES,
)
from federated_learning.custom_strategy import FedAvgWithModelSaving
from federated_learning.model import BaseModel
from federated_learning.dataset import TrustDataset

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.INFO)
flwr_logger.addFilter(lambda record: record.levelno in (logging.INFO, logging.ERROR))


def aggregate_fit_metrics(metrics):
    """
    This function aggregates the metrics from all clients
    after a 'fit' round.
    """
    total_examples = sum([num_examples for num_examples, _ in metrics])

    weighted_train_accuracy = (
        sum([num_examples * m["train_accuracy"] for num_examples, m in metrics])
        / total_examples
    )

    weighted_val_accuracy = (
        sum([num_examples * m["val_accuracy"] for num_examples, m in metrics])
        / total_examples
    )

    class_metrics = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if "class_" in k:
                if k not in class_metrics:
                    class_metrics[k] = []
                class_metrics[k].append((num_examples, v))

    aggregated_class_metrics = {}
    for k, values in class_metrics.items():
        agg = sum([n * v for n, v in values]) / sum([n for n, _ in values])
        aggregated_class_metrics[k] = agg
        log(INFO, f"{k} (weighted): {agg:.4f}")

    log(INFO, f"Round accuracy (weighted): {weighted_train_accuracy:.4f}")
    log(INFO, f"Round validation accuracy (weighted): {weighted_val_accuracy:.4f}")

    return {
        "train_accuracy": weighted_train_accuracy,
        "val_accuracy": weighted_val_accuracy,
        **aggregated_class_metrics,
    }


def evaluate_model(server_round, parameters, config):
    """Evaluate the global model using a held-out test dataset."""
    test_df = pd.read_csv(f"{OUTPUT_DATASET_DIR}/global_test.csv")
    X_test, y_test = test_df[FEATURES].values, test_df["Class"].values
    test_dataset = TrustDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = BaseModel()
    model.set_weights(parameters)

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    num_classes = len(np.unique(y_test))
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    avg_loss = running_loss / total if total > 0 else 0.0
    test_accuracy = correct / total if total > 0 else 0.0

    metrics = {"accuracy": test_accuracy}
    for cls in range(num_classes):
        acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        metrics[f"class_{cls}_accuracy"] = acc

    log(INFO, f"Server-side evaluation after round {server_round}: loss={avg_loss:.4f}, accuracy={test_accuracy:.4f}")
    for cls in range(num_classes):
        log(INFO, f"Class {cls} accuracy after round {server_round}: {metrics[f'class_{cls}_accuracy']:.4f}")

    return avg_loss, metrics


strategy = FedAvgWithModelSaving(
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    min_fit_clients=CLIENTS_REQUIRED,
    min_available_clients=CLIENTS_REQUIRED,
    save_path=f"{ARTIFACTS_DIR}/model_checkpoints",
    evaluate_fn=evaluate_model,
)

print("Starting Flower server...")
fl.server.start_server(
    server_address=SERVER_SOCKET,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
