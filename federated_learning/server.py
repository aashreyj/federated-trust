import logging

import flwr as fl
from flwr.common.logger import log
from logging import INFO

from constants import SERVER_SOCKET, NUM_ROUNDS, CLIENTS_REQUIRED, ARTIFACTS_DIR
from federated_learning.custom_strategy import FedAvgWithModelSaving

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

    log(INFO, f"Round accuracy (weighted): {weighted_train_accuracy:.4f}")
    log(INFO, f"Round validation accuracy (weighted): {weighted_val_accuracy:.4f}")

    return {
        "train_accuracy": weighted_train_accuracy,
        "val_accuracy": weighted_val_accuracy,
    }


strategy = FedAvgWithModelSaving(
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    min_fit_clients=CLIENTS_REQUIRED,
    min_available_clients=CLIENTS_REQUIRED,
    save_path=f"{ARTIFACTS_DIR}/model_checkpoints",
)

print("Starting Flower server...")
fl.server.start_server(
    server_address=SERVER_SOCKET,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
