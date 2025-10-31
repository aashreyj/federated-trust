from pathlib import Path
import pickle

import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.common.logger import log
from logging import INFO


class FedAvgWithModelSaving(fl.server.strategy.FedAvg):
    """
    This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """

    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def save_global_model(self, server_round: int, parameters):
        """Save the parameters to disk."""

        ndarrays = parameters_to_ndarrays(parameters)
        data = {"global_parameters": ndarrays}
        filename = str(self.save_path / f"parameters_round_{server_round}.pkl")
        with open(filename, "wb") as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        self.save_global_model(server_round, parameters)

        return super().evaluate(server_round, parameters)
