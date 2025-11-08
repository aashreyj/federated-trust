<H1 style="text-align: center;"> CS8.501: Research in Information Security </H1>
<H3 style="text-align: center;"> Trust-aware Authentication and Authorization for IoT Devices </H3>

Lightweight Flower-based Federated Learning project for trust classification using physical layer features. This repo implements a server and 3 clients, per-client preprocessing, dataset generation, model checkpoints, and evaluation utilities.

## Requirements

- Python 3.10+ (used in development)
- Recommended packages: torch, flwr, pandas, scikit-learn, joblib, seaborn, matplotlib, numpy
- Use a virtualenv:

    ```python3
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Project structure

- federated_learning/
    - server.py — starts Flower server and evaluates global model each round
    - custom_strategy.py — FedAvgWithModelSaving
    - base_client.py — base client implementation used by client scripts
    - model.py — BaseModel, get/set weights utilities
    - dataset.py — TrustDataset wrapper
    - model_evaluation.py — evaluate saved checkpoints, produce confusion matrix & metric plots

- data_preprocessing/
    - generate_dataset.py — parse raw captures into per-distance CSVs
    - generate_trust_label.py — compute `Trust Factor` and map to `Class` labels
    - split_train_test_data.py — split, scale and write per-client train / global test CSVs
    - data/preprocessed/ — processed per-distance CSVs

## Execution Steps

1. Activate your virtualenv
   - source .venv/bin/activate

2. Generate the dataset

    ```python3
    python3 -m data_preprocessing.generate_dataset
   python3 -m data_preprocessing.generate_trust_label
   python3 -m data_preprocessing.split_train_test_data
   ```

3. Start Flower server (run in its own terminal)
   - python3 -m federated_learning.server

4. Start client processes (one per client, separate terminals)
   - python3 -m federated_learning.client_script_name

5. After training, evaluate a saved checkpoint:
   - python3 -m federated_learning.model_evaluation artifacts/model_checkpoints/parameters_round_x.pkl --output-dir artifacts

## Useful constants

Edit `constants.py` to change training behavior:
- `NUM_ROUNDS`, `CLIENTS_REQUIRED`, `BATCH_SIZE`, `WINDOW_SIZE`, `SCALER_PATH`, `NUM_CLASSES`
