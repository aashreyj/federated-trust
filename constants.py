SERVER_SOCKET = "127.0.0.1:6000"
NUM_ROUNDS = 6
CLIENTS_REQUIRED = 3

COLUMNS = ["RSSI", "LQI", "BatteryLevel", "BatteryDiff", "AntennaAngle"]
DISTANCES = ["1m", "2m", "3m"]
FEATURES = ["RSSI", "LQI", "BatteryLevel", "BatteryDiff"]

FL_DIR = "federated_learning"
OUTPUT_DATASET_DIR = "data/dataset"
PROCESSED_DATA_DIR = "data/preprocessed"
RAW_DATA_DIR = "data/raw"
ARTIFACTS_DIR = "artifacts"
SCALER_PATH = "artifacts/global_scaler.pkl"

BATCH_SIZE = 32
WINDOW_SIZE = 600
