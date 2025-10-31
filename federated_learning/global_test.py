import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle

from federated_learning.model import BaseModel
from federated_learning.dataset import TrustDataset
from constants import FEATURES, ARTIFACTS_DIR, NUM_ROUNDS, OUTPUT_DATASET_DIR

# Load global model parameters
with open(
    f"{ARTIFACTS_DIR}/model_checkpoints/parameters_round_{NUM_ROUNDS}.pkl", "rb"
) as f:
    data = pickle.load(f)
global_parameters = data["global_parameters"]

# Initialize and set model weights
model = BaseModel()
model.set_weights(global_parameters)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load global test dataset
df_test = pd.read_csv(f"{OUTPUT_DATASET_DIR}/global_test.csv")
X_test = df_test[FEATURES].values
y_test = df_test["Class"].values

test_dataset = TrustDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
correct, total, running_loss = 0, 0, 0.0

with torch.no_grad():
    for features, labels in test_loader:
        features = features.float().to(device)
        labels = labels.long().to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        preds = torch.max(outputs.data, 1)[1]
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)

test_accuracy = correct / total if total > 0 else 0.0
test_loss = running_loss / total if total > 0 else 0.0

print(f"Global Test Accuracy: {test_accuracy:.4f}")
print(f"Global Test Loss: {test_loss:.4f}")
print(f"Test Samples: {total}")
