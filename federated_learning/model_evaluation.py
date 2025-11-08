import pickle
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

from constants import OUTPUT_DATASET_DIR, FEATURES
from federated_learning.model import BaseModel
from federated_learning.dataset import TrustDataset


def load_model_from_checkpoint(checkpoint_path):
    """Load model parameters from checkpoint file"""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    model = BaseModel()
    model.set_weights(data["global_parameters"])
    model.eval()
    return model


def evaluate_checkpoint(checkpoint_path, output_dir=None):
    """
    Evaluate a saved model checkpoint on the global test set
    """
    print(f"Loading model from: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path)

    # Load test data
    test_df = pd.read_csv(f"{OUTPUT_DATASET_DIR}/global_test.csv")
    X_test, y_test = test_df[FEATURES].values, test_df["Class"].values
    test_dataset = TrustDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Get predictions
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    num_classes = len(np.unique(all_labels))

    # Overall metrics
    overall_accuracy = accuracy_score(all_labels, all_predictions)

    # Per-class metrics
    precision_per_class = precision_score(
        all_labels, all_predictions, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        all_labels, all_predictions, average=None, zero_division=0
    )
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    f05_per_class = fbeta_score(
        all_labels, all_predictions, beta=0.5, average=None, zero_division=0
    )

    # Per-class accuracy
    class_accuracy = []
    for cls in range(num_classes):
        mask = all_labels == cls
        if mask.sum() > 0:
            acc = accuracy_score(all_labels[mask], all_predictions[mask])
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0.0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Print results
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print("\nPer-Class Metrics:")

    print(
        f"{'Class':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'F0.5':<12}"
    )
    print("-" * 70)

    for cls in range(num_classes):
        print(
            f"{cls:<8} {class_accuracy[cls]:<12.4f} {precision_per_class[cls]:<12.4f} "
            f"{recall_per_class[cls]:<12.4f} {f1_per_class[cls]:<12.4f} {f05_per_class[cls]:<12.4f}"
        )

    print("\nMacro-averaged metrics:")
    print(
        f"Precision: {precision_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"F1 Score:  {f1_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"F0.5 Score: {fbeta_score(all_labels, all_predictions, beta=0.5, average='macro', zero_division=0):.4f}"
    )

    print("\nWeighted-averaged metrics:")
    print(
        f"Precision: {precision_score(all_labels, all_predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(all_labels, all_predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1 Score:  {f1_score(all_labels, all_predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F0.5 Score: {fbeta_score(all_labels, all_predictions, beta=0.5, average='weighted', zero_division=0):.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Summary:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[f"Class {i}" for i in range(num_classes)],
        )
    )

    # Plot confusion matrix
    if output_dir:
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[f"Class {i}" for i in range(num_classes)],
            yticklabels=[f"Class {i}" for i in range(num_classes)],
        )
        plt.title("Confusion Matrix", fontweight="bold")
        plt.ylabel("True Label", fontweight="bold", labelpad=14)
        plt.xlabel("Predicted Label", fontweight="bold", labelpad=14)

        plt.xticks(fontsize=10, fontweight="bold")
        plt.yticks(fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", pad=12)
        ax.tick_params(axis="y", pad=12)

        output_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nConfusion matrix saved to: {output_path}")
        plt.close()

        metrics = {
            "Accuracy": class_accuracy,
            "Precision": precision_per_class.tolist(),
            "Recall": recall_per_class.tolist(),
            "F1 Score": f1_per_class.tolist(),
            "F0.5 Score": f05_per_class.tolist(),
        }
        metric_names = list(metrics.keys())
        n_metrics = len(metric_names)
        class_labels = [f"Class {i}" for i in range(num_classes)]

        bar_width = 0.15
        x = np.arange(n_metrics)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, cls in enumerate(class_labels):
            vals = [metrics[m][i] for m in metric_names]
            offset = (i - (num_classes - 1) / 2) * bar_width
            ax.bar(x + offset, vals, width=bar_width, label=cls)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontweight="bold")
        ax.set_ylabel("Score", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=11)
        ax.legend(title="Class", frameon=False, bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
        plt.tight_layout(rect=(0, 0, 0.85, 1.0))

        metrics_output_path = f"{output_dir}/metrics_by_class.png"
        fig.savefig(metrics_output_path, dpi=300, bbox_inches="tight")
        print(f"Metrics bar chart saved to: {metrics_output_path}")
        plt.close()


model_checkpoint = sys.argv[1]
confusion_matrix_output_dir = sys.argv[2] if len(sys.argv) > 2 else None

evaluate_checkpoint(model_checkpoint, confusion_matrix_output_dir)
