import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from dataset import AudioDeepfakeDataset
from features import MelFeatureExtractor
from model import SimpleCNN


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    feat = MelFeatureExtractor(sr=16000, n_mels=80)
    test_ds = AudioDeepfakeDataset("dataset/test", feat)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    with open("results/best_threshold.json", "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    y_true = []
    y_score = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x).squeeze(1).cpu()
            probs = torch.sigmoid(logits)

            y_true.extend(y.numpy().tolist())
            y_score.extend(probs.numpy().tolist())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    eer, eer_threshold = compute_eer(y_true, y_score)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "test_auc": round(float(auc), 4),
        "eer": round(float(eer), 4),
        "eer_threshold": round(float(eer_threshold), 4),
        "chosen_threshold_from_validation": round(float(threshold), 4),
        "dataset_size": int(len(test_ds)),
        "model_name": "SimpleCNN + MelSpectrogram",
        "confusion_matrix": cm.tolist()
    }

    print("\n=== Test Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n✅ Saved test metrics to results/metrics.json")


if __name__ == "__main__":
    evaluate()
