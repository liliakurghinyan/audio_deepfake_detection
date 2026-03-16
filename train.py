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


def find_best_threshold_by_f1(y_true, y_score):
    thresholds = np.linspace(0.0, 1.0, 201)
    best_threshold = 0.5
    best_f1 = -1.0

    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    return float(best_threshold), float(best_f1)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics


def validate(model, val_loader, device):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x).squeeze(1).cpu()
            probs = torch.sigmoid(logits)

            y_true.extend(y.numpy().tolist())
            y_score.extend(probs.numpy().tolist())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    best_threshold, best_f1 = find_best_threshold_by_f1(y_true, y_score)
    eer, eer_threshold = compute_eer(y_true, y_score)

    metrics = compute_metrics(y_true, y_score, threshold=best_threshold)
    metrics["best_threshold"] = best_threshold
    metrics["best_f1_from_search"] = best_f1
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold

    return metrics


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    feat = MelFeatureExtractor(sr=16000, n_mels=80)

    train_ds = AudioDeepfakeDataset("dataset/train", feat)
    val_ds = AudioDeepfakeDataset("dataset/val", feat)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = SimpleCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_epoch = -1
    best_metrics = None

    os.makedirs("results", exist_ok=True)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_metrics = validate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss       : {avg_loss:.4f}")
        print(f"Val Accuracy     : {val_metrics['accuracy']:.4f}")
        print(f"Val Precision    : {val_metrics['precision']:.4f}")
        print(f"Val Recall       : {val_metrics['recall']:.4f}")
        print(f"Val F1-score     : {val_metrics['f1_score']:.4f}")
        print(f"Val ROC-AUC      : {val_metrics['roc_auc']:.4f}")
        print(f"Val EER          : {val_metrics['eer']:.4f}")
        print(f"Best Threshold   : {val_metrics['best_threshold']:.4f}")
        print(f"Confusion Matrix : {val_metrics['confusion_matrix']}")

        # choose best model by validation AUC
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            best_metrics = val_metrics

            torch.save(model.state_dict(), "model.pt")

            with open("results/val_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "train_loss": round(avg_loss, 4),
                        **{k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()}
                    },
                    f,
                    indent=4
                )

            with open("results/best_threshold.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"threshold": best_metrics["best_threshold"]},
                    f,
                    indent=4
                )

            print("✅ Saved best model and validation metrics")

    print("\n=== Training finished ===")
    print("Best epoch       :", best_epoch)
    print("Best val ROC-AUC :", round(best_val_auc, 4))


if __name__ == "__main__":
    train()
