"""
Phase 1 — LSTM Training with Attention
=======================================
Architecture  : 2-layer BiLSTM + feature-level attention
                Attention weights preserved for XAI (Phase 2)
                Each feature treated as a time step (feature-as-sequence)
Datasets      : CIC_IIoT_2025_consolidated.csv, IDS2025_Balanced_final_with_split.csv
GPU           : RTX A6000 (CUDA:0)

Outputs:
  Models/DeepLearning/
    lstm_{dataset}.pth
    lstm_{dataset}_config.json
  Models/Performance_Metrics/
    dl_lstm_performance_2025.csv
    model_comparison_plots/
      lstm_results_{dataset}.png

Random seed: 42  |  AMP mixed precision: enabled
"""

import os, json, time, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
READY_DIR   = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
MODELS_DIR  = os.path.join(ROOT, "Models", "DeepLearning")
METRICS_DIR = os.path.join(ROOT, "Models", "Performance_Metrics")
PLOTS_DIR   = os.path.join(METRICS_DIR, "model_comparison_plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

DATASETS = {
    "CIC_IIoT_2025":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
    "IDS2025_Balanced": os.path.join(READY_DIR, "IDS2025_Balanced_final_with_split.csv"),
}

CONFIG = {
    "hidden_size": 64,
    "n_layers":    2,
    "dropout":     0.3,
    "bidirectional": True,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "batch_size":  512,
    "max_epochs":  100,
    "patience":    10,
    "seed":        SEED,
}


# ── dataset ──────────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── model ─────────────────────────────────────────────────────────────────────
class FeatureAttention(nn.Module):
    """Soft-attention over LSTM hidden states at each feature position."""

    def __init__(self, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        h = hidden_size * 2 if bidirectional else hidden_size
        self.attn = nn.Linear(h, 1)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (B, seq_len, H)
        scores  = self.attn(lstm_out).squeeze(-1)       # (B, seq_len)
        weights = torch.softmax(scores, dim=-1)          # (B, seq_len)
        context = (weights.unsqueeze(-1) * lstm_out).sum(1)  # (B, H)
        return context, weights


class LSTMClassifier(nn.Module):
    """
    2-layer BiLSTM + feature attention.
    Input: (B, n_features) → treat each feature as a time step (B, n_features, 1).
    Attention weights give per-feature importance (for AttentionExplanation).
    """

    def __init__(self, n_features: int, n_classes: int,
                 hidden_size: int = 64, n_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Input normalization (raw features not normalized in CSV)
        self.input_norm = nn.BatchNorm1d(n_features)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.attention  = FeatureAttention(hidden_size, bidirectional)
        h_out           = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(h_out),
            nn.Dropout(dropout),
            nn.Linear(h_out, n_classes),
        )
        self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) → normalize → (B, n_features, 1) — each feature = one time step
        x = self.input_norm(x)
        x_seq = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_seq)          # (B, n_features, H)
        context, weights = self.attention(lstm_out)
        self.last_attn_weights = weights.detach()
        return self.classifier(context)

    @torch.no_grad()
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """Return attention weights (B, n_features) — feature importance proxy."""
        x = self.input_norm(x)
        x_seq = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_seq)
        _, weights   = self.attention(lstm_out)
        return weights.cpu().numpy()


# ── training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            out  = model(X_batch)
            loss = criterion(out, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(X_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(X_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, le):
    model.eval()
    all_preds, all_labels, all_proba = [], [], []
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        with autocast():
            out  = model(X_batch)
            loss = criterion(out, y_batch)
        total_loss  += loss.item() * len(X_batch)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        all_proba.extend(torch.softmax(out, 1).cpu().numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_proba = np.array(all_proba)
    n       = len(y_true)
    avg     = "binary" if len(le.classes_) == 2 else "weighted"

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")
    return total_loss / n, acc, f1, auc, y_true, y_pred, y_proba


# ── main ──────────────────────────────────────────────────────────────────────
def train_dataset(ds_name: str, ds_path: str, all_metrics: list):
    print(f"\n{'─' * 65}")
    print(f"  DATASET: {ds_name}")
    print(f"{'─' * 65}")

    df = pd.read_csv(ds_path)
    feature_cols = [c for c in df.columns if c not in ("label", "split", "label_original")]
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_enc"].values.astype(np.int64)
    s = df["split"].values

    X_tr, y_tr = X[s == "train"], y[s == "train"]
    X_va, y_va = X[s == "val"],   y[s == "val"]
    X_te, y_te = X[s == "test"],  y[s == "test"]
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")

    train_loader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=CONFIG["batch_size"],
                               shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(TabularDataset(X_va, y_va), batch_size=CONFIG["batch_size"] * 2,
                               shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(TabularDataset(X_te, y_te), batch_size=CONFIG["batch_size"] * 2,
                               shuffle=False, num_workers=4, pin_memory=True)

    model = LSTMClassifier(
        n_features=len(feature_cols), n_classes=len(le.classes_),
        hidden_size=CONFIG["hidden_size"], n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"], bidirectional=CONFIG["bidirectional"]
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    class_counts  = np.bincount(y_tr)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * len(le.classes_)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["max_epochs"])
    scaler    = GradScaler()

    best_val_f1, best_epoch, patience_counter = 0.0, 0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    t0 = time.time()
    print(f"\n  Training (max {CONFIG['max_epochs']} epochs, patience={CONFIG['patience']}) …")
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        va_loss, va_acc, va_f1, va_auc, *_ = evaluate(model, val_loader, criterion, le)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
                  f"| va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_f1={va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1; best_epoch = epoch; patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_encoder_classes": le.classes_,
                "feature_names": feature_cols,
                "config": CONFIG, "dataset": ds_name,
                "best_val_f1": best_val_f1, "best_epoch": best_epoch,
            }, os.path.join(MODELS_DIR, f"lstm_{ds_name}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"  Early stop at epoch {epoch} (best={best_epoch}, val_f1={best_val_f1:.4f})")
                break

    ckpt = torch.load(os.path.join(MODELS_DIR, f"lstm_{ds_name}.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc, te_f1, te_auc, y_true, y_pred, y_proba = evaluate(model, test_loader, criterion, le)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    runtime = round(time.time() - t0, 1)

    print(f"\n  TEST | Acc={te_acc:.4f}  F1={te_f1:.4f}  AUC={te_auc:.4f}  ({runtime}s)")
    print(f"  Best epoch: {best_epoch}")

    all_metrics.append({
        "model": "LSTM", "dataset": ds_name, "split": "test",
        "accuracy": round(te_acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(te_f1, 4), "auc_roc": round(te_auc, 4),
        "runtime_train_s": runtime, "best_epoch": best_epoch, "n_params": total_params,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train", color="#3498db")
    axes[0].plot(history["val_loss"],   label="Val",   color="#e74c3c")
    axes[0].axvline(best_epoch-1, linestyle="--", color="gray", alpha=0.7)
    axes[0].set_title(f"BiLSTM Loss — {ds_name}", fontweight="bold"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc", color="#3498db")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#e74c3c")
    axes[1].plot(history["val_f1"],    label="Val F1",    color="#2ecc71", linestyle="--")
    axes[1].set_title(f"BiLSTM Accuracy/F1 — {ds_name}", fontweight="bold"); axes[1].legend()

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=axes[2], xticks_rotation=45, colorbar=True, cmap="Blues"
    )
    axes[2].set_title(f"BiLSTM Confusion — {ds_name}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"lstm_results_{ds_name}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, f"lstm_results_{ds_name}.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))


def main():
    t_total = time.time()
    print("=" * 65)
    print("Phase 1 — BiLSTM+Attention Training  (seed=42, CUDA AMP)")
    print(f"Config: {CONFIG}")
    print("=" * 65)

    all_metrics = []
    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            alt = ds_path.replace("_with_split", "")
            if os.path.exists(alt):
                ds_path = alt
            else:
                print(f"[SKIP] {ds_path}"); continue
        train_dataset(ds_name, ds_path, all_metrics)

    csv_path = os.path.join(METRICS_DIR, "dl_lstm_performance_2025.csv")
    pd.DataFrame(all_metrics).to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    print(f"\n{'=' * 65}")
    print(f"  DONE — {round(time.time() - t_total, 1)}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
