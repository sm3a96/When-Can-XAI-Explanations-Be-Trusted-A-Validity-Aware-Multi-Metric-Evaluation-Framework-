"""
Phase 1 — Classical ML Training (4 models × 2 datasets)
=========================================================
Models  : Decision Tree (DT), Logistic Regression (LR),
          Random Forest (RF, 100 est), XGBoost (XGB, 100 est)
Datasets: CIC_IIoT_2025_consolidated.csv, IDS2025_Balanced_final.csv
Split   : uses 'split' column (train/val/test)  — seed 42

Outputs (all saved for paper):
  Models/Classical_ML/
    classical_{model}_{dataset}.pkl        ← trained model
  Models/Performance_Metrics/
    classical_performance_2025.csv         ← Table 2 material
    classical_per_class_f1_2025.csv        ← Table 3 material
    model_comparison_plots/
      classical_roc_{dataset}.png          ← Figure material
      classical_confusion_{model}_{dataset}.png
      classical_feature_importance_{model}_{dataset}.png

Random seed: 42 everywhere
"""

import os, json, time, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree       import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble   import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── paths ────────────────────────────────────────────────────────────────────
# Script lives at <project_root>/Models/Classical_ML/ → go up 3 levels for root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
READY_DIR   = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
MODELS_DIR  = os.path.join(ROOT, "Models", "Classical_ML")
METRICS_DIR = os.path.join(ROOT, "Models", "Performance_Metrics")
PLOTS_DIR   = os.path.join(METRICS_DIR, "model_comparison_plots")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASETS = {
    "CIC_IIoT_2025":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
    "IDS2025_Balanced": os.path.join(READY_DIR, "IDS2025_Balanced_final.csv"),
}

MODELS = {
    "DT":  DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=20, min_samples_split=5),
    "LR":  LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, solver="lbfgs",
                               multi_class="auto", C=1.0, n_jobs=-1),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                   n_jobs=-1, min_samples_split=5),
    "XGB": xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                               use_label_encoder=False, eval_metric="mlogloss",
                               n_jobs=-1, tree_method="hist"),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: str, name: str):
    """Load CSV, return X_train, X_val, X_test, y_train, y_val, y_test, le, feature_names."""
    print(f"  Loading {name} from {os.path.basename(path)} …")
    df = pd.read_csv(path)

    # IDS2025_Balanced has no split column → create one
    if "split" not in df.columns:
        print("    No 'split' column found — creating stratified split (70/15/15)")
        train_val, test = train_test_split(
            df, test_size=0.15, random_state=RANDOM_SEED, stratify=df["label"]
        )
        train, val = train_test_split(
            train_val, test_size=0.15 / 0.85,
            random_state=RANDOM_SEED, stratify=train_val["label"]
        )
        df.loc[train.index, "split"] = "train"
        df.loc[val.index,   "split"] = "val"
        df.loc[test.index,  "split"] = "test"
        # Persist the split for reproducibility
        split_path = path.replace(".csv", "_with_split.csv")
        df.to_csv(split_path, index=False)
        print(f"    Split saved → {os.path.basename(split_path)}")

    feature_cols = [c for c in df.columns if c not in ("label", "split", "label_original")]
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"    Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"    Features: {len(feature_cols)}")

    return (
        train_df[feature_cols].values, val_df[feature_cols].values, test_df[feature_cols].values,
        train_df["label_enc"].values,  val_df["label_enc"].values,  test_df["label_enc"].values,
        le, feature_cols, train_df[feature_cols]
    )


def compute_metrics(y_true, y_pred, y_proba, le, model_name, dataset_name, split="test"):
    """Return a metrics dict for one model × dataset × split combination."""
    n_classes = len(le.classes_)
    avg = "binary" if n_classes == 2 else "weighted"

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
    fpr  = 1 - rec  # approximate FPR as 1-recall for multi-class

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    return {
        "model":    model_name,
        "dataset":  dataset_name,
        "split":    split,
        "accuracy": round(acc,  4),
        "precision":round(prec, 4),
        "recall":   round(rec,  4),
        "f1":       round(f1,   4),
        "auc_roc":  round(auc,  4),
        "fpr_approx": round(fpr, 4),
        "n_classes": n_classes,
        "n_test":  int(len(y_true)),
    }


def save_confusion_matrix(y_true, y_pred, le, model_name, dataset_name):
    fig, ax = plt.subplots(figsize=(max(6, len(le.classes_)), max(5, len(le.classes_) - 1)))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name} — {dataset_name}\nConfusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = f"classical_confusion_{model_name}_{dataset_name}.png"
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names, model_name, dataset_name, top_n=20):
    """Save feature importance bar chart for tree-based models."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])

    if importances is None:
        return

    idx = np.argsort(importances)[::-1][:top_n]
    top_feats  = [feature_names[i] for i in idx]
    top_imps   = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_feats[::-1], top_imps[::-1], color="#3498db", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"{model_name} — {dataset_name}\nTop {top_n} Feature Importances", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = f"classical_feat_importance_{model_name}_{dataset_name}.png"
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_per_class_f1(y_true, y_pred, le, model_name, dataset_name, rows):
    """Append per-class F1 rows for Table 3."""
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    for cls_name in le.classes_:
        if cls_name in report:
            rows.append({
                "model":     model_name,
                "dataset":   dataset_name,
                "class":     cls_name,
                "precision": round(report[cls_name]["precision"], 4),
                "recall":    round(report[cls_name]["recall"], 4),
                "f1":        round(report[cls_name]["f1-score"], 4),
                "support":   int(report[cls_name]["support"]),
            })


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=" * 65)
    print("Phase 1 — Classical ML Training  (seed=42)")
    print("=" * 65)

    all_metrics    = []
    all_per_class  = []

    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"\n[SKIP] {ds_path} not found — run label consolidation first")
            continue

        print(f"\n{'─' * 65}")
        print(f"  DATASET: {ds_name}")
        print(f"{'─' * 65}")

        (X_tr, X_va, X_te,
         y_tr, y_va, y_te,
         le, feature_names, X_train_df) = load_dataset(ds_path, ds_name)

        for mdl_name, mdl in MODELS.items():
            t0 = time.time()
            print(f"\n  ▶ {mdl_name} … ", end="", flush=True)

            # ── train ────────────────────────────────────────────────────────
            if mdl_name == "XGB":
                mdl.fit(X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False)
            else:
                mdl.fit(X_tr, y_tr)

            # ── evaluate on val + test ────────────────────────────────────────
            for split_name, X_s, y_s in [("val", X_va, y_va), ("test", X_te, y_te)]:
                y_pred  = mdl.predict(X_s)
                y_proba = mdl.predict_proba(X_s)
                m = compute_metrics(y_s, y_pred, y_proba, le, mdl_name, ds_name, split_name)
                m["runtime_train_s"] = round(time.time() - t0, 1)
                all_metrics.append(m)

            # ── per-class F1 on test ─────────────────────────────────────────
            y_pred_test  = mdl.predict(X_te)
            save_per_class_f1(y_te, y_pred_test, le, mdl_name, ds_name, all_per_class)

            # ── visualizations ────────────────────────────────────────────────
            save_confusion_matrix(y_te, y_pred_test, le, mdl_name, ds_name)
            save_feature_importance(mdl, feature_names, mdl_name, ds_name)

            # ── save model ────────────────────────────────────────────────────
            model_path = os.path.join(MODELS_DIR, f"classical_{mdl_name}_{ds_name}.pkl")
            joblib.dump({"model": mdl, "label_encoder": le,
                         "feature_names": feature_names,
                         "dataset": ds_name, "random_seed": RANDOM_SEED},
                        model_path)

            test_m = [m for m in all_metrics if m["model"]==mdl_name and m["dataset"]==ds_name and m["split"]=="test"][-1]
            print(f"done ({round(time.time()-t0,1)}s) | "
                  f"Acc={test_m['accuracy']:.4f}  F1={test_m['f1']:.4f}  AUC={test_m['auc_roc']:.4f}")

    # ── save all results for paper ────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  Saving paper results …")

    perf_csv = os.path.join(METRICS_DIR, "classical_performance_2025.csv")
    pd.DataFrame(all_metrics).sort_values(["dataset", "model", "split"]).to_csv(perf_csv, index=False)
    print(f"  ✓ {perf_csv}")

    cls_csv = os.path.join(METRICS_DIR, "classical_per_class_f1_2025.csv")
    pd.DataFrame(all_per_class).sort_values(["dataset", "model", "class"]).to_csv(cls_csv, index=False)
    print(f"  ✓ {cls_csv}")

    # ── summary comparison plot ───────────────────────────────────────────────
    df_perf = pd.DataFrame(all_metrics)
    df_test = df_perf[df_perf["split"] == "test"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = ["accuracy", "f1", "auc_roc"]
    titles = ["Accuracy", "Weighted F1", "AUC-ROC"]

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        pivot = df_test.pivot(index="model", columns="dataset", values=metric)
        pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5, rot=0)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, frameon=True)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)

    plt.suptitle("Classical ML — Test Set Performance (2025 Datasets)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "classical_performance_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, "classical_performance_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ classical_performance_comparison.png/.pdf")

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print("  TEST SET RESULTS SUMMARY")
    print(f"{'═' * 65}")
    for ds in DATASETS:
        print(f"\n  {ds}:")
        sub = df_test[df_test["dataset"] == ds][["model", "accuracy", "f1", "auc_roc", "fpr_approx"]]
        print(sub.to_string(index=False))

    print(f"\n{'=' * 65}")
    print(f"  DONE — {round(time.time() - t_total, 1)}s total")
    print(f"  Models saved: {MODELS_DIR}")
    print(f"  Metrics saved: {METRICS_DIR}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
