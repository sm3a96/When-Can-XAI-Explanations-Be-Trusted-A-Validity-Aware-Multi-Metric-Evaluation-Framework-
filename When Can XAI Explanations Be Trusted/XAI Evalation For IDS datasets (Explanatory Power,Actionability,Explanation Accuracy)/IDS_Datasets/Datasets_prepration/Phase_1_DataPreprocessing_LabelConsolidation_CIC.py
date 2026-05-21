"""
Phase 1 — Label Consolidation for CIC IIoT 2025
================================================
Input : IDS_Datasets/Ready Datasets/CIC_IIoT_2025_final.csv  (383,470 × 67, 937 labels)
Output: IDS_Datasets/Ready Datasets/CIC_IIoT_2025_consolidated.csv (383,470 × 68, 8 labels + split col)

Label mapping (attack family extracted from granular label string):
  benign_*              → benign
  attack_recon_*        → recon
  attack_dos_*          → dos
  attack_ddos_*         → ddos
  attack_mitm_*         → mitm
  attack_malware_*      → malware
  attack_web_*          → web
  attack_bruteforce_*   → bruteforce

Random seed: 42 (all splits)
Split: 70% train / 15% val / 15% test (stratified)

Documented for: Section 3 (Datasets) of IEEE TIFS paper
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READY_DIR  = os.path.join(ROOT, "Ready Datasets")
INPUT_CSV  = os.path.join(READY_DIR, "CIC_IIoT_2025_final.csv")
OUTPUT_CSV = os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv")
RESULTS_DIR = os.path.join(ROOT, "..", "IDS_Datasets", "Dataset_Analysis_Results")
PLOTS_DIR   = os.path.join(ROOT, "..", "Models", "Performance_Metrics", "model_comparison_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── label mapping ────────────────────────────────────────────────────────────
def consolidate_label(label: str) -> str:
    if label.startswith("benign"):
        return "benign"
    parts = label.split("_")
    if len(parts) < 2:
        return "unknown"
    family = parts[1].lower()
    mapping = {
        "recon":       "recon",
        "dos":         "dos",
        "ddos":        "ddos",
        "mitm":        "mitm",
        "malware":     "malware",
        "web":         "web",
        "bruteforce":  "bruteforce",
    }
    return mapping.get(family, "unknown")


def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 1 — CIC IIoT 2025 Label Consolidation")
    print("=" * 65)

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    print(f"      Shape: {df.shape}  |  unique labels: {df['label'].nunique()}")

    # ── consolidate ──────────────────────────────────────────────────────────
    print("\n[2/6] Applying label consolidation …")
    df["label_original"] = df["label"].copy()
    df["label"]          = df["label"].apply(consolidate_label)

    dist_before = df["label_original"].value_counts()
    dist_after  = df["label"].value_counts()

    print(f"\n      Before: {len(dist_before)} unique labels")
    print(f"      After : {df['label'].nunique()} consolidated classes")
    print("\n      Consolidated distribution:")
    for cls, cnt in dist_after.items():
        pct = 100 * cnt / len(df)
        print(f"        {cls:15s}: {cnt:7,d}  ({pct:.1f}%)")

    assert "unknown" not in df["label"].values, \
        "ERROR: Some labels did not map — check consolidate_label()"

    # ── stratified train / val / test split ──────────────────────────────────
    print("\n[3/6] Stratified split (70 / 15 / 15) …")
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

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        print(f"      {split_name:5s}: {len(sub):7,d} samples  "
              f"| classes: {sub['label'].value_counts().to_dict()}")

    # ── verify no leakage ─────────────────────────────────────────────────────
    print("\n[4/6] Verifying no leakage …")
    assert len(set(train.index) & set(test.index)) == 0
    assert len(set(train.index) & set(val.index))  == 0
    assert len(set(val.index)   & set(test.index)) == 0
    assert df.isnull().sum().sum() == 0
    print("      ✓ No overlap between splits, zero missing values")

    # ── save consolidated CSV ─────────────────────────────────────────────────
    print(f"\n[5/6] Saving consolidated CSV …")
    # Drop original label column — keep consolidated + split
    df_out = df.drop(columns=["label_original"])
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"      Saved: {OUTPUT_CSV}")
    print(f"      Shape: {df_out.shape}")

    # ── save documentation + figures ─────────────────────────────────────────
    print("\n[6/6] Saving results for paper …")

    # Label distribution CSV (paper Table 1 material)
    stats = {
        "class":   list(dist_after.index),
        "count":   list(dist_after.values),
        "percent": [round(100 * c / len(df), 2) for c in dist_after.values],
    }
    pd.DataFrame(stats).to_csv(
        os.path.join(RESULTS_DIR, "CIC_IIoT_2025_label_distribution.csv"),
        index=False
    )

    # Split distribution CSV (reproducibility)
    split_stats = []
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        for cls, cnt in sub["label"].value_counts().items():
            split_stats.append({"split": sp, "class": cls, "count": cnt,
                                 "pct_within_split": round(100 * cnt / len(sub), 2)})
    pd.DataFrame(split_stats).to_csv(
        os.path.join(RESULTS_DIR, "CIC_IIoT_2025_split_distribution.csv"),
        index=False
    )

    # Figure: consolidated label distribution (paper Figure material)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6",
              "#3498db", "#1abc9c", "#f39c12", "#c0392b"]
    classes = list(dist_after.index)
    counts  = list(dist_after.values)

    axes[0].barh(classes, counts, color=colors[:len(classes)], edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Sample Count", fontsize=12)
    axes[0].set_title("CIC IIoT 2025 — Consolidated Class Distribution", fontsize=13, fontweight="bold")
    for i, (c, v) in enumerate(zip(classes, counts)):
        axes[0].text(v + 500, i, f"{v:,}", va="center", fontsize=9)
    axes[0].invert_yaxis()

    axes[1].pie(counts, labels=classes, autopct="%1.1f%%", colors=colors[:len(classes)],
                startangle=90, pctdistance=0.85)
    axes[1].set_title("Percentage Breakdown", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "CIC_IIoT_2025_label_distribution.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, "CIC_IIoT_2025_label_distribution.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("      Saved: label_distribution.png/.pdf")

    # Summary JSON for Phase 1 log
    summary = {
        "dataset": "CIC_IIoT_2025",
        "input_shape": list(df.shape),
        "output_shape": list(df_out.shape),
        "original_label_count": int(len(dist_before)),
        "consolidated_label_count": int(df["label"].nunique()),
        "class_distribution": {k: int(v) for k, v in dist_after.items()},
        "split_sizes": {sp: int((df["split"] == sp).sum()) for sp in ["train", "val", "test"]},
        "random_seed": RANDOM_SEED,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS_DIR, "CIC_IIoT_2025_consolidation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  DONE — {round(time.time() - t0, 1)}s")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"  Classes: {df['label'].nunique()} | Samples: {len(df):,}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
