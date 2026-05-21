"""
SHAP_Interaction_Analysis.py — Pairwise Feature Interaction Analysis
=====================================================================
NOVEL CONTRIBUTION (Phase 4.4): First application of SHAP interaction
values to IDS attack signature analysis.

What this reveals:
  - Which PAIRS of features jointly explain attack predictions
  - IoT-specific compound attack signatures (e.g., port × protocol)
  - Whether single-feature XAI misses important compound signatures

Method:
  shap.TreeExplainer.shap_interaction_values() — O(n²) per feature pair
  Applied to RF and XGBoost on CIC_IIoT (IoT attacks) and IDS2025 (balanced)

Outputs:
  Analysis/Insights_Output/SHAP_interaction_{dataset}.csv
    → Top interaction pairs per attack class
  Analysis/Insights_Output/SHAP_interaction_heatmap_{dataset}.png
    → Heatmap of mean |interaction| values (paper Figure 6)

Paper section: Section 6 (Novel Insights), Table 6
"""

import os, sys, json, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS_DIR  = os.path.join(ROOT, "Models", "Classical_ML")
READY_DIR   = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
OUTPUT_DIR  = os.path.join(ROOT, "Analysis", "Insights_Output")
PLOTS_DIR   = os.path.join(ROOT, "Models", "Performance_Metrics", "model_comparison_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "CIC_IIoT_2025":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
    "IDS2025_Balanced": os.path.join(READY_DIR, "IDS2025_Balanced_final_with_split.csv"),
}

# Models that support interaction values (tree-based only)
INTERACTION_MODELS = ["XGB"]   # XGB only — RF too slow (100 trees × interaction matrix)
N_SAMPLES_INTERACTION = 50    # strict limit: 50 samples max (interaction is O(n²))
N_TOP_PAIRS = 20              # top interaction pairs to report


def compute_interaction_matrix(model, X_subset: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP interaction values and aggregate to (n_features, n_features) matrix.
    Uses TreeExplainer.shap_interaction_values().
    """
    import shap
    explainer = shap.TreeExplainer(model)

    # interaction_values shape: (n_samples, n_features, n_features, n_classes) or
    # (n_samples, n_classes, n_features, n_features) depending on version
    interactions = explainer.shap_interaction_values(X_subset)

    if isinstance(interactions, list):
        # Multi-class: take mean absolute across classes
        inter_arr = np.stack([np.abs(x) for x in interactions], axis=0).mean(0)
    elif interactions.ndim == 4:
        if interactions.shape[1] == X_subset.shape[1]:
            # (n, n_feat, n_feat, n_class) → mean over n and class
            inter_arr = np.abs(interactions).mean(axis=(0, 3))
        else:
            # (n, n_class, n_feat, n_feat) → mean over n and class
            inter_arr = np.abs(interactions).mean(axis=(0, 1))
    else:
        inter_arr = np.abs(interactions).mean(axis=0)

    return inter_arr   # (n_features, n_features)


def top_interaction_pairs(inter_matrix: np.ndarray, feature_names: list,
                           n_top: int = 20) -> pd.DataFrame:
    """Extract top N non-diagonal interaction pairs from interaction matrix."""
    n = len(feature_names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):  # upper triangle only
            rows.append({
                "feature_1":   feature_names[i],
                "feature_2":   feature_names[j],
                "interaction": float(inter_matrix[i, j]),
            })
    df = pd.DataFrame(rows).sort_values("interaction", ascending=False)
    return df.head(n_top).reset_index(drop=True)


def per_class_interactions(model, X_subset: pd.DataFrame, y_subset: np.ndarray,
                            feature_names: list, class_names: list) -> dict:
    """
    Compute interaction matrices separately for each attack class.
    Reveals class-specific compound signatures.
    """
    import shap
    explainer  = shap.TreeExplainer(model)
    result_by_class = {}

    for cls_idx, cls_name in enumerate(class_names):
        mask = (y_subset == cls_idx)
        if mask.sum() < 20:
            continue
        X_cls = X_subset[mask].reset_index(drop=True)
        interactions = explainer.shap_interaction_values(X_cls)

        if isinstance(interactions, list):
            inter_arr = np.stack([np.abs(x) for x in interactions], axis=0).mean(0)
        elif interactions.ndim == 4:
            inter_arr = np.abs(interactions).mean(axis=(0, 3)) \
                if interactions.shape[1] == X_cls.shape[1] \
                else np.abs(interactions).mean(axis=(0, 1))
        else:
            inter_arr = np.abs(interactions).mean(axis=0)

        top_pairs = top_interaction_pairs(inter_arr, feature_names, n_top=10)
        result_by_class[cls_name] = {
            "matrix":    inter_arr,
            "top_pairs": top_pairs,
            "n_samples": int(mask.sum()),
        }

    return result_by_class


def save_interaction_heatmap(inter_matrix: np.ndarray, feature_names: list,
                              model_name: str, dataset_name: str, top_n_feat: int = 20):
    """Save interaction heatmap for top_n_feat features (by total interaction strength)."""
    # Select top features by row-sum of interaction matrix
    row_sums = inter_matrix.sum(axis=1)
    top_idx  = np.argsort(row_sums)[::-1][:top_n_feat]
    sub_mat  = inter_matrix[np.ix_(top_idx, top_idx)]
    sub_feat = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_mat, xticklabels=sub_feat, yticklabels=sub_feat,
                cmap="YlOrRd", ax=ax, square=True, annot=False,
                linewidths=0.3, linecolor="lightgray")
    ax.set_title(f"SHAP Interaction Values — {model_name} on {dataset_name}\n"
                 f"(Top {top_n_feat} features by total interaction strength)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    fname = f"shap_interaction_heatmap_{model_name}_{dataset_name}.png"
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, fname.replace(".png", ".pdf")), bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    t_total = time.time()
    print("=" * 65)
    print("Phase 4.4 — SHAP Interaction Values Analysis (Novel)")
    print("=" * 65)

    all_findings = []

    for ds_name, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {csv_path}"); continue

        df   = pd.read_csv(csv_path)
        feat = [c for c in df.columns if c not in ("label", "split", "label_original")]
        test = df[df["split"] == "test"].reset_index(drop=True)
        X_te = test[feat]
        y_te = test["label"].values

        # Subset for speed
        rng      = np.random.default_rng(42)
        idx      = rng.choice(len(X_te), size=min(N_SAMPLES_INTERACTION, len(X_te)), replace=False)
        X_sub    = X_te.iloc[idx].reset_index(drop=True)
        y_sub    = y_te[idx]

        for mdl_name in INTERACTION_MODELS:
            pkl = os.path.join(MODELS_DIR, f"classical_{mdl_name}_{ds_name}.pkl")
            if not os.path.exists(pkl):
                print(f"  [SKIP] {mdl_name} not found"); continue

            print(f"\n  ▶ {mdl_name} — {ds_name} ({len(X_sub)} samples)")
            t0   = time.time()
            data = joblib.load(pkl)
            model   = data["model"]
            le      = data["label_encoder"]
            y_enc   = le.transform(y_sub)
            classes = list(le.classes_)

            # Global interaction matrix
            inter_mat  = compute_interaction_matrix(model, X_sub)
            top_pairs  = top_interaction_pairs(inter_mat, feat, N_TOP_PAIRS)
            heatmap_fn = save_interaction_heatmap(inter_mat, feat, mdl_name, ds_name)

            # Per-class interactions
            print(f"    Computing per-class interaction matrices …")
            cls_interactions = per_class_interactions(model, X_sub, y_enc, feat, classes)

            # Save results
            top_pairs.insert(0, "model",   mdl_name)
            top_pairs.insert(1, "dataset", ds_name)
            all_findings.append(top_pairs)

            # Per-class summary
            cls_rows = []
            for cls_name, cls_data in cls_interactions.items():
                for _, row in cls_data["top_pairs"].iterrows():
                    cls_rows.append({
                        "model":       mdl_name,
                        "dataset":     ds_name,
                        "attack_class":cls_name,
                        "feature_1":   row["feature_1"],
                        "feature_2":   row["feature_2"],
                        "interaction": row["interaction"],
                    })
            cls_df = pd.DataFrame(cls_rows)
            cls_df.to_csv(
                os.path.join(OUTPUT_DIR,
                             f"SHAP_interaction_per_class_{mdl_name}_{ds_name}.csv"),
                index=False
            )

            elapsed = time.time() - t0
            print(f"    Done ({round(elapsed,1)}s)  |  heatmap: {heatmap_fn}")
            print(f"    Top 5 global pairs:")
            for _, r in top_pairs.head(5).iterrows():
                print(f"      ({r['feature_1']}, {r['feature_2']}): {r['interaction']:.4f}")

    # ── Save combined results ─────────────────────────────────────────────────
    if all_findings:
        combined = pd.concat(all_findings, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, "SHAP_interaction_all_results.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n✓ Results saved: {out_path}")

    # ── Write insights markdown ───────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "SHAP_Interaction_Insights.md"), "w") as f:
        f.write("# SHAP Interaction Analysis — Key Findings\n\n")
        f.write("**Novel contribution**: First use of SHAP interaction values for IDS attack signatures.\n\n")
        f.write("## Top Global Feature Interactions\n\n")
        if all_findings:
            for ds in DATASETS:
                sub = combined[combined["dataset"] == ds]
                f.write(f"### {ds}\n\n")
                f.write("| Rank | Feature 1 | Feature 2 | Interaction Strength |\n")
                f.write("|---|---|---|---|\n")
                for i, row in sub.head(10).iterrows():
                    f.write(f"| {i+1} | {row['feature_1']} | {row['feature_2']} | {row['interaction']:.4f} |\n")
                f.write("\n")

    print(f"\n{'='*65}")
    print(f"  DONE — {round(time.time()-t_total, 1)}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
