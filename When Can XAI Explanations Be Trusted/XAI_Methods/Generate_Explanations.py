"""
Generate_Explanations.py — Master Explanation Generation Script
===============================================================
Applies all 5 XAI methods to all model × dataset combinations.
Saves explanations as .pkl files for use in Phase 3 (metrics).

Coverage:
  Datasets  : CIC_IIoT_2025, IDS2025_Balanced
  Classical : DT, LR, RF, XGB  → SHAP, LIME, Anchors  (3 methods)
  DL        : Transformer, LSTM  → SHAP (kernel), LIME, IG, Attention  (4 methods)
  Total     : 2 datasets × (4 classical × 3 + 2 DL × 4) = 40 explanation sets

Each explanation saved as:
  explanations/{method}_{model}_{dataset}.pkl
  → dict: {'values': np.ndarray (n, f), 'feature_names': list, 'dataset': str,
            'model': str, 'method': str, 'n_samples': int, 'generation_time_s': float}

Timing logged for Phase 4.5 (SOC operational feasibility).
n_samples = 1000 per combination (max — limited by test set size).
"""

import os, sys, json, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from XAI_Methods.SHAP       import SHAPExplainer
from XAI_Methods.LIME       import LIMEExplainer
from XAI_Methods.IntegratedGradients import IntegratedGradientsExplainer
from XAI_Methods.AttentionExplanation import AttentionExplainer
from XAI_Methods.XAI_Config  import CONFIG, DATASET_META, CLASSICAL_MODELS, DL_MODELS

READY_DIR   = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
MODELS_CML  = os.path.join(ROOT, "Models", "Classical_ML")
MODELS_DL   = os.path.join(ROOT, "Models", "DeepLearning")
EXPL_DIR    = os.path.join(ROOT, "explanations")
METRICS_DIR = os.path.join(ROOT, "Models", "Performance_Metrics")
os.makedirs(EXPL_DIR, exist_ok=True)

DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_SAMPLES   = CONFIG["shap_explain_samples"]   # 1000
SEED        = CONFIG["random_seed"]             # 42

DATASETS = {
    "CIC_IIoT_2025":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
    "IDS2025_Balanced": os.path.join(READY_DIR, "IDS2025_Balanced_final_with_split.csv"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_test_split(csv_path: str, feature_cols: list, label_col: str = "label"):
    df   = pd.read_csv(csv_path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    X    = test[feature_cols]
    y    = test[label_col].values
    return X, y


def save_explanation(values: np.ndarray, meta: dict, method: str,
                     model_name: str, dataset_name: str):
    fname = f"{method}_{model_name}_{dataset_name}.pkl"
    path  = os.path.join(EXPL_DIR, fname)
    payload = {
        "values":        values,
        "feature_names": meta["feature_names"],
        "label_classes": meta["label_classes"],
        "dataset":       dataset_name,
        "model":         model_name,
        "method":        method,
        "n_samples":     len(values),
        "generation_time_s": meta["time_s"],
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return path


def timing_log(method: str, model_name: str, dataset_name: str,
               n_samples: int, elapsed_s: float, log_rows: list):
    log_rows.append({
        "method":            method,
        "model":             model_name,
        "dataset":           dataset_name,
        "n_samples":         n_samples,
        "total_seconds":     round(elapsed_s, 2),
        "seconds_per_sample":round(elapsed_s / max(n_samples, 1), 4),
    })


# ── classical ML explanation generation ──────────────────────────────────────

def generate_classical(ds_name: str, csv_path: str, log_rows: list):
    print(f"\n{'─'*60}")
    print(f"  Classical ML — {ds_name}")
    print(f"{'─'*60}")

    # Load one model to get feature names
    ref_pkl  = os.path.join(MODELS_CML, f"classical_RF_{ds_name}.pkl")
    if not os.path.exists(ref_pkl):
        print(f"  [SKIP] No classical models found for {ds_name}")
        return

    ref_data     = joblib.load(ref_pkl)
    feature_cols = ref_data["feature_names"]
    le           = ref_data["label_encoder"]
    X_test, _    = load_test_split(csv_path, feature_cols)
    X_train_csv  = pd.read_csv(csv_path)
    X_train      = X_train_csv[X_train_csv["split"] == "train"][feature_cols].reset_index(drop=True)

    print(f"  Test samples: {len(X_test):,}  |  Features: {len(feature_cols)}")

    for mdl_name in CLASSICAL_MODELS:
        pkl_path = os.path.join(MODELS_CML, f"classical_{mdl_name}_{ds_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [SKIP] {mdl_name} not found")
            continue

        mdl_data = joblib.load(pkl_path)
        model    = mdl_data["model"]
        model_le = mdl_data["label_encoder"]

        print(f"\n  ▶ {mdl_name}")

        # SHAP
        t0 = time.time()
        mtype = "tree" if mdl_name in ("DT", "RF", "XGB") else "linear"
        shap_exp = SHAPExplainer(model, X_train, model_type=mtype, random_state=SEED)
        shap_vals = shap_exp.explain_batch(X_test, n_samples=N_SAMPLES)
        t_shap = time.time() - t0
        meta = {"feature_names": feature_cols, "label_classes": list(model_le.classes_),
                "time_s": round(t_shap, 2)}
        save_explanation(shap_vals, meta, "SHAP", mdl_name, ds_name)
        timing_log("SHAP", mdl_name, ds_name, len(shap_vals), t_shap, log_rows)
        print(f"    SHAP: {round(t_shap,1)}s  |  shape={shap_vals.shape}")

        # LIME
        t0 = time.time()
        lime_exp  = LIMEExplainer(model, X_train, random_state=SEED)
        lime_vals = lime_exp.explain_batch(X_test, n_samples=N_SAMPLES)
        t_lime = time.time() - t0
        meta["time_s"] = round(t_lime, 2)
        save_explanation(lime_vals, meta, "LIME", mdl_name, ds_name)
        timing_log("LIME", mdl_name, ds_name, len(lime_vals), t_lime, log_rows)
        print(f"    LIME: {round(t_lime,1)}s  |  shape={lime_vals.shape}")

        # Anchors (fewer samples — slow method)
        try:
            from XAI_Methods.Anchors import AnchorsExplainer
            anch_samples = min(200, len(X_test))
            t0 = time.time()
            anch_exp  = AnchorsExplainer(model, X_train, random_state=SEED)
            anch_vals = anch_exp.explain_batch(X_test, n_samples=anch_samples)
            t_anch = time.time() - t0
            meta["time_s"] = round(t_anch, 2)
            save_explanation(anch_vals, meta, "Anchors", mdl_name, ds_name)
            timing_log("Anchors", mdl_name, ds_name, len(anch_vals), t_anch, log_rows)
            print(f"    Anchors: {round(t_anch,1)}s  |  shape={anch_vals.shape}")
        except Exception as e:
            print(f"    Anchors: SKIP ({e})")


# ── DL explanation generation ─────────────────────────────────────────────────

def generate_dl(ds_name: str, csv_path: str, log_rows: list):
    print(f"\n{'─'*60}")
    print(f"  Deep Learning — {ds_name}")
    print(f"{'─'*60}")

    dl_arch = {"Transformer": "transformer", "LSTM": "lstm"}

    for model_key in DL_MODELS:
        pth_path = os.path.join(MODELS_DL, f"{model_key.lower()}_{ds_name}.pth")
        if not os.path.exists(pth_path):
            print(f"  [SKIP] {model_key} .pth not found at {pth_path}")
            continue

        ckpt         = torch.load(pth_path, map_location=DEVICE)
        feature_cols = ckpt["feature_names"]
        label_classes = list(ckpt["label_encoder_classes"])
        n_features   = len(feature_cols)
        n_classes    = len(label_classes)
        cfg          = ckpt["config"]

        X_test, _ = load_test_split(csv_path, feature_cols)
        X_train   = pd.read_csv(csv_path)
        X_train   = X_train[X_train["split"] == "train"][feature_cols].reset_index(drop=True)

        print(f"\n  ▶ {model_key}")

        # Load model
        if model_key == "Transformer":
            from Models.DeepLearning.Phase_1_Training_Transformer_2025 import FeatureTokenizerTransformer
            model = FeatureTokenizerTransformer(
                n_features=n_features, n_classes=n_classes,
                d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"], dropout=cfg["dropout"]
            )
        else:
            from Models.DeepLearning.Phase_1_Training_LSTM_2025 import LSTMClassifier
            model = LSTMClassifier(
                n_features=n_features, n_classes=n_classes,
                hidden_size=cfg["hidden_size"], n_layers=cfg["n_layers"],
                dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", True)
            )

        model.load_state_dict(ckpt["model_state"])
        model.eval().to(DEVICE)

        meta = {"feature_names": feature_cols, "label_classes": label_classes}

        # SHAP (KernelExplainer for DL)
        t0 = time.time()
        shap_exp  = SHAPExplainer(model, X_train, model_type="deep", random_state=SEED)
        shap_vals = shap_exp.explain_batch(X_test, n_samples=min(500, N_SAMPLES))
        t_s = time.time() - t0
        meta["time_s"] = round(t_s, 2)
        save_explanation(shap_vals, meta, "SHAP", model_key, ds_name)
        timing_log("SHAP", model_key, ds_name, len(shap_vals), t_s, log_rows)
        print(f"    SHAP: {round(t_s,1)}s  |  shape={shap_vals.shape}")

        # LIME (DL adapter)
        t0 = time.time()
        lime_exp  = LIMEExplainer(model, X_train, random_state=SEED)
        lime_vals = lime_exp.explain_batch(X_test, n_samples=min(500, N_SAMPLES))
        t_l = time.time() - t0
        meta["time_s"] = round(t_l, 2)
        save_explanation(lime_vals, meta, "LIME", model_key, ds_name)
        timing_log("LIME", model_key, ds_name, len(lime_vals), t_l, log_rows)
        print(f"    LIME: {round(t_l,1)}s  |  shape={lime_vals.shape}")

        # Integrated Gradients (Captum)
        t0 = time.time()
        ig_exp   = IntegratedGradientsExplainer(model, device=DEVICE, n_steps=CONFIG["ig_steps"])
        ig_vals  = ig_exp.explain_batch(X_test, n_samples=N_SAMPLES)
        t_ig = time.time() - t0
        meta["time_s"] = round(t_ig, 2)
        save_explanation(ig_vals, meta, "IntegratedGradients", model_key, ds_name)
        timing_log("IntegratedGradients", model_key, ds_name, len(ig_vals), t_ig, log_rows)
        print(f"    IG:   {round(t_ig,1)}s  |  shape={ig_vals.shape}")

        # Attention (native)
        arch = dl_arch[model_key]
        t0 = time.time()
        attn_exp  = AttentionExplainer(model, model_arch=arch,
                                        feature_names=feature_cols, device=DEVICE)
        attn_vals = attn_exp.explain_batch(X_test, n_samples=N_SAMPLES)
        t_at = time.time() - t0
        meta["time_s"] = round(t_at, 2)
        save_explanation(attn_vals, meta, "Attention", model_key, ds_name)
        timing_log("Attention", model_key, ds_name, len(attn_vals), t_at, log_rows)
        print(f"    Attn: {round(t_at,1)}s  |  shape={attn_vals.shape}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t_total  = time.time()
    log_rows = []

    print("=" * 65)
    print("Phase 2 — Generate All Explanations")
    print(f"  n_samples={N_SAMPLES}  |  seed={SEED}  |  device={DEVICE}")
    print("=" * 65)

    for ds_name, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {csv_path} not found"); continue
        generate_classical(ds_name, csv_path, log_rows)
        generate_dl(ds_name, csv_path, log_rows)

    # ── Save timing log (Phase 4.5 input) ─────────────────────────────────────
    if log_rows:
        timing_df = pd.DataFrame(log_rows)
        timing_path = os.path.join(METRICS_DIR, "explanation_timing_2025.csv")
        timing_df.to_csv(timing_path, index=False)
        print(f"\n✓ Timing saved: {timing_path}")
        print("\nTIMING SUMMARY:")
        print(timing_df.groupby("method")[["total_seconds", "seconds_per_sample"]].mean().round(3))

    print(f"\n{'='*65}")
    print(f"  DONE — {round(time.time()-t_total, 1)}s total")
    print(f"  Explanations saved to: {EXPL_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
