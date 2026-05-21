"""
Generate_Explanations.py — Master Explanation Generation Script
===============================================================
Applies all 5 XAI methods to all model × dataset combinations.
Run AFTER all models are trained (Phase 1 complete).

Coverage:
  Classical (DT, LR, RF, XGB) × 2 datasets: SHAP, LIME, Anchors
  DL (Transformer, LSTM) × 2 datasets:       SHAP(kernel), LIME, IG, Attention
  Total: ~40 explanation sets, 1000 instances each

Output:
  explanations/{Method}_{Model}_{Dataset}.pkl
  → dict: {values (n,f), feature_names, label_classes, method, model, dataset,
            n_samples, generation_time_s}

  Models/Performance_Metrics/explanation_timing_2025.csv  (Phase 4.5 input)

Usage: python XAI_Methods/Generate_Explanations.py
       (run from project root)
"""

import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch

# ── project root setup ────────────────────────────────────────────────────────
# This script: <root>/XAI_Methods/Generate_Explanations.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from XAI_Methods.SHAP  import SHAPExplainer
from XAI_Methods.LIME  import LIMEExplainer
from XAI_Methods.IntegratedGradients  import IntegratedGradientsExplainer
from XAI_Methods.AttentionExplanation import AttentionExplainer
from XAI_Methods.XAI_Config import CONFIG

# ── paths ─────────────────────────────────────────────────────────────────────
READY_DIR   = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
CML_DIR     = os.path.join(ROOT, "Models", "Classical_ML")
DL_DIR      = os.path.join(ROOT, "Models", "DeepLearning")
EXPL_DIR    = os.path.join(ROOT, "explanations")
METRICS_DIR = os.path.join(ROOT, "Models", "Performance_Metrics")
os.makedirs(EXPL_DIR, exist_ok=True)

DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_SAMPLES = CONFIG["shap_explain_samples"]   # 1000
SEED      = CONFIG["random_seed"]             # 42

DATASETS = {
    "CIC_IIoT_2025":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
    "IDS2025_Balanced": os.path.join(READY_DIR, "IDS2025_Balanced_final_with_split.csv"),
}
CLASSICAL_MODELS = ["DT", "LR", "RF", "XGB"]
DL_MODELS        = ["Transformer", "LSTM"]


# ── model loading — uses exact attribute names matching saved .pth files ───────

def load_dl_model(pth_path):
    """Load saved DL model using the EXACT class definitions from training scripts."""
    # Import the training script classes directly via importlib (avoids package issues)
    import importlib.util, types

    ckpt  = torch.load(pth_path, map_location=DEVICE, weights_only=False)
    cfg   = ckpt["config"]
    feats = ckpt["feature_names"]
    classes = list(ckpt["label_encoder_classes"])
    n_f, n_c = len(feats), len(classes)

    dl_dir = os.path.join(ROOT, "Models", "DeepLearning")
    if "d_model" in cfg:
        script = os.path.join(dl_dir, "Phase_1_Training_Transformer_2025.py")
        spec   = importlib.util.spec_from_file_location("transformer_module", script)
        mod    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.FeatureTokenizerTransformer(
            n_features=n_f, n_classes=n_c,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"]
        )
    else:
        script = os.path.join(dl_dir, "Phase_1_Training_LSTM_2025.py")
        spec   = importlib.util.spec_from_file_location("lstm_module", script)
        mod    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.LSTMClassifier(
            n_features=n_f, n_classes=n_c,
            hidden_size=cfg["hidden_size"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], bidirectional=cfg.get("bidirectional", True)
        )

    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    return model, feats, classes


# ── helpers ───────────────────────────────────────────────────────────────────

def get_test_split(csv_path, feature_cols):
    df   = pd.read_csv(csv_path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return test[feature_cols], test["label"].values

def get_train_split(csv_path, feature_cols):
    df    = pd.read_csv(csv_path)
    train = df[df["split"] == "train"].reset_index(drop=True)
    return train[feature_cols]

def expl_exists(method, model_name, dataset_name):
    """Check if explanation pkl already exists (skip re-generation)."""
    return os.path.exists(os.path.join(EXPL_DIR, f"{method}_{model_name}_{dataset_name}.pkl"))

def save_expl(values, method, model_name, dataset_name, feature_names, label_classes, elapsed):
    fname = f"{method}_{model_name}_{dataset_name}.pkl"
    with open(os.path.join(EXPL_DIR, fname), "wb") as f:
        pickle.dump({
            "values": values, "feature_names": feature_names,
            "label_classes": label_classes, "method": method,
            "model": model_name, "dataset": dataset_name,
            "n_samples": len(values), "generation_time_s": round(elapsed, 2),
        }, f, protocol=4)
    return fname


# ── classical ML explanations ─────────────────────────────────────────────────

def run_classical(ds_name, csv_path, log_rows):
    print(f"\n{'─'*60}\n  Classical ML — {ds_name}\n{'─'*60}")
    ref = os.path.join(CML_DIR, f"classical_RF_{ds_name}.pkl")
    if not os.path.exists(ref):
        print("  [SKIP] No classical models"); return

    ref_data  = joblib.load(ref)
    feat_cols = ref_data["feature_names"]
    X_test, _ = get_test_split(csv_path, feat_cols)
    X_train   = get_train_split(csv_path, feat_cols)

    for m in CLASSICAL_MODELS:
        pkl = os.path.join(CML_DIR, f"classical_{m}_{ds_name}.pkl")
        if not os.path.exists(pkl): continue
        d       = joblib.load(pkl)
        model   = d["model"]
        le      = d["label_encoder"]
        label_cls = list(le.classes_)
        mtype   = "tree" if m in ("DT","RF","XGB") else "linear"
        print(f"\n  ▶ {m}")

        # SHAP
        if expl_exists("SHAP", m, ds_name):
            print(f"    SHAP: SKIP (already exists)")
        else:
            t0 = time.time()
            shap_exp  = SHAPExplainer(model, X_train, model_type=mtype, random_state=SEED)
            vals      = shap_exp.explain_batch(X_test, n_samples=N_SAMPLES)
            elapsed   = time.time() - t0
            save_expl(vals, "SHAP", m, ds_name, feat_cols, label_cls, elapsed)
            log_rows.append({"method":"SHAP","model":m,"dataset":ds_name,"n_samples":len(vals),
                              "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
            print(f"    SHAP: {elapsed:.1f}s  shape={vals.shape}")

        # LIME
        if expl_exists("LIME", m, ds_name):
            print(f"    LIME: SKIP (already exists)")
        else:
            t0 = time.time()
            lime_exp = LIMEExplainer(model, X_train, random_state=SEED)
            vals     = lime_exp.explain_batch(X_test, n_samples=N_SAMPLES)
            elapsed  = time.time() - t0
            save_expl(vals, "LIME", m, ds_name, feat_cols, label_cls, elapsed)
            log_rows.append({"method":"LIME","model":m,"dataset":ds_name,"n_samples":len(vals),
                              "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
            print(f"    LIME: {elapsed:.1f}s  shape={vals.shape}")

        # Anchors — only DT and XGB (RF/LR too slow: 100-tree predict × many calls = hours)
        # Paper note: timing difference is itself a finding for Phase 4.5 (SOC feasibility)
        if m in ("DT", "XGB"):
            if expl_exists("Anchors", m, ds_name):
                print(f"    Anchors: SKIP (already exists)")
            else:
                try:
                    from XAI_Methods.Anchors import AnchorsExplainer
                    n_anch = 30  # hard limit: 30 samples max to keep under 10 min per model
                    t0   = time.time()
                    ae   = AnchorsExplainer(model, X_train, random_state=SEED)
                    vals = ae.explain_batch(X_test, n_samples=n_anch)
                    elapsed = time.time() - t0
                    save_expl(vals, "Anchors", m, ds_name, feat_cols, label_cls, elapsed)
                    log_rows.append({"method":"Anchors","model":m,"dataset":ds_name,"n_samples":len(vals),
                                      "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
                    print(f"    Anchors: {elapsed:.1f}s  shape={vals.shape}")
                except Exception as e:
                    print(f"    Anchors: SKIP ({e.__class__.__name__})")
        else:
            print(f"    Anchors: SKIP ({m} — too slow for paper timeline; DT/XGB results representative)")


# ── DL explanations ───────────────────────────────────────────────────────────

def run_dl(ds_name, csv_path, log_rows):
    print(f"\n{'─'*60}\n  Deep Learning — {ds_name}\n{'─'*60}")
    arch_map = {"Transformer": "transformer", "LSTM": "lstm"}

    for mdl_key in DL_MODELS:
        pth = os.path.join(DL_DIR, f"{mdl_key.lower()}_{ds_name}.pth")
        if not os.path.exists(pth): print(f"  [SKIP] {mdl_key}.pth not found"); continue

        # Check if ALL 4 DL methods already exist — skip loading model entirely
        all_done = all(expl_exists(m, mdl_key, ds_name)
                       for m in ("SHAP","LIME","IntegratedGradients","Attention"))
        if all_done:
            print(f"\n  ▶ {mdl_key}: ALL SKIP (all 4 explanations exist)")
            continue

        model, feat_cols, label_cls = load_dl_model(pth)
        X_test, _   = get_test_split(csv_path, feat_cols)
        X_train     = get_train_split(csv_path, feat_cols)
        # SHAP on CPU is ~37s/sample — cap at 50 for feasibility (itself a paper finding)
        n_dl_shap   = 50
        print(f"\n  ▶ {mdl_key}  features={len(feat_cols)}")

        # SHAP — KernelExplainer on CPU (DeepExplainer incompatible with BatchNorm+Transformer)
        # Paper note: 37s/sample demonstrates SHAP is NOT suitable for real-time SOC
        if expl_exists("SHAP", mdl_key, ds_name):
            print(f"    SHAP: SKIP (already exists)")
        else:
            t0  = time.time()
            model_cpu = model.cpu()
            se  = SHAPExplainer(model_cpu, X_train.sample(min(100,len(X_train)),random_state=SEED),
                                 model_type="deep", random_state=SEED)
            vals = se.explain_batch(X_test, n_samples=n_dl_shap)
            model.to(DEVICE)
            torch.cuda.empty_cache()  # free fragmented CUDA memory after CPU SHAP
            elapsed = time.time() - t0
            save_expl(vals, "SHAP", mdl_key, ds_name, feat_cols, label_cls, elapsed)
            log_rows.append({"method":"SHAP","model":mdl_key,"dataset":ds_name,"n_samples":len(vals),
                              "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
            print(f"    SHAP: {elapsed:.1f}s  shape={vals.shape}  (CPU KernelExplainer, {n_dl_shap} samples)")

        # LIME — CPU for device consistency
        if expl_exists("LIME", mdl_key, ds_name):
            print(f"    LIME: SKIP (already exists)")
        else:
            t0  = time.time()
            model_cpu = model.cpu()
            le  = LIMEExplainer(model_cpu, X_train, random_state=SEED)
            vals = le.explain_batch(X_test, n_samples=n_dl_shap)
            model.to(DEVICE)
            torch.cuda.empty_cache()  # free fragmented CUDA memory before IG
            elapsed = time.time() - t0
            save_expl(vals, "LIME", mdl_key, ds_name, feat_cols, label_cls, elapsed)
            log_rows.append({"method":"LIME","model":mdl_key,"dataset":ds_name,"n_samples":len(vals),
                              "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
            print(f"    LIME: {elapsed:.1f}s  shape={vals.shape}")

        # Integrated Gradients — GPU (fast, cuDNN LSTM backward fix applied)
        if expl_exists("IntegratedGradients", mdl_key, ds_name):
            print(f"    IG:   SKIP (already exists)")
        else:
            t0  = time.time()
            ig  = IntegratedGradientsExplainer(model, device=DEVICE, n_steps=CONFIG["ig_steps"])
            vals = ig.explain_batch(X_test, n_samples=min(500, N_SAMPLES), batch_size=64)
            elapsed = time.time() - t0
            save_expl(vals, "IntegratedGradients", mdl_key, ds_name, feat_cols, label_cls, elapsed)
            log_rows.append({"method":"IntegratedGradients","model":mdl_key,"dataset":ds_name,"n_samples":len(vals),
                              "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
            print(f"    IG:   {elapsed:.1f}s  shape={vals.shape}")

        # Attention — GPU, zero overhead
        if expl_exists("Attention", mdl_key, ds_name):
            print(f"    Attn: SKIP (already exists)")
        else:
            t0  = time.time()
            ae  = AttentionExplainer(model, arch_map[mdl_key], feat_cols, device=DEVICE)
            vals = ae.explain_batch(X_test, n_samples=N_SAMPLES)
            elapsed = time.time() - t0
            save_expl(vals, "Attention", mdl_key, ds_name, feat_cols, label_cls, elapsed)
            log_rows.append({"method":"Attention","model":mdl_key,"dataset":ds_name,"n_samples":len(vals),
                              "total_seconds":round(elapsed,2),"seconds_per_sample":round(elapsed/max(len(vals),1),4)})
            print(f"    Attn: {elapsed:.1f}s  shape={vals.shape}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"Phase 2 — Generate All Explanations | device={DEVICE} | n={N_SAMPLES}")
    print("=" * 65)

    log_rows = []
    for ds_name, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {csv_path}"); continue
        run_classical(ds_name, csv_path, log_rows)
        run_dl(ds_name, csv_path, log_rows)

    if log_rows:
        timing_df = pd.DataFrame(log_rows)
        timing_csv = os.path.join(METRICS_DIR, "explanation_timing_2025.csv")
        timing_df.to_csv(timing_csv, index=False)
        print(f"\n✓ Timing log: {timing_csv}")
        print("\nMEAN SECONDS PER SAMPLE:")
        print(timing_df.groupby("method")["seconds_per_sample"].mean().sort_values().round(4).to_string())

    files = os.listdir(EXPL_DIR)
    print(f"\n✓ Explanations saved: {len(files)} files in {EXPL_DIR}")
    print(f"\n{'='*65}")
    print(f"  DONE — {round(time.time()-t0,1)}s total")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
