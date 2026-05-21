"""
Run_All_Metrics_2025.py — Phase 3 Master Evaluation Runner
===========================================================
Loads all explanation pkl files and computes all 4 metrics:
  1. Explanatory Power (5-fold CV, CI, Cohen's d)
  2. Actionability (3-tier NIST framework)
  3. Explanation Accuracy (KS-validated distribution-preserving perturbation)
  4. FIC Score (Feature Importance Consensus — novel)

Saves all results to XAI_Evaluation_Metrices/Results/ for paper tables.
Run from project root: python XAI_Evaluation_Metrices/Run_All_Metrics_2025.py
"""

import os, sys, pickle, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from XAI_Evaluation_Metrices.Explanatory_Power_2025  import ExplanatoryPowerEvaluator2025
from XAI_Evaluation_Metrices.Actionability_2025      import ActionabilityEvaluator2025
from XAI_Evaluation_Metrices.Explanation_Accuracy_2025 import ExplanationAccuracyEvaluator2025
from XAI_Evaluation_Metrices.XAI_Consensus_Score     import FICScoreEvaluator
from XAI_Evaluation_Metrices.Statistical_Tests_2025  import run_all_tests

EXPL_DIR    = os.path.join(ROOT, "explanations")
RESULTS_DIR = os.path.join(ROOT, "XAI_Evaluation_Metrices", "Results")
READY_DIR   = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
CML_DIR     = os.path.join(ROOT, "Models", "Classical_ML")
DL_DIR      = os.path.join(ROOT, "Models", "DeepLearning")
PLOTS_DIR   = os.path.join(ROOT, "Models", "Performance_Metrics", "model_comparison_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED   = 42
TOP_K  = 10
N_BOOTSTRAP = 2000

DATASETS = {
    "CIC_IIoT_2025":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
    "IDS2025_Balanced": os.path.join(READY_DIR, "IDS2025_Balanced_final_with_split.csv"),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_expl(method, model_name, dataset_name):
    path = os.path.join(EXPL_DIR, f"{method}_{model_name}_{dataset_name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(model_name, dataset_name):
    """Load classical or DL model for metric computation."""
    # Classical ML
    pkl = os.path.join(CML_DIR, f"classical_{model_name}_{dataset_name}.pkl")
    if os.path.exists(pkl):
        d = joblib.load(pkl)
        return d["model"], d["label_encoder"]

    # DL model
    import importlib.util
    pth = os.path.join(DL_DIR, f"{model_name.lower()}_{dataset_name}.pth")
    if not os.path.exists(pth):
        return None, None
    ckpt = torch.load(pth, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    n_f  = len(ckpt["feature_names"])
    n_c  = len(ckpt["label_encoder_classes"])

    if "d_model" in cfg:
        script = os.path.join(DL_DIR, "Phase_1_Training_Transformer_2025.py")
        spec   = importlib.util.spec_from_file_location("t_mod", script)
        mod    = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        model  = mod.FeatureTokenizerTransformer(n_f, n_c, cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["dropout"])
    else:
        script = os.path.join(DL_DIR, "Phase_1_Training_LSTM_2025.py")
        spec   = importlib.util.spec_from_file_location("l_mod", script)
        mod    = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        model  = mod.LSTMClassifier(n_f, n_c, cfg["hidden_size"], cfg["n_layers"], cfg["dropout"], cfg.get("bidirectional", True))

    model.load_state_dict(ckpt["model_state"]); model.eval().to(DEVICE)

    # Wrap DL model to look like sklearn for metric computation
    class DLWrapper:
        def __init__(self, m, le_classes):
            self.m = m
            self.classes_ = le_classes
        def predict(self, X):
            import torch
            t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                return self.m(t).argmax(1).cpu().numpy()
        def predict_proba(self, X):
            import torch
            t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                return torch.softmax(self.m(t), 1).cpu().numpy()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(ckpt["label_encoder_classes"])
    return DLWrapper(model, le.classes_), le


def get_test_data(csv_path, feature_cols):
    df   = pd.read_csv(csv_path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return test[feature_cols], test["label"].values

def get_train_data(csv_path, feature_cols):
    df    = pd.read_csv(csv_path)
    train = df[df["split"] == "train"].reset_index(drop=True)
    return train[feature_cols]


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_explanations():
    """Find all generated pkl files and group by dataset → model → method."""
    groups = {}
    for fname in os.listdir(EXPL_DIR):
        if not fname.endswith(".pkl"):
            continue
        parts = fname[:-4].split("_", 1)
        if len(parts) < 2:
            continue
        method = parts[0]
        rest   = parts[1]
        for ds in ["CIC_IIoT_2025", "IDS2025_Balanced"]:
            if rest.endswith(f"_{ds}"):
                model_name = rest[: -(len(ds) + 1)]
                groups.setdefault(ds, {}).setdefault(model_name, []).append(method)
                break
    return groups


# ── Phase 3 main runner ───────────────────────────────────────────────────────

def run_metrics():
    t0    = time.time()
    print("=" * 65)
    print("Phase 3 — Compute All Evaluation Metrics")
    print("=" * 65)

    groups = discover_explanations()
    if not groups:
        print("\n[ERROR] No explanation files found in", EXPL_DIR)
        print("  Run XAI_Methods/Generate_Explanations.py first.")
        return

    print(f"\n  Found explanations: {sum(len(v) for ds in groups.values() for v in ds.values())} sets")
    for ds, mdls in groups.items():
        for mdl, methods in mdls.items():
            print(f"    {ds} / {mdl}: {methods}")

    ep_rows, act_rows, acc_rows, fic_rows = [], [], [], []

    for ds_name, model_groups in groups.items():
        csv_path = DATASETS.get(ds_name)
        if not csv_path or not os.path.exists(csv_path):
            print(f"\n[SKIP] CSV not found for {ds_name}"); continue

        print(f"\n{'─'*65}")
        print(f"  DATASET: {ds_name}")
        print(f"{'─'*65}")

        for model_name, methods in model_groups.items():
            print(f"\n  ▶ Model: {model_name}  Methods: {methods}")

            # Load model and first explanation to get feature names
            first_expl = load_expl(methods[0], model_name, ds_name)
            if first_expl is None:
                continue
            feature_names = first_expl["feature_names"]
            label_classes = first_expl.get("label_classes", [])

            X_test,  y_test  = get_test_data(csv_path, feature_names)
            X_train          = get_train_data(csv_path, feature_names)
            model, le        = load_model(model_name, ds_name)
            if model is None:
                print(f"    [SKIP] Model not found"); continue

            # Encode y_test
            from sklearn.preprocessing import LabelEncoder
            le_enc = LabelEncoder()
            le_enc.classes_ = np.array(label_classes)
            try:
                y_enc = le_enc.transform(y_test)
            except Exception:
                y_enc = np.zeros(len(y_test), dtype=int)

            # ── FIX #1: aligned instance set ────────────────────────────────
            # Load all explanations for this model × dataset
            expl_dict_full = {}   # original arrays (all samples)
            n_per_method   = {}   # sample count per method before alignment
            for m in methods:
                d = load_expl(m, model_name, ds_name)
                if d is not None:
                    expl_dict_full[m] = d["values"]
                    n_per_method[m]   = len(d["values"])

            # Determine common aligned N — same first N rows for ALL methods
            n_expl = min(len(v) for v in expl_dict_full.values())

            # ALL metrics use this identical slice — guarantees same instances
            expl_dict = {m: v[:n_expl] for m, v in expl_dict_full.items()}
            X_sub     = X_test.iloc[:n_expl].reset_index(drop=True)
            y_sub     = y_enc[:n_expl]

            # Sample adequacy flag — thresholds for paper transparency
            if n_expl >= 500:
                adequacy = "full"
            elif n_expl >= 100:
                adequacy = "limited"
            else:
                adequacy = "minimal"   # results reported but flagged; n<100

            # Shared metadata injected into every metric row
            sample_meta = {
                "n_aligned":        n_expl,
                "sample_adequacy":  adequacy,
                "n_per_method":     str(n_per_method),   # stored as string for CSV
            }
            print(f"    Aligned n={n_expl} [{adequacy}] — {n_per_method}")

            # ── METRIC 1: Explanatory Power ────────────────────────────────
            print("    [EP] Explanatory Power ...", end=" ", flush=True)
            ep_eval = ExplanatoryPowerEvaluator2025(model, n_cv_folds=5,
                                                     n_bootstrap=N_BOOTSTRAP, random_state=SEED)
            for method, vals in expl_dict.items():
                try:
                    res = ep_eval.evaluate(vals, X_sub, method_name=method,
                                           top_k=TOP_K, dataset_name=ds_name)
                    res["model"] = model_name
                    res.update(sample_meta)
                    ep_rows.append(res)
                except Exception as e:
                    print(f"[WARN EP {method}: {e}]", end=" ")
            print("done")

            # ── METRIC 2: Actionability ────────────────────────────────────
            print("    [ACT] Actionability ...", end=" ", flush=True)
            act_eval = ActionabilityEvaluator2025(ds_name, feature_names, random_state=SEED)
            for method, vals in expl_dict.items():
                try:
                    res = act_eval.evaluate(vals, X_sub, method_name=method,
                                            top_k=TOP_K, y_pred=y_sub,
                                            class_names=label_classes)
                    res["model"] = model_name
                    row = {k: v for k, v in res.items() if k != "per_class_actionability"
                           and not isinstance(v, dict)}
                    row.update(sample_meta)
                    act_rows.append(row)
                except Exception as e:
                    print(f"[WARN ACT {method}: {e}]", end=" ")
            print("done")

            # ── METRIC 3: Explanation Accuracy ─────────────────────────────
            print("    [EA] Explanation Accuracy ...", end=" ", flush=True)
            acc_eval = ExplanationAccuracyEvaluator2025(
                model, X_train, strategy="distribution_preserving", random_state=SEED
            )
            for method, vals in expl_dict.items():
                try:
                    res = acc_eval.evaluate(vals, X_sub, method_name=method,
                                            top_k=TOP_K, dataset_name=ds_name,
                                            n_samples=min(300, n_expl))
                    res["model"] = model_name
                    res.update(sample_meta)
                    acc_rows.append(res)
                except Exception as e:
                    print(f"[WARN EA {method}: {e}]", end=" ")
            print("done")

            # ── METRIC 4: FIC Score — FIX: use aligned expl_dict ──────────
            # BUG IN PRIOR VERSION: expl_dict (untruncated) passed to FIC
            # FIX: expl_dict is already truncated to n_expl above
            # → global FIC now computed on same N instances for all methods
            if len(expl_dict) >= 2:
                print("    [FIC] Consensus Score ...", end=" ", flush=True)
                fic_eval = FICScoreEvaluator(feature_names, top_k=TOP_K, random_state=SEED)
                try:
                    fic_res  = fic_eval.compute(expl_dict, dataset_name=ds_name,
                                                 model_name=model_name)
                    fic_eval.save_for_paper(fic_res, RESULTS_DIR)
                    fic_row = {
                        "model": model_name, "dataset": ds_name,
                        "global_fic": fic_res["global_fic"],
                        "n_consensus_features": fic_res["n_consensus_features"],
                        "mean_instance_fic": fic_res["mean_instance_fic"],
                        **fic_res["method_pairs_rho"],
                    }
                    fic_row.update(sample_meta)
                    fic_rows.append(fic_row)
                    print(f"done (global FIC={fic_res['global_fic']:.3f}, n={n_expl})")
                except Exception as e:
                    print(f"[WARN FIC: {e}]")

    # ── Save all results ──────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Saving results for paper …")

    def safe_save(rows, name):
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(RESULTS_DIR, f"{name}.csv")
            df.to_csv(path, index=False)
            print(f"  ✓ {name}.csv  ({len(df)} rows)")
            return df
        else:
            print(f"  ⚠ No rows for {name}")
            return pd.DataFrame()

    ep_df  = safe_save(ep_rows,  "Explanatory_Power_2025")
    act_df = safe_save(act_rows, "Actionability_2025")
    acc_df = safe_save(acc_rows, "Explanation_Accuracy_2025")
    fic_df = safe_save(fic_rows, "FIC_Scores_2025")

    # ── Statistical Tests ─────────────────────────────────────────────────────
    if not ep_df.empty:
        print("\n  Running statistical tests …")
        try:
            stat_path = os.path.join(RESULTS_DIR, "Statistical_Analysis_2025.txt")
            run_all_tests(ep_df, metric_col="mean_xai_power", method_col="method",
                          dataset_col="dataset", model_col="model",
                          output_path=stat_path)
            print(f"  ✓ Statistical_Analysis_2025.txt")
        except Exception as e:
            print(f"  ⚠ Statistical tests failed: {e}")

    print(f"\n{'='*65}")
    print(f"  Phase 3 COMPLETE — {round(time.time()-t0, 1)}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_metrics()
