"""
Run_All_Analysis_2025.py — Phase 4 Master Analysis Runner
==========================================================
Runs all 5 analysis scripts sequentially:
  1. Dataset_Comparison_Analysis    — CIC_IIoT vs IDS2025 XAI ranking comparison
  2. DL_vs_Classical_Analysis       — DL vs Classical interpretability agreement
  3. Attack_Type_Analysis           — Per-attack-class XAI profiles
  4. SHAP_Interaction_Analysis      — Novel pairwise feature interactions
  5. Operational_Feasibility        — SOC timing benchmarks + recommendations

Also generates the master comparison visualization for the paper.
Run from project root: python Analysis/Run_All_Analysis_2025.py
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR  = os.path.join(ROOT, "XAI_Evaluation_Metrices", "Results")
INSIGHTS_DIR = os.path.join(ROOT, "Analysis", "Insights_Output")
PLOTS_DIR    = os.path.join(ROOT, "Models", "Performance_Metrics", "model_comparison_plots")
os.makedirs(INSIGHTS_DIR, exist_ok=True)


def run_step(name, script_path, module_name):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
        print(f"  ✓ {name} — {round(time.time()-t0, 1)}s")
        return True
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


def generate_master_comparison_plot():
    """Generate the master summary figure for the paper — all metrics side by side."""
    ep_path  = os.path.join(RESULTS_DIR, "Explanatory_Power_2025.csv")
    act_path = os.path.join(RESULTS_DIR, "Actionability_2025.csv")
    acc_path = os.path.join(RESULTS_DIR, "Explanation_Accuracy_2025.csv")
    fic_path = os.path.join(RESULTS_DIR, "FIC_Scores_2025.csv")

    if not all(os.path.exists(p) for p in [ep_path, act_path, acc_path]):
        print("  [SKIP] Metric CSVs not found — run Phase 3 first")
        return

    ep  = pd.read_csv(ep_path)
    act = pd.read_csv(act_path)
    acc = pd.read_csv(acc_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("XAI Evaluation Results — IEEE TIFS 2025\n"
                 "CIC IIoT 2025 + IDS2025 Balanced Datasets",
                 fontsize=14, fontweight="bold")

    # Panel 1: Explanatory Power (Cohen's d) — NEW metric (not tautological R²)
    if not ep.empty and "cohens_d" in ep.columns:
        pivot = ep.pivot_table(values="cohens_d", index="method", columns="dataset", aggfunc="mean")
        pivot.plot(kind="bar", ax=axes[0,0], colormap="Set2", edgecolor="black", rot=30)
        axes[0,0].axhline(0.5, linestyle="--", color="red", alpha=0.7, label="Practical sig. threshold (d=0.5)")
        axes[0,0].set_title("Explanatory Power\n(Cohen's d vs random baseline)", fontweight="bold")
        axes[0,0].set_ylabel("Cohen's d"); axes[0,0].legend(fontsize=8)
        axes[0,0].set_xlabel("")

    # Panel 2: Actionability — Tier 1 %
    if not act.empty and "pct_tier1_directly_actionable" in act.columns:
        pivot = act.pivot_table(values="pct_tier1_directly_actionable",
                                index="method", columns="dataset", aggfunc="mean")
        pivot.plot(kind="bar", ax=axes[0,1], colormap="Set1", edgecolor="black", rot=30)
        axes[0,1].set_title("Actionability\n(% Tier 1 — Directly Actionable Features)",
                             fontweight="bold")
        axes[0,1].set_ylabel("% Tier 1 Features in Top-10"); axes[0,1].set_xlabel("")

    # Panel 3: Explanation Accuracy (Flip Rate)
    if not acc.empty and "flip_rate" in acc.columns:
        pivot = acc.pivot_table(values="flip_rate", index="method", columns="dataset", aggfunc="mean")
        pivot.plot(kind="bar", ax=axes[1,0], colormap="Set3", edgecolor="black", rot=30)
        axes[1,0].set_title("Explanation Accuracy\n(Flip Rate — Distribution-Preserving Perturbation)",
                             fontweight="bold")
        axes[1,0].set_ylabel("Flip Rate"); axes[1,0].set_xlabel("")

    # Panel 4: FIC Score (novel metric)
    if os.path.exists(fic_path):
        fic = pd.read_csv(fic_path)
        if not fic.empty and "global_fic" in fic.columns:
            pivot = fic.pivot_table(values="global_fic", index="model", columns="dataset", aggfunc="mean")
            pivot.plot(kind="bar", ax=axes[1,1], colormap="Paired", edgecolor="black", rot=30)
            axes[1,1].set_title("FIC Score — Feature Importance Consensus\n(Novel: mean pairwise Spearman ρ across all XAI methods)",
                                 fontweight="bold")
            axes[1,1].set_ylabel("Global FIC Score"); axes[1,1].set_xlabel("")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "master_xai_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Master comparison figure: {out}")


def print_key_findings():
    """Print key quantitative findings for the paper."""
    print(f"\n{'='*65}")
    print("  KEY FINDINGS FOR PAPER")
    print(f"{'='*65}")

    ep_path = os.path.join(RESULTS_DIR, "Explanatory_Power_2025.csv")
    if os.path.exists(ep_path):
        ep = pd.read_csv(ep_path)
        if not ep.empty and "cohens_d" in ep.columns:
            best = ep.groupby("method")["cohens_d"].mean().sort_values(ascending=False)
            print(f"\n  Explanatory Power (Cohen's d) ranking:")
            for m, d in best.items():
                sig = " ← PRACTICALLY SIGNIFICANT" if abs(d) > 0.5 else ""
                print(f"    {m:25s}: d={d:.3f}{sig}")

    act_path = os.path.join(RESULTS_DIR, "Actionability_2025.csv")
    if os.path.exists(act_path):
        act = pd.read_csv(act_path)
        if not act.empty and "mean_actionability" in act.columns:
            best = act.groupby("method")["mean_actionability"].mean().sort_values(ascending=False)
            print(f"\n  Actionability ranking:")
            for m, s in best.items():
                print(f"    {m:25s}: score={s:.3f}")
            # Check if Anchors/LIME > SHAP (expected counter-intuitive finding)
            shap_act  = act[act["method"]=="SHAP"]["mean_actionability"].mean()
            anchors_act = act[act["method"]=="Anchors"]["mean_actionability"].mean() if "Anchors" in act["method"].values else None
            if anchors_act is not None and anchors_act > shap_act:
                print(f"\n  ⭐ COUNTER-INTUITIVE FINDING: Anchors ({anchors_act:.3f}) > SHAP ({shap_act:.3f}) by actionability")
                print(f"     → Most faithful ≠ Most actionable for SOC deployment")

    fic_path = os.path.join(RESULTS_DIR, "FIC_Scores_2025.csv")
    if os.path.exists(fic_path):
        fic = pd.read_csv(fic_path)
        if not fic.empty and "global_fic" in fic.columns:
            print(f"\n  FIC Score (method consensus):")
            for _, row in fic.iterrows():
                print(f"    {row['model']:20s} / {row['dataset']}: FIC={row['global_fic']:.3f}")


def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 4 — Run All Analysis Scripts")
    print("=" * 65)

    analysis_dir = os.path.join(ROOT, "Analysis")

    scripts = [
        ("Dataset Comparison Analysis",  "Dataset_Comparison_Analysis",  "dataset_comp"),
        ("DL vs Classical Analysis",     "DL_vs_Classical_Analysis",     "dl_vs_cl"),
        ("Attack Type Analysis",         "Attack_Type_Analysis",         "attack_type"),
        ("SHAP Interaction Analysis",    "SHAP_Interaction_Analysis",    "shap_inter"),
        ("Operational Feasibility",      "Operational_Feasibility",      "op_feas"),
    ]

    results = {}
    for name, script_name, mod_name in scripts:
        path = os.path.join(analysis_dir, f"{script_name}.py")
        if not os.path.exists(path):
            print(f"\n  [SKIP] {script_name}.py not found"); continue
        results[name] = run_step(name, path, mod_name)

    # Generate master comparison figure
    print(f"\n{'─'*60}\n  Generating master comparison figure\n{'─'*60}")
    generate_master_comparison_plot()

    # Print key findings
    print_key_findings()

    print(f"\n{'='*65}")
    passed  = sum(1 for v in results.values() if v)
    total   = len(results)
    print(f"  Phase 4 COMPLETE — {passed}/{total} scripts succeeded")
    print(f"  Runtime: {round(time.time()-t0, 1)}s")
    print(f"  Insights: {INSIGHTS_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
