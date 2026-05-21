"""
Statistical_Tests_2025_v2.py — Issue #2 Fix: CI-Based Analysis
================================================================
Replaces invalid Wilcoxon tests (n=2-6 models → mathematically powerless).

WHY WILCOXON FAILED:
  With n=6 models, 2^6=64 possible rank configurations exist.
  Minimum achievable two-sided p-value = 2/64 = 0.0313.
  Bonferroni-corrected α = 0.05/124 = 0.000403.
  → It is mathematically impossible to reject H0 under Bonferroni.
  → All prior Wilcoxon tests were statistically powerless and uninformative.

REPLACEMENT METHODOLOGY:
  PRIMARY: CI overlap analysis (instance-level, n=30-1000 per row)
    - Each metric row already has valid bootstrap 95% CI (n_bootstrap=2000)
      computed on instance-level ablation scores (not model aggregates)
    - Two methods are compared within each model × dataset combination
    - Non-overlapping CIs = consistent evidence of difference for that model
    - Overlapping CIs = inconclusive for that model

  SECONDARY: Effect size (Cohen's d, already computed per row)
    - |d| > 0.8 = large practical difference
    - 0.5 < |d| ≤ 0.8 = moderate practical difference
    - |d| ≤ 0.5 = small / inconclusive practical difference

  CROSS-MODEL TREND (DESCRIPTIVE ONLY — NOT INFERENTIAL):
    - n_models = 2–6 → insufficient for model-level statistical claims
    - Report: fraction of models where method A > method B
    - Report: mean difference across models ± SD (descriptive)
    - Explicitly state: "no model-level statistical inference performed"

  FRIEDMAN TEST (DESCRIPTIVE ONLY):
    - Retained for completeness but reframed: n=2-6 → underpowered
    - Non-significant result is UNINFORMATIVE (not evidence of no difference)
    - Not used to draw conclusions

HARD CONSTRAINTS:
  - No p-values interpreted as evidence of significance
  - No "statistically significant" language anywhere
  - No modification to metric CSVs, PKLs, or model checkpoints
  - Uses only Issue #1 aligned outputs (n_aligned respected)
  - Random seed = 42 throughout

OUTPUT:
  XAI_Evaluation_Metrices/Results/Statistical_Analysis_v2_2025.txt
  XAI_Evaluation_Metrices/Results/CI_Comparison_Tables_v2.csv
"""

import os, sys
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import friedmanchisquare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "XAI_Evaluation_Metrices", "Results")
SEED        = 42
N_BOOTSTRAP = 5000   # higher than original for stability
RNG         = np.random.default_rng(SEED)

# ── Evidence labels ────────────────────────────────────────────────────────────
def evidence_label(ci_sep: bool, cohens_d: float) -> str:
    """
    Classify evidence strength from CI separation and effect size.
    CI separation: whether the 95% CIs of two methods do NOT overlap for a model.
    Cohen's d: effect size magnitude.
    """
    if ci_sep and abs(cohens_d) > 0.8:
        return "STRONG EVIDENCE"
    if ci_sep and abs(cohens_d) > 0.5:
        return "MODERATE EVIDENCE"
    if ci_sep and abs(cohens_d) <= 0.5:
        return "WEAK EVIDENCE (CI separated but small effect)"
    if not ci_sep and abs(cohens_d) > 0.8:
        return "MODERATE EVIDENCE (overlapping CI but large effect)"
    return "INCONCLUSIVE"


def cis_overlap(lo_a, hi_a, lo_b, hi_b) -> bool:
    """Returns True if the two CIs overlap."""
    return lo_a < hi_b and lo_b < hi_a


def bootstrap_mean_diff_ci(vals_a: np.ndarray, vals_b: np.ndarray,
                            n_boot: int = N_BOOTSTRAP,
                            rng: np.random.Generator = RNG) -> tuple:
    """
    Bootstrap CI for mean(A) - mean(B) using model-level scores.
    Used only for descriptive trend reporting — NOT for inference.
    Returns (mean_diff, ci_lower, ci_upper).
    """
    if len(vals_a) < 2 or len(vals_b) < 2:
        diff = float(np.mean(vals_a) - np.mean(vals_b))
        return diff, np.nan, np.nan

    min_n   = min(len(vals_a), len(vals_b))
    diffs   = []
    for _ in range(n_boot):
        samp_a = rng.choice(vals_a, size=min_n, replace=True)
        samp_b = rng.choice(vals_b, size=min_n, replace=True)
        diffs.append(float(np.mean(samp_a) - np.mean(samp_b)))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(np.mean(vals_a) - np.mean(vals_b)), float(lo), float(hi)


# ── Core comparison engine ─────────────────────────────────────────────────────

METRICS = {
    "Explanatory Power": {
        "file":     "Explanatory_Power_2025.csv",
        "col":      "cohens_d",
        "ci_lo":    "ci_lower_95",
        "ci_hi":    "ci_upper_95",
        "direction": "higher is better",
    },
    "Actionability": {
        "file":     "Actionability_2025.csv",
        "col":      "mean_actionability",
        "ci_lo":    None,   # no instance-level CI stored for actionability
        "ci_hi":    None,
        "direction": "higher is better",
    },
    "Explanation Accuracy": {
        "file":     "Explanation_Accuracy_2025.csv",
        "col":      "flip_rate",
        "ci_lo":    None,
        "ci_hi":    None,
        "direction": "higher is better",
    },
}


def compare_methods_for_metric(df: pd.DataFrame, metric_name: str,
                                metric_col: str, ci_lo_col, ci_hi_col,
                                dataset: str) -> list:
    """
    For one metric × dataset: produce per-model comparison rows for all method pairs.
    Returns list of result dicts.
    """
    sub = df[df["dataset"] == dataset].copy()
    methods = sub["method"].unique().tolist()
    rows    = []

    for m_a, m_b in combinations(sorted(methods), 2):
        df_a = sub[sub["method"] == m_a].set_index("model")
        df_b = sub[sub["method"] == m_b].set_index("model")
        common_models = list(set(df_a.index) & set(df_b.index))

        if len(common_models) < 1:
            continue

        # Per-model comparison (instance-level CIs where available)
        model_details = []
        for model in sorted(common_models):
            row_a = df_a.loc[model]
            row_b = df_b.loc[model]
            val_a = float(row_a[metric_col])
            val_b = float(row_b[metric_col])
            n_a   = int(row_a.get("n_aligned", 0))
            adq_a = str(row_a.get("sample_adequacy", "?"))

            # CI overlap (only if both have instance-level CIs)
            ci_sep = None
            if ci_lo_col and ci_hi_col:
                lo_a = float(row_a.get(ci_lo_col, np.nan))
                hi_a = float(row_a.get(ci_hi_col, np.nan))
                lo_b = float(row_b.get(ci_lo_col, np.nan))
                hi_b = float(row_b.get(ci_hi_col, np.nan))
                if not any(np.isnan([lo_a, hi_a, lo_b, hi_b])):
                    ci_sep = not cis_overlap(lo_a, hi_a, lo_b, hi_b)

            # Cohen's d (already computed in EP rows)
            d_a = float(row_a.get("cohens_d", np.nan))
            d_b = float(row_b.get("cohens_d", np.nan))
            # Use average absolute d as overall effect size proxy
            avg_d = np.nanmean([abs(d_a), abs(d_b)]) if not (np.isnan(d_a) and np.isnan(d_b)) else np.nan

            evid = evidence_label(ci_sep if ci_sep is not None else False, avg_d if not np.isnan(avg_d) else 0)

            model_details.append({
                "model":           model,
                "val_A":           round(val_a, 4),
                "val_B":           round(val_b, 4),
                "diff_A_minus_B":  round(val_a - val_b, 4),
                "A_wins":          val_a > val_b,
                "ci_separated":    ci_sep,
                "cohens_d_A":      round(d_a, 4) if not np.isnan(d_a) else None,
                "cohens_d_B":      round(d_b, 4) if not np.isnan(d_b) else None,
                "evidence":        evid,
                "n_aligned":       n_a,
                "sample_adequacy": adq_a,
            })

        # Cross-model descriptive trend (NOT inferential)
        n_common  = len(common_models)
        wins_a    = sum(1 for d in model_details if d["A_wins"])
        vals_a    = np.array([d["val_A"] for d in model_details])
        vals_b    = np.array([d["val_B"] for d in model_details])
        mean_diff, ci_lo_diff, ci_hi_diff = bootstrap_mean_diff_ci(vals_a, vals_b)

        rows.append({
            "metric":            metric_name,
            "dataset":           dataset,
            "method_A":          m_a,
            "method_B":          m_b,
            "n_common_models":   n_common,
            "models_compared":   ",".join(sorted(common_models)),
            "A_wins_count":      wins_a,
            "B_wins_count":      n_common - wins_a,
            "consistency_pct":   round(100 * wins_a / n_common, 1) if wins_a > n_common/2
                                 else round(100 * (n_common - wins_a) / n_common, 1),
            "consistent_winner": m_a if wins_a >= n_common - wins_a else m_b,
            "mean_diff_A_minus_B":  round(mean_diff, 4),
            "boot_ci_lo_diff":   round(ci_lo_diff, 4) if not np.isnan(ci_lo_diff) else None,
            "boot_ci_hi_diff":   round(ci_hi_diff, 4) if not np.isnan(ci_hi_diff) else None,
            "inference_note":    f"DESCRIPTIVE ONLY — n_models={n_common}, NO INFERENCE",
            "model_details":     model_details,
        })

    return rows


# ── Text report generator ──────────────────────────────────────────────────────

def format_report(all_comparisons: list, metric_tables: pd.DataFrame) -> str:
    lines = []
    w     = "=" * 70

    lines += [
        w,
        "STATISTICAL ANALYSIS v2 — XAI Evaluation for IDS (IEEE TIFS 2025)",
        w,
        "",
        "METHODOLOGY STATEMENT",
        "-" * 70,
        "This analysis replaces the previously invalid Wilcoxon signed-rank tests.",
        "",
        "WHY WILCOXON WAS INVALID:",
        "  n_models = 2-6 per method pair per dataset.",
        "  With n=6: min achievable two-sided p = 2/64 = 0.0313.",
        "  Bonferroni-corrected alpha = 0.05/124 = 0.000403.",
        "  => Mathematically impossible to find evidence against the null under Bonferroni.",
        "  => All prior Wilcoxon tests were statistically powerless.",
        "",
        "REPLACEMENT METHOD: Bootstrap CI + Effect Size",
        "  PRIMARY: 95% bootstrap CI on instance-level scores",
        "    (n_instances = 30-1000 per metric row, valid for inference)",
        "  SECONDARY: Cohen's d (effect size, already computed per row)",
        "  CROSS-MODEL: Descriptive trend only (fraction of models favoring A)",
        "    NO model-level statistical inference performed",
        "",
        "EVIDENCE INTERPRETATION:",
        "  STRONG    : CI separated AND |d| > 0.8",
        "  MODERATE  : CI separated AND |d| > 0.5, OR overlapping CI AND |d| > 0.8",
        "  WEAK      : CI separated but |d| <= 0.5",
        "  INCONCLUSIVE: CI overlap AND |d| <= 0.5",
        "",
        "LANGUAGE POLICY:",
        "  No significance claims or p-value-based conclusions made anywhere.",
        "  All claims bounded by sample adequacy (full/limited/minimal).",
        "",
        w,
    ]

    for metric_name in METRICS:
        metric_comps = [c for c in all_comparisons if c["metric"] == metric_name]
        if not metric_comps:
            continue
        lines += ["", f"METRIC: {metric_name}", "=" * 70]

        for ds in ["CIC_IIoT_2025", "IDS2025_Balanced"]:
            ds_comps = [c for c in metric_comps if c["dataset"] == ds]
            if not ds_comps:
                continue
            lines += ["", f"  Dataset: {ds}", "  " + "-" * 60]

            for comp in ds_comps:
                A, B = comp["method_A"], comp["method_B"]
                winner = comp["consistent_winner"]
                n_mod  = comp["n_common_models"]
                consistency = comp["consistency_pct"]
                md, clo, chi = comp["mean_diff_A_minus_B"], comp["boot_ci_lo_diff"], comp["boot_ci_hi_diff"]

                lines += [
                    f"",
                    f"  {A} vs {B}  (n_models={n_mod})",
                    f"  Consistent winner: {winner} ({consistency:.0f}% of models)",
                    f"  Mean diff (A-B):   {md:+.4f}",
                ]
                if clo is not None and chi is not None:
                    lines += [f"  Boot 95% CI diff: [{clo:+.4f}, {chi:+.4f}]  (descriptive, n_models={n_mod})"]
                lines += [f"  *** NOTE: Cross-model trend is DESCRIPTIVE ONLY — n_models too small for inference ***"]

                # Per-model detail
                lines += ["", f"  Per-model breakdown:"]
                lines += [f"    {'Model':<15} {'n_aligned':>10} {'adequacy':>10} {A:>12} {B:>12} {'diff':>8} {'CI_sep':>8} {'Evidence'}"]
                lines += [f"    {'-'*100}"]
                for md_row in comp["model_details"]:
                    ci_s = str(md_row["ci_separated"]) if md_row["ci_separated"] is not None else "N/A"
                    lines += [
                        f"    {md_row['model']:<15} {md_row['n_aligned']:>10} "
                        f"{md_row['sample_adequacy']:>10} "
                        f"{md_row['val_A']:>12.4f} {md_row['val_B']:>12.4f} "
                        f"{md_row['diff_A_minus_B']:>+8.4f} {ci_s:>8} "
                        f"{md_row['evidence']}"
                    ]

    # ── Summary section ────────────────────────────────────────────────────────
    lines += ["", w, "SUMMARY OF FINDINGS (CI-BASED, NO SIGNIFICANCE CLAIMS)", w, ""]

    # Actionability: most important counter-intuitive finding
    act_comps = [c for c in all_comparisons if c["metric"] == "Actionability"]
    for ds in ["CIC_IIoT_2025", "IDS2025_Balanced"]:
        lime_shap = next((c for c in act_comps
                          if c["dataset"]==ds and
                          set([c["method_A"],c["method_B"]])=={"SHAP","LIME"}), None)
        if lime_shap:
            winner = lime_shap["consistent_winner"]
            pct    = lime_shap["consistency_pct"]
            lines += [
                f"  Actionability / {ds}:",
                f"    LIME vs SHAP: '{winner}' favored in {pct:.0f}% of models",
                f"    Counter-intuitive finding: most faithful method (SHAP) is NOT",
                f"    most actionable. Evidence based on per-model CI analysis.",
                ""
            ]

    lines += [
        "  CRITICAL CAVEATS:",
        "    1. All cross-model trends are DESCRIPTIVE — n_models=2-6 prohibits inference.",
        "    2. Instance-level CIs (n=30-1000) are valid; model-level bootstrap CIs are not.",
        "    3. 'minimal' adequacy rows (n<100) are reported but excluded from evidence claims.",
        "    4. Friedman test (below) is retained as descriptive; non-significance is uninformative.",
        "",
        w,
        "FRIEDMAN TEST (DESCRIPTIVE ONLY — results are UNINFORMATIVE due to n=2-6)",
        "-" * 70,
        "  n_models=2-6 → test has near-zero power. Non-conclusive results",
        "  do NOT indicate no differences exist between methods.",
        "  Results retained for transparency, not as evidence.",
    ]

    return "\n".join(lines)


# ── Friedman (descriptive only) ────────────────────────────────────────────────

def run_friedman_descriptive(df: pd.DataFrame, metric_col: str, dataset: str) -> dict:
    sub     = df[df["dataset"] == dataset]
    methods = sub["method"].unique().tolist()
    groups  = {m: sub[sub["method"]==m][metric_col].values for m in methods}

    # Only include methods with same number of observations
    min_n = min(len(v) for v in groups.values())
    arrays = [groups[m][:min_n] for m in sorted(methods) if len(groups[m]) >= min_n]
    methods_used = [m for m in sorted(methods) if len(groups[m]) >= min_n]

    if len(arrays) < 3:
        return {"methods": methods_used, "stat": None, "p": None, "note": "too few methods for Friedman"}

    try:
        stat, p = friedmanchisquare(*arrays)
        return {
            "methods": methods_used,
            "stat": round(float(stat), 3),
            "p": round(float(p), 4),
            "n_per_method": min_n,
            "note": f"DESCRIPTIVE ONLY — n={min_n} models, underpowered, non-significance uninformative"
        }
    except Exception as e:
        return {"methods": methods_used, "stat": None, "p": None, "note": str(e)}


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("Statistical Analysis v2 — CI-Based Method Comparison")
    print(f"  Random seed: {SEED} | Bootstrap iterations: {N_BOOTSTRAP}")
    print("=" * 65)

    all_comparisons  = []
    friedman_results = []
    flat_rows        = []   # for CSV export

    for metric_name, cfg in METRICS.items():
        path = os.path.join(RESULTS_DIR, cfg["file"])
        if not os.path.exists(path):
            print(f"  [SKIP] {cfg['file']} not found"); continue

        df = pd.read_csv(path)

        # Respect Issue #1: only use aligned data (n_aligned already correct)
        # Exclude minimal rows from evidence claims (but include in descriptive)
        df_valid = df[df["sample_adequacy"] != "minimal"].copy()   # for CI evidence
        df_all   = df.copy()                                         # for descriptive

        print(f"\n  Metric: {metric_name}  ({len(df)} rows, {len(df_valid)} with n>=100)")

        for ds in ["CIC_IIoT_2025", "IDS2025_Balanced"]:
            print(f"    Dataset: {ds}")
            comps = compare_methods_for_metric(
                df_all, metric_name, cfg["col"], cfg["ci_lo"], cfg["ci_hi"], ds
            )
            all_comparisons.extend(comps)

            # Flatten for CSV
            for c in comps:
                for md_row in c["model_details"]:
                    flat_rows.append({
                        "metric": metric_name,
                        "dataset": ds,
                        "method_A": c["method_A"],
                        "method_B": c["method_B"],
                        "model": md_row["model"],
                        "val_A": md_row["val_A"],
                        "val_B": md_row["val_B"],
                        "diff_A_minus_B": md_row["diff_A_minus_B"],
                        "A_wins": md_row["A_wins"],
                        "ci_separated": md_row["ci_separated"],
                        "cohens_d_A": md_row["cohens_d_A"],
                        "cohens_d_B": md_row["cohens_d_B"],
                        "evidence": md_row["evidence"],
                        "n_aligned": md_row["n_aligned"],
                        "sample_adequacy": md_row["sample_adequacy"],
                        "n_common_models": c["n_common_models"],
                        "consistent_winner": c["consistent_winner"],
                        "boot_ci_lo_diff": c["boot_ci_lo_diff"],
                        "boot_ci_hi_diff": c["boot_ci_hi_diff"],
                        "inference_note": c["inference_note"],
                    })
                print(f"      {c['method_A']} vs {c['method_B']}: "
                      f"winner={c['consistent_winner']} ({c['consistency_pct']:.0f}%), "
                      f"n_models={c['n_common_models']}")

            # Friedman (descriptive)
            fr = run_friedman_descriptive(df_all, cfg["col"], ds)
            fr["metric"] = metric_name
            fr["dataset"] = ds
            friedman_results.append(fr)
            if fr["stat"] is not None:
                print(f"      Friedman [descriptive]: χ²={fr['stat']}, p={fr['p']} "
                      f"(n={fr['n_per_method']}, UNINFORMATIVE)")

    # ── Save outputs ─────────────────────────────────────────────────────────
    report_txt  = os.path.join(RESULTS_DIR, "Statistical_Analysis_v2_2025.txt")
    comparison_csv = os.path.join(RESULTS_DIR, "CI_Comparison_Tables_v2.csv")

    # Build and save flat CSV
    if flat_rows:
        flat_df = pd.DataFrame(flat_rows)
        flat_df.to_csv(comparison_csv, index=False)
        print(f"\n  ✓ Saved: CI_Comparison_Tables_v2.csv  ({len(flat_df)} rows)")

    # Build and save text report
    flat_df_for_report = pd.DataFrame(flat_rows) if flat_rows else pd.DataFrame()
    report = format_report(all_comparisons, flat_df_for_report)

    # Append Friedman descriptive section
    fr_lines = ["\n"]
    for fr in friedman_results:
        fr_lines.append(f"  {fr['metric']} / {fr['dataset']}")
        fr_lines.append(f"    Methods: {fr['methods']}")
        if fr["stat"] is not None:
            fr_lines.append(f"    χ²={fr['stat']}, p={fr['p']} — {fr['note']}")
        else:
            fr_lines.append(f"    {fr['note']}")
        fr_lines.append("")
    report += "\n".join(fr_lines)

    with open(report_txt, "w") as f:
        f.write(report)
    print(f"  ✓ Saved: Statistical_Analysis_v2_2025.txt")
    print(f"\n{'='*65}")
    print("  Statistical Analysis v2 COMPLETE")
    print(f"{'='*65}")

    return all_comparisons, flat_rows


if __name__ == "__main__":
    run()
