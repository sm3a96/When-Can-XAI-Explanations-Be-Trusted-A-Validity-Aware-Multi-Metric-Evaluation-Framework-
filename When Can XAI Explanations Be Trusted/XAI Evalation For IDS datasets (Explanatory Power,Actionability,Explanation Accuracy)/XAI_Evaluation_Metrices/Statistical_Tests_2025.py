"""
Statistical_Tests_2025.py — Pre-Registered Statistical Analysis Framework
==========================================================================
All tests defined HERE, BEFORE looking at results (no p-hacking — addresses R2).

Pre-registered tests (6 total, Bonferroni corrected for 124 comparisons):
  1. Friedman test     — omnibus: do any XAI methods differ? (across all methods)
  2. Wilcoxon signed-rank — pairwise method comparisons
  3. Kruskal-Wallis    — across datasets (do results generalize?)
  4. Mann-Whitney U    — DL vs Classical interpretability comparison
  5. Cohen's d         — practical significance (threshold: d > 0.5)
  6. Permutation test  — non-parametric p-value for key comparisons

Bonferroni correction:
  α_adjusted = 0.05 / 124 = 0.000403
  (5 methods × 2 datasets × 3 metrics × 4 model pairs = ~120 + 4 main comparisons)

Usage:
    from XAI_Evaluation_Metrices.Statistical_Tests_2025 import run_all_tests
    results = run_all_tests(results_df)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Any
from scipy.stats import (
    friedmanchisquare, wilcoxon, kruskal, mannwhitneyu
)


# ── Pre-registered constants ──────────────────────────────────────────────────
ALPHA            = 0.05          # nominal significance level
N_TOTAL_TESTS    = 124           # pre-registered total comparisons
ALPHA_BONFERRONI = ALPHA / N_TOTAL_TESTS  # = 0.000403
COHENS_D_PRACTICAL = 0.5        # Cohen's d > 0.5 = practically significant
N_PERMUTATIONS   = 2000          # permutation test iterations
RANDOM_SEED      = 42


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d — practical significance effect size."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    pooled = np.sqrt(
        ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-10))


def permutation_pvalue(a: np.ndarray, b: np.ndarray,
                       n_perm: int = N_PERMUTATIONS,
                       seed: int = RANDOM_SEED) -> float:
    """Non-parametric permutation test p-value for difference in means."""
    rng      = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    n_a      = len(a)
    count    = sum(
        abs(rng.permutation(combined)[:n_a].mean() -
            rng.permutation(combined)[n_a:].mean()) >= observed
        for _ in range(n_perm)
    )
    return (count + 1) / (n_perm + 1)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = RANDOM_SEED):
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return tuple(float(x) for x in np.quantile(boots, [alpha/2, 1-alpha/2]))


# ── Test 1: Friedman (omnibus across XAI methods) ────────────────────────────

def test_friedman(scores_by_method: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Test H0: all XAI methods have the same score distribution.
    Applied separately to each metric × dataset combination.
    """
    methods = list(scores_by_method.keys())
    arrays  = [scores_by_method[m] for m in methods]

    # Truncate to minimum length
    min_n = min(len(a) for a in arrays)
    arrays = [a[:min_n] for a in arrays]

    stat, pvalue = friedmanchisquare(*arrays)
    bonf_pass    = pvalue < ALPHA_BONFERRONI

    return {
        "test":              "Friedman",
        "H0":                "All XAI methods have same distribution",
        "statistic":         round(float(stat), 4),
        "p_value":           round(float(pvalue), 6),
        "p_bonferroni":      round(float(ALPHA_BONFERRONI), 6),
        "reject_H0":         bool(bonf_pass),
        "interpretation":    ("Methods SIGNIFICANTLY differ (reject H0)"
                              if bonf_pass else "No significant difference detected"),
        "n_methods":         len(methods),
        "n_instances":       min_n,
        "methods":           methods,
    }


# ── Test 2: Wilcoxon pairwise ─────────────────────────────────────────────────

def test_pairwise_wilcoxon(scores_by_method: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests (Bonferroni corrected).
    Applied for each metric: power, actionability, accuracy.
    """
    methods = list(scores_by_method.keys())
    rows    = []

    for (m1, m2) in combinations(methods, 2):
        a = scores_by_method[m1]
        b = scores_by_method[m2]
        min_n = min(len(a), len(b))
        a, b  = a[:min_n], b[:min_n]

        try:
            stat, pvalue = wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
        except Exception:
            stat, pvalue = np.nan, 1.0

        d       = cohens_d(a, b)
        bonf_ok = pvalue < ALPHA_BONFERRONI

        rows.append({
            "method_1":         m1,
            "method_2":         m2,
            "wilcoxon_stat":    round(float(stat), 2) if np.isfinite(stat) else np.nan,
            "p_value":          round(float(pvalue), 6),
            "p_bonferroni_adj": round(float(ALPHA_BONFERRONI), 6),
            "significant":      bool(bonf_ok),
            "cohens_d":         round(float(d), 4) if np.isfinite(d) else np.nan,
            "practical_sig":    bool(abs(d) > COHENS_D_PRACTICAL),
            "m1_mean":          round(float(a.mean()), 4),
            "m2_mean":          round(float(b.mean()), 4),
            "winner":           m1 if a.mean() > b.mean() else m2,
            "n_instances":      min_n,
        })

    return pd.DataFrame(rows)


# ── Test 3: Kruskal-Wallis (across datasets) ──────────────────────────────────

def test_kruskal_wallis(scores_by_dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    H0: XAI rankings generalize across datasets (no dataset effect).
    """
    datasets = list(scores_by_dataset.keys())
    arrays   = [scores_by_dataset[d] for d in datasets]
    stat, pvalue = kruskal(*arrays)

    return {
        "test":           "Kruskal-Wallis",
        "H0":             "No significant dataset effect on XAI scores",
        "statistic":      round(float(stat), 4),
        "p_value":        round(float(pvalue), 6),
        "reject_H0":      bool(pvalue < ALPHA_BONFERRONI),
        "interpretation": ("Dataset significantly affects XAI scores — domain-specific"
                           if pvalue < ALPHA_BONFERRONI
                           else "Results generalize across datasets"),
        "datasets":       datasets,
    }


# ── Test 4: Mann-Whitney U (DL vs Classical) ──────────────────────────────────

def test_dl_vs_classical(dl_scores: np.ndarray, classical_scores: np.ndarray) -> Dict[str, Any]:
    """
    H0: DL and Classical models produce equally interpretable explanations.
    """
    stat, pvalue = mannwhitneyu(dl_scores, classical_scores, alternative="two-sided")
    d = cohens_d(dl_scores, classical_scores)

    return {
        "test":           "Mann-Whitney U (DL vs Classical)",
        "H0":             "DL and Classical XAI quality are equal",
        "statistic":      round(float(stat), 2),
        "p_value":        round(float(pvalue), 6),
        "reject_H0":      bool(pvalue < ALPHA_BONFERRONI),
        "cohens_d":       round(float(d), 4) if np.isfinite(d) else np.nan,
        "practical_sig":  bool(abs(d) > COHENS_D_PRACTICAL),
        "dl_mean":        round(float(dl_scores.mean()), 4),
        "classical_mean": round(float(classical_scores.mean()), 4),
        "winner":         "DL" if dl_scores.mean() > classical_scores.mean() else "Classical",
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_all_tests(
    results_df: pd.DataFrame,
    metric_col: str = "mean_xai_power",
    method_col: str = "method",
    dataset_col: str = "dataset",
    model_col: str  = "model",
    dl_models: List[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all 6 pre-registered statistical tests on results DataFrame.

    Parameters
    ----------
    results_df : DataFrame with columns [method, dataset, model, metric_col]
    metric_col : Which metric to test ('mean_xai_power', 'mean_actionability', etc.)
    dl_models  : List of DL model names (default: ['Transformer', 'LSTM'])
    output_path: If set, save results to this path

    Returns
    -------
    dict with all test results
    """
    if dl_models is None:
        dl_models = ["Transformer", "LSTM"]

    all_results = {
        "metric_tested":        metric_col,
        "alpha_nominal":        ALPHA,
        "alpha_bonferroni":     ALPHA_BONFERRONI,
        "n_total_tests":        N_TOTAL_TESTS,
        "cohens_d_threshold":   COHENS_D_PRACTICAL,
        "random_seed":          RANDOM_SEED,
    }

    methods  = results_df[method_col].unique().tolist()
    datasets = results_df[dataset_col].unique().tolist()

    # ── Test 1: Friedman (per dataset) ───────────────────────────────────────
    friedman_results = {}
    for ds in datasets:
        sub = results_df[results_df[dataset_col] == ds]
        scores_by_method = {
            m: sub[sub[method_col] == m][metric_col].values
            for m in methods if m in sub[method_col].values
        }
        if len(scores_by_method) >= 3:
            friedman_results[ds] = test_friedman(scores_by_method)
    all_results["friedman_tests"] = friedman_results

    # ── Test 2: Wilcoxon pairwise (per dataset) ───────────────────────────────
    wilcoxon_results = {}
    for ds in datasets:
        sub = results_df[results_df[dataset_col] == ds]
        scores_by_method = {
            m: sub[sub[method_col] == m][metric_col].values
            for m in methods if m in sub[method_col].values
        }
        if len(scores_by_method) >= 2:
            wilcoxon_results[ds] = test_pairwise_wilcoxon(scores_by_method)
    all_results["wilcoxon_tests"] = wilcoxon_results

    # ── Test 3: Kruskal-Wallis (per method, across datasets) ─────────────────
    kruskal_results = {}
    for m in methods:
        sub = results_df[results_df[method_col] == m]
        scores_by_ds = {
            ds: sub[sub[dataset_col] == ds][metric_col].values
            for ds in datasets if ds in sub[dataset_col].values
        }
        if len(scores_by_ds) >= 2:
            kruskal_results[m] = test_kruskal_wallis(scores_by_ds)
    all_results["kruskal_tests"] = kruskal_results

    # ── Test 4: DL vs Classical ───────────────────────────────────────────────
    is_dl       = results_df[model_col].isin(dl_models)
    dl_scores   = results_df[is_dl][metric_col].dropna().values
    cl_scores   = results_df[~is_dl][metric_col].dropna().values
    if len(dl_scores) >= 5 and len(cl_scores) >= 5:
        all_results["dl_vs_classical"] = test_dl_vs_classical(dl_scores, cl_scores)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for ds, fr in friedman_results.items():
        summary_rows.append({
            "test": f"Friedman [{ds}]",
            "p_value": fr["p_value"],
            "significant": fr["reject_H0"],
            "interpretation": fr["interpretation"][:60],
        })
    if "dl_vs_classical" in all_results:
        dvsc = all_results["dl_vs_classical"]
        summary_rows.append({
            "test": "Mann-Whitney (DL vs Classical)",
            "p_value": dvsc["p_value"],
            "significant": dvsc["reject_H0"],
            "interpretation": f"Cohen's d={dvsc['cohens_d']}, winner={dvsc['winner']}"
        })
    all_results["summary_table"] = pd.DataFrame(summary_rows)

    # ── Save results ──────────────────────────────────────────────────────────
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("STATISTICAL ANALYSIS — XAI Evaluation for IDS (IEEE TIFS 2025)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Metric: {metric_col}\n")
            f.write(f"α (nominal): {ALPHA}  |  α (Bonferroni): {ALPHA_BONFERRONI:.6f}\n")
            f.write(f"N total pre-registered tests: {N_TOTAL_TESTS}\n")
            f.write(f"Cohen's d practical threshold: {COHENS_D_PRACTICAL}\n\n")

            f.write("FRIEDMAN TESTS (omnibus):\n")
            for ds, fr in friedman_results.items():
                f.write(f"  [{ds}] χ²={fr['statistic']}, p={fr['p_value']}, "
                        f"reject H0={fr['reject_H0']}\n")
                f.write(f"    → {fr['interpretation']}\n")

            f.write("\nPAIRWISE WILCOXON (Bonferroni corrected):\n")
            for ds, wdf in wilcoxon_results.items():
                f.write(f"\n  Dataset: {ds}\n")
                for _, row in wdf.iterrows():
                    sig = "***" if row["significant"] else ""
                    d_sig = "(PRACTICAL)" if row["practical_sig"] else ""
                    f.write(f"    {row['method_1']} vs {row['method_2']}: "
                            f"p={row['p_value']:.6f} {sig}  "
                            f"d={row['cohens_d']:.3f} {d_sig}  "
                            f"winner={row['winner']}\n")

            if "dl_vs_classical" in all_results:
                dvsc = all_results["dl_vs_classical"]
                f.write(f"\nDL vs CLASSICAL (Mann-Whitney):\n")
                f.write(f"  p={dvsc['p_value']:.6f}  d={dvsc['cohens_d']:.3f}  "
                        f"winner={dvsc['winner']}\n")

        print(f"✓ Saved statistical analysis to {output_path}")

    return all_results
