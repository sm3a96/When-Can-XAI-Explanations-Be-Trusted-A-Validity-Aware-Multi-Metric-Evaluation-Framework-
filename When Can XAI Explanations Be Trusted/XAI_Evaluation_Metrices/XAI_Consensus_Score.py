"""
XAI_Consensus_Score.py — Feature Importance Consensus (FIC) Score
=================================================================
NOVEL CONTRIBUTION — Not present in any prior IDS/XAI paper.

Definition:
  For each pair of XAI methods (SHAP, LIME, IG, Anchors, Attention):
    Compute Spearman ρ between their feature importance RANKINGS.
  FIC score for a feature = mean ρ across all method pairs that rank it
  highly (in top-k).

Intuition (for paper):
  "A feature that all five XAI methods unanimously rank as important
   provides stronger evidence for SOC action than a feature flagged
   by only one method. High consensus = high trustworthiness."

Addresses reviewer criticism:
  - R3: "Limited insights beyond SHAP > LIME" → FIC reveals which
    features are universally agreed upon vs. method-specific artifacts

Outputs:
  - Method-pairwise ρ matrix (heatmap data)
  - Per-feature FIC scores (bar chart data)
  - Consensus features list (for paper recommendation)
  - Instance-level consensus (which instances have high/low consensus)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Any, Tuple
from scipy.stats import spearmanr


class FICScoreEvaluator:
    """
    Feature Importance Consensus (FIC) Score calculator.

    Core algorithm:
    1. For each method pair (i, j): compute Spearman ρ over mean |attribution| rankings
    2. FIC_global = mean ρ across all C(n_methods, 2) pairs
    3. Per-feature FIC = fraction of method pairs that include this feature in top-k
    4. High FIC features → recommended for SOC action
    """

    def __init__(self, feature_names: List[str],
                 top_k: int = 10,
                 random_state: int = 42):
        self.feature_names = feature_names
        self.top_k         = top_k
        self.random_state  = random_state

    # ── public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        explanation_dict: Dict[str, np.ndarray],
        dataset_name: str = "",
        model_name: str = "",
    ) -> Dict[str, Any]:
        """
        Compute FIC Score for all XAI methods on one model × dataset combination.

        Parameters
        ----------
        explanation_dict : {'SHAP': (n,f), 'LIME': (n,f), 'IG': (n,f), ...}
                           All arrays must have same shape (n_samples, n_features)

        Returns
        -------
        dict with:
          - pairwise_rho_matrix : pd.DataFrame (heatmap data for paper)
          - per_feature_fic     : pd.DataFrame (FIC score per feature)
          - global_fic          : float (mean ρ across all pairs)
          - consensus_features  : list[str] (top features by FIC score)
          - instance_consensus  : np.ndarray (per-instance mean ρ)
        """
        methods     = list(explanation_dict.keys())
        n_methods   = len(methods)
        n_features  = len(self.feature_names)

        if n_methods < 2:
            raise ValueError("Need at least 2 XAI methods to compute FIC Score")

        # ── mean absolute attribution per feature (global ranking) ────────────
        mean_abs = {}
        for m, vals in explanation_dict.items():
            mean_abs[m] = np.abs(vals).mean(axis=0)  # (n_features,)

        # ── pairwise Spearman ρ (method agreement on feature rankings) ────────
        rho_matrix = pd.DataFrame(
            np.eye(n_methods), index=methods, columns=methods
        )
        all_rhos = []
        for (m1, m2) in combinations(methods, 2):
            rho, _ = spearmanr(mean_abs[m1], mean_abs[m2])
            rho = float(rho) if np.isfinite(rho) else 0.0
            rho_matrix.loc[m1, m2] = rho
            rho_matrix.loc[m2, m1] = rho
            all_rhos.append(rho)

        global_fic = float(np.mean(all_rhos))

        # ── per-feature FIC: fraction of method pairs where both include feature ──
        # A feature is "in top-k" for method m if it's in the top-k by |attribution|
        in_top_k = {}
        for m, imp in mean_abs.items():
            top_k_idx      = set(np.argsort(imp)[::-1][:self.top_k])
            in_top_k[m]    = np.zeros(n_features, dtype=bool)
            in_top_k[m][list(top_k_idx)] = True

        feat_fic_scores = np.zeros(n_features, dtype=float)
        for f_idx in range(n_features):
            n_pairs   = 0
            n_agree   = 0
            for (m1, m2) in combinations(methods, 2):
                n_pairs += 1
                if in_top_k[m1][f_idx] and in_top_k[m2][f_idx]:
                    n_agree += 1
            feat_fic_scores[f_idx] = n_agree / max(n_pairs, 1)

        per_feature_df = pd.DataFrame({
            "feature":   self.feature_names,
            "fic_score": feat_fic_scores,
            "in_top_k_count": [sum(in_top_k[m][i] for m in methods) for i in range(n_features)],
        }).sort_values("fic_score", ascending=False).reset_index(drop=True)

        # ── per-instance consensus ────────────────────────────────────────────
        # For each instance, compute mean ρ across method pairs on instance-level attributions
        n_inst = min(v.shape[0] for v in explanation_dict.values())
        inst_rhos = []
        for i in range(n_inst):
            inst_rhos_i = []
            for (m1, m2) in combinations(methods, 2):
                imp1 = np.abs(explanation_dict[m1][i])
                imp2 = np.abs(explanation_dict[m2][i])
                if imp1.std() > 0 and imp2.std() > 0:
                    rho, _ = spearmanr(imp1, imp2)
                    if np.isfinite(rho):
                        inst_rhos_i.append(float(rho))
            inst_rhos.append(float(np.mean(inst_rhos_i)) if inst_rhos_i else 0.0)

        # ── consensus features (those with FIC > 0.5 = agreed by majority) ───
        consensus_features = per_feature_df[
            per_feature_df["fic_score"] > 0.5
        ]["feature"].tolist()

        return {
            "global_fic":           round(global_fic, 4),
            "pairwise_rho_matrix":  rho_matrix.round(4),
            "per_feature_fic":      per_feature_df,
            "consensus_features":   consensus_features,
            "n_consensus_features": len(consensus_features),
            "instance_consensus":   np.array(inst_rhos),
            "mean_instance_fic":    round(float(np.mean(inst_rhos)), 4),

            # For paper Table 5
            "method_pairs_rho":     {
                f"{m1}_{m2}": round(rho_matrix.loc[m1, m2], 4)
                for (m1, m2) in combinations(methods, 2)
            },
            "model":   model_name,
            "dataset": dataset_name,
        }

    def cross_model_fic(
        self,
        model_explanation_dict: Dict[str, Dict[str, np.ndarray]],
        dataset_name: str = "",
    ) -> pd.DataFrame:
        """
        Compute global FIC for each model and return comparison DataFrame.

        Parameters
        ----------
        model_explanation_dict : {'RF': {'SHAP': ..., 'LIME': ...}, 'XGB': {...}, ...}

        Returns DataFrame for paper: rows=models, column=global_FIC
        """
        rows = []
        for model_name, expl_dict in model_explanation_dict.items():
            result = self.compute(expl_dict, dataset_name=dataset_name,
                                  model_name=model_name)
            row = {"model": model_name, "global_fic": result["global_fic"],
                   "n_consensus_features": result["n_consensus_features"],
                   "mean_instance_fic": result["mean_instance_fic"]}
            row.update(result["method_pairs_rho"])
            rows.append(row)
        return pd.DataFrame(rows)

    def save_for_paper(self, result: Dict[str, Any], output_dir: str) -> None:
        """Save FIC results in paper-ready format (CSV + LaTeX)."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        tag = f"{result.get('model','')}_{result.get('dataset','')}".strip("_")

        # Pairwise ρ heatmap data
        result["pairwise_rho_matrix"].to_csv(
            os.path.join(output_dir, f"FIC_pairwise_rho_{tag}.csv")
        )

        # Per-feature FIC scores
        result["per_feature_fic"].to_csv(
            os.path.join(output_dir, f"FIC_per_feature_{tag}.csv"),
            index=False
        )

        # Summary
        with open(os.path.join(output_dir, f"FIC_summary_{tag}.txt"), "w") as f:
            f.write(f"FIC Score Summary — {tag}\n{'='*50}\n\n")
            f.write(f"Global FIC (mean pairwise ρ):  {result['global_fic']:.4f}\n")
            f.write(f"Mean instance FIC:             {result['mean_instance_fic']:.4f}\n")
            f.write(f"Consensus features (FIC>0.5):  {result['n_consensus_features']}\n\n")
            f.write("Method-pair Spearman ρ:\n")
            for pair, rho in result["method_pairs_rho"].items():
                f.write(f"  {pair}: {rho:.4f}\n")
            f.write(f"\nTop 10 consensus features:\n")
            for _, row in result["per_feature_fic"].head(10).iterrows():
                f.write(f"  {row['feature']}: FIC={row['fic_score']:.3f}  "
                        f"(in top-k for {row['in_top_k_count']} methods)\n")
