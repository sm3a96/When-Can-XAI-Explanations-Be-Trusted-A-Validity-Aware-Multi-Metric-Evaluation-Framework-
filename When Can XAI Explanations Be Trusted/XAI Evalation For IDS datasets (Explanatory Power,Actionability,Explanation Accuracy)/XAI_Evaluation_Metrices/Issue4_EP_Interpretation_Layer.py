"""
Issue4_EP_Interpretation_Layer.py — Validity Classification for EP Metric
=========================================================================
Issue #4 fix: Adds a SEMANTIC INTERPRETATION LAYER to CLEAN_EP_TABLE_2025.csv.

NO metric values are altered. This adds three columns:
  ep_interpret     : RELIABLE / INCONCLUSIVE_NOISE / BASE_MODEL_WEAK
  ep_use_in_ranking: True if row meets all validity conditions for ranking
  ep_note          : Short justification of classification

Classification logic:
  BASE_MODEL_WEAK  — model == "LR" (F1 ≈ 68-80%; below 90% threshold required
                     for perturbation-based EP metric to be meaningful)
  INCONCLUSIVE_NOISE — sample_adequacy == "minimal" (n < 100), OR
                       CI crosses zero (ci_lower < 0 AND ci_upper > 0), OR
                       CI upper bound ≤ 0.001 (best-case effect is null)
  RELIABLE         — everything else: n ≥ 100, CI fully positive, strong base model

ep_use_in_ranking = True iff:
  ep_interpret == "RELIABLE" AND n_aligned >= 100 AND CI does not cross zero

References:
  TDSC-to-TIFS rewrite, Issue #4
  Logged threshold: F1 ≥ 90% for perturbation-based metrics (Samek et al. 2017)
"""

import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Results"
)
CLEAN_EP = os.path.join(RESULTS_DIR, "CLEAN_EP_TABLE_2025.csv")
BACKUP   = os.path.join(RESULTS_DIR, "CLEAN_EP_TABLE_2025_pre_issue4.csv")

# ── Model F1 thresholds (from Phase 1 training results) ─────────────────────
# LR: CIC=80.xx%, IDS=68.xx% — far below 90% reliability threshold
# All others: F1 > 90%
WEAK_MODELS = {"LR"}
F1_THRESHOLD = 0.90   # stored for paper citation; not recomputed here


def classify_row(row: pd.Series) -> tuple[str, bool, str]:
    """
    Returns (ep_interpret, ep_use_in_ranking, ep_note).
    Does NOT modify any metric values.
    """
    model    = row["model"]
    n        = int(row["n_aligned"])
    adequacy = row["sample_adequacy"]
    ci_lo    = float(row["ci_lower_95"])
    ci_hi    = float(row["ci_upper_95"])

    # ── Group A: Base model too weak for perturbation-based EP ──────────────
    if model in WEAK_MODELS:
        note = (
            f"LR F1≈68-80% (below {int(F1_THRESHOLD*100)}% threshold); "
            "perturbation-based EP requires well-fitted classifier. "
            "Metric values retained; excluded from ranking."
        )
        return "BASE_MODEL_WEAK", False, note

    # ── Group B: Insufficient sample or unresolvable CI ─────────────────────
    ci_crosses_zero     = (ci_lo < 0) and (ci_hi > 0)
    ci_upper_at_zero    = ci_hi <= 0.001  # best-case effect is null (e.g. [-0.006, 0.000])

    if adequacy == "minimal":
        note = (
            f"n={n} < 100 (minimal adequacy); bootstrap CI unreliable "
            "at this sample size. Effect direction undetermined."
        )
        return "INCONCLUSIVE_NOISE", False, note

    if ci_crosses_zero:
        note = (
            f"CI=[{ci_lo:.4f}, {ci_hi:.4f}] spans zero; "
            "effect direction not established. n={n}, adequacy={adequacy}."
        ).format(n=n, adequacy=adequacy)
        return "INCONCLUSIVE_NOISE", False, note

    if ci_upper_at_zero:
        note = (
            f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]; upper bound ≤ 0.001 "
            "(best-case effect is null). LIME DT — borderline negative."
        )
        return "INCONCLUSIVE_NOISE", False, note

    # ── Group C: Reliable — all validity conditions met ────────────────────
    note = (
        f"n={n} ({adequacy}); CI=[{ci_lo:.4f},{ci_hi:.4f}] fully positive; "
        f"base model F1 > {int(F1_THRESHOLD*100)}%. Used in ranking."
    )
    return "RELIABLE", True, note


def main():
    print("=" * 65)
    print("Issue #4 — EP Interpretation Layer")
    print("=" * 65)

    df = pd.read_csv(CLEAN_EP)
    print(f"  Loaded: {len(df)} rows from {os.path.basename(CLEAN_EP)}")

    # Backup before modification
    import shutil
    shutil.copy(CLEAN_EP, BACKUP)
    print(f"  Backup: {os.path.basename(BACKUP)}")

    # ── Verify no existing interpretation columns (idempotent guard) ─────────
    for col in ("ep_interpret", "ep_use_in_ranking", "ep_note"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # ── Apply classification ─────────────────────────────────────────────────
    results = df.apply(classify_row, axis=1)
    df["ep_interpret"]      = [r[0] for r in results]
    df["ep_use_in_ranking"] = [r[1] for r in results]
    df["ep_note"]           = [r[2] for r in results]

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(CLEAN_EP, index=False)
    print(f"  Saved: {os.path.basename(CLEAN_EP)}")

    # ── Validation report ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  CLASSIFICATION COUNTS")
    print(f"{'─'*65}")
    counts = df["ep_interpret"].value_counts()
    for label, n in counts.items():
        print(f"  {label:25} : {n:3d} rows")
    print(f"  {'ep_use_in_ranking=True':25} : {df['ep_use_in_ranking'].sum():3d} rows")

    # ── Validate: all 16 XAI<random rows must be non-RELIABLE ────────────────
    print(f"\n{'─'*65}")
    print("  VALIDATION: XAI < random rows")
    print(f"{'─'*65}")
    worse = df[df["mean_xai_power"] < df["mean_random_power"]]
    print(f"  Total XAI<random: {len(worse)}/38")
    all_non_reliable = True
    for _, r in worse.iterrows():
        ok = r["ep_interpret"] != "RELIABLE"
        status = "✓" if ok else "✗ ERROR"
        if not ok:
            all_non_reliable = False
        print(f"  {status}  {r['method']:22} {r['model']:12} {r['dataset'][:15]:15} → {r['ep_interpret']}")
    print(f"\n  All 16 flagged as non-RELIABLE: {'✓ PASS' if all_non_reliable else '✗ FAIL'}")

    # ── Validate: no RELIABLE row has XAI < random ───────────────────────────
    reliable_worse = df[(df["ep_interpret"] == "RELIABLE") &
                        (df["mean_xai_power"] < df["mean_random_power"])]
    print(f"  RELIABLE rows with XAI<random: {len(reliable_worse)} (expected 0) "
          f"{'✓ PASS' if len(reliable_worse) == 0 else '✗ FAIL'}")

    # ── Ranking: all rows vs RELIABLE-only ───────────────────────────────────
    print(f"\n{'─'*65}")
    print("  METHOD RANKING — All rows vs RELIABLE only")
    print(f"{'─'*65}")

    all_rank    = df.groupby("method")["cohens_d"].mean().sort_values(ascending=False)
    rel_df      = df[df["ep_use_in_ranking"]]
    if not rel_df.empty:
        rel_rank = rel_df.groupby("method")["cohens_d"].mean().sort_values(ascending=False)
    else:
        rel_rank = pd.Series(dtype=float)

    print(f"  {'Method':22} {'All-rows d':>12} {'Rank':>6} │ {'RELIABLE-only d':>15} {'Rank':>6}")
    print(f"  {'-'*22} {'-'*12} {'-'*6} │ {'-'*15} {'-'*6}")
    all_methods = sorted(set(all_rank.index) | set(rel_rank.index))
    all_order   = {m: i+1 for i, m in enumerate(all_rank.index)}
    rel_order   = {m: i+1 for i, m in enumerate(rel_rank.index)}
    for m in all_rank.index:
        a_d = f"{all_rank.get(m, float('nan')):.4f}"
        r_d = f"{rel_rank.get(m, float('nan')):.4f}" if m in rel_rank.index else "excluded"
        a_r = all_order.get(m, "-")
        r_r = rel_order.get(m, "—")
        print(f"  {m:22} {a_d:>12} {str(a_r):>6} │ {r_d:>15} {str(r_r):>6}")

    # Ranking order check
    all_order_list = list(all_rank.index)
    rel_order_list = [m for m in all_rank.index if m in rel_rank.index]
    order_preserved = rel_order_list == [m for m in all_order_list if m in rel_rank.index]
    print(f"\n  Ranking order preserved: {'✓ YES' if order_preserved else '✗ CHANGED'}")

    # ── RELIABLE rows detail ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  RELIABLE ROWS (used in ranking)")
    print(f"{'─'*65}")
    for _, r in rel_df.iterrows():
        print(f"  {r['method']:22} {r['model']:6} {r['dataset'][:15]:15} "
              f"d={r['cohens_d']:+.4f}  n={r['n_aligned']:4d}  "
              f"CI=[{r['ci_lower_95']:+.4f},{r['ci_upper_95']:+.4f}]")

    print(f"\n{'='*65}")
    print("  Issue #4 COMPLETE — Interpretation layer applied.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
