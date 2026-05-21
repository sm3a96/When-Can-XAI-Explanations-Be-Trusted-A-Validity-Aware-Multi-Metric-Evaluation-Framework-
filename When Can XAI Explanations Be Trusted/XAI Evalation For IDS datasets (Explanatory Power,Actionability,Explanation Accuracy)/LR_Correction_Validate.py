"""
LR_Correction_Validate.py — Post-correction validation and comparison report.
Runs after LR_Correction_Action1.py completes.
Checks: old vs new LR F1, EP/FIC changes, ranking stability, and prints
the full comparison table required by the safety rule.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RES_DIR   = os.path.join(ROOT, "XAI_Evaluation_Metrices", "Results")
BACKUP    = os.path.join(RES_DIR, "LR_correction_backup")

def run():
    print("=" * 70)
    print("LR CORRECTION VALIDATION REPORT")
    print("=" * 70)

    # ── 1. Model performance ──────────────────────────────────────────────────
    print("\n1. MODEL PERFORMANCE")
    print("─" * 70)
    perf = pd.read_csv(os.path.join(ROOT, "Models", "Performance_Metrics", "classical_performance_2025.csv"))
    lr_perf = perf[(perf["model"] == "LR") & (perf["split"] == "test")]
    print("  Old LR performance (from classical_performance_2025.csv):")
    print(lr_perf[["model","dataset","f1","accuracy"]].to_string(index=False))
    # Note: these reflect the old convergence-failed model

    # ── 2. EP before vs after ─────────────────────────────────────────────────
    print("\n2. EP COMPARISON — LR rows only")
    print("─" * 70)

    ep_new = pd.read_csv(os.path.join(RES_DIR, "Explanatory_Power_2025.csv"))
    # Try to load old backup
    ep_old_path = os.path.join(BACKUP, "Explanatory_Power_2025_pre_LR_correction.csv")
    if not os.path.exists(ep_old_path):
        # Find in Results backup dir
        ep_old_path = os.path.join(RES_DIR, "Results_backup_pre_fix1", "Explanatory_Power_2025.csv")

    ep_old_exists = os.path.exists(ep_old_path)
    if ep_old_exists:
        ep_old = pd.read_csv(ep_old_path)
        lr_old = ep_old[ep_old["model"] == "LR"].copy()
        lr_new = ep_new[ep_new["model"] == "LR"].copy()

        print(f"  {'Method':22} {'Dataset':15} {'OLD d':>8} {'NEW d':>8} {'Δd':>8} {'OLD_interp':>18} {'NEW_interp':>18}")
        print(f"  {'-'*22} {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*18} {'-'*18}")

        for _, ro in lr_old.iterrows():
            m, ds = ro["method"], ro["dataset"]
            match = lr_new[(lr_new["method"]==m) & (lr_new["dataset"]==ds)]
            if not match.empty:
                rn = match.iloc[0]
                old_interp = ro.get("ep_interpret", "?")
                new_interp = rn.get("ep_interpret", "?")
                delta = rn["cohens_d"] - ro["cohens_d"]
                flag = " ← CHANGED" if abs(delta) > 0.1 else ""
                print(f"  {m:22} {ds[:15]:15} {ro['cohens_d']:>8.4f} {rn['cohens_d']:>8.4f} {delta:>+8.4f} {old_interp:>18} {new_interp:>18}{flag}")
    else:
        print("  Old EP backup not found — showing new LR EP rows only:")
        lr_new = ep_new[ep_new["model"] == "LR"]
        print(lr_new[["method","dataset","cohens_d","ep_interpret","sample_adequacy"]].to_string())

    # ── 3. FIC before vs after ────────────────────────────────────────────────
    print("\n3. FIC COMPARISON — LR rows")
    print("─" * 70)
    fic_new = pd.read_csv(os.path.join(RES_DIR, "FIC_Scores_2025.csv"))
    fic_old_path = os.path.join(BACKUP, "FIC_Scores_2025_pre_LR_correction.csv")
    if os.path.exists(fic_old_path):
        fic_old = pd.read_csv(fic_old_path)
        for _, ro in fic_old[fic_old["model"] == "LR"].iterrows():
            ds = ro["dataset"]
            match = fic_new[(fic_new["model"] == "LR") & (fic_new["dataset"] == ds)]
            if not match.empty:
                rn = match.iloc[0]
                delta = rn["global_fic"] - ro["global_fic"]
                paradox_old = f"F1~old,FIC={ro['global_fic']:.3f}"
                paradox_new = f"F1~new,FIC={rn['global_fic']:.3f}"
                print(f"  {ds[:20]:20}: OLD={ro['global_fic']:.4f}  NEW={rn['global_fic']:.4f}  Δ={delta:+.4f}")
                print(f"    n_consensus: OLD={ro['n_consensus_features']}  NEW={rn['n_consensus_features']}")
    else:
        print("  FIC backup not found; showing new FIC LR rows:")
        print(fic_new[fic_new["model"]=="LR"][["model","dataset","global_fic","n_consensus_features"]].to_string())

    # ── 4. Cross-method ranking stability ─────────────────────────────────────
    print("\n4. CROSS-METHOD RANKING — RELIABLE rows (must not change)")
    print("─" * 70)
    clean = pd.read_csv(os.path.join(RES_DIR, "CLEAN_EP_TABLE_2025.csv"))
    rel   = clean[clean["ep_use_in_ranking"] == True]
    print(f"  RELIABLE rows: {len(rel)}")
    print(f"  LR in RELIABLE: {len(rel[rel['model']=='LR'])}  (expected 0 — LR is BASE_MODEL_WEAK)")
    print(f"\n  RELIABLE-only ranking (unchanged from pre-correction):")
    rank = rel.groupby("method")["cohens_d"].mean().sort_values(ascending=False)
    for i, (m, d) in enumerate(rank.items(), 1):
        print(f"    {i}. {m:22} mean_d={d:.4f}")

    # ── 5. EA comparison ──────────────────────────────────────────────────────
    print("\n5. EA COMPARISON — LR flip_rate change")
    print("─" * 70)
    acc_new = pd.read_csv(os.path.join(RES_DIR, "Explanation_Accuracy_2025.csv"))
    acc_old_path = os.path.join(BACKUP, "Explanation_Accuracy_2025_pre_LR_correction.csv")
    if os.path.exists(acc_old_path):
        acc_old = pd.read_csv(acc_old_path)
        for _, ro in acc_old[acc_old["model"] == "LR"].iterrows():
            m, ds = ro["method"], ro["dataset"]
            match = acc_new[(acc_new["method"]==m) & (acc_new["model"]=="LR") & (acc_new["dataset"]==ds)]
            if not match.empty:
                rn = match.iloc[0]
                print(f"  {m:22} {ds[:15]:15} flip_rate: OLD={ro['flip_rate']:.4f} → NEW={rn['flip_rate']:.4f}  Δ={rn['flip_rate']-ro['flip_rate']:+.4f}")

    # ── 6. Safety rule: flag any ranking changes ──────────────────────────────
    print("\n6. SAFETY RULE — Cross-method ranking changes (ALL rows)")
    print("─" * 70)
    if ep_old_exists:
        ep_old = pd.read_csv(ep_old_path)
        old_rank = ep_old.groupby("method")["cohens_d"].mean().sort_values(ascending=False)
        new_rank = ep_new.groupby("method")["cohens_d"].mean().sort_values(ascending=False)
        print("  Overall method ranking by mean cohens_d (all 38 rows):")
        print(f"  {'Method':22} {'OLD d':>8} {'OLD rank':>10} {'NEW d':>8} {'NEW rank':>10} {'Rank Δ':>8}")
        old_r = {m: i+1 for i, m in enumerate(old_rank.index)}
        new_r = {m: i+1 for i, m in enumerate(new_rank.index)}
        for m in old_rank.index:
            od = old_rank.get(m, float('nan'))
            nd = new_rank.get(m, float('nan'))
            or_ = old_r.get(m, '-')
            nr_ = new_r.get(m, '-')
            chg = nr_ - or_ if isinstance(nr_, int) and isinstance(or_, int) else 0
            flag = " ← RANK CHANGE" if chg != 0 else ""
            print(f"  {m:22} {od:>8.4f} {str(or_):>10} {nd:>8.4f} {str(nr_):>10} {chg:>+8}{flag}")

    print("\n" + "=" * 70)
    print("  Validation complete.")
    print("=" * 70)

if __name__ == "__main__":
    run()
