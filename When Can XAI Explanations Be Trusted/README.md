# When Can XAI Explanations Be Trusted?
## A Validity-Aware Multi-Metric Evaluation Framework for Network Intrusion Detection Systems

> **IEEE Transactions on Information Forensics and Security (IEEE TIFS) — Under Review**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Ismail Bibers¹, Issa Khalil², Mustafa Abdallah¹  
¹ Purdue University, West Lafayette / Indianapolis, IN, USA  
² Hamad Bin Khalifa University, Doha, Qatar

---

## Overview

Explainable AI (XAI) methods are increasingly deployed in network intrusion detection systems (IDS), yet a critical question has received little systematic attention: **under what conditions do XAI evaluation metrics yield valid, statistically reliable conclusions?**

This repository provides the full implementation of a **four-dimensional, validity-aware XAI evaluation framework** applied to 5 XAI methods, 6 classifier architectures, and 475,210 network flow records from two 2025 IDS benchmark datasets.

---

## Framework Overview

![Four-Dimensional XAI Evaluation Framework](Paper/Figures/framework_fig2.png)

*The proposed pipeline: raw network traffic feeds six classifiers; five XAI methods generate feature attributions; a validity classification layer screens each configuration before computing four complementary metrics.*

---

## Key Contributions

| # | Contribution | Key Finding |
|---|---|---|
| 1 | **Validity Classification Layer** | 84% of standard evaluation configurations fail at least one reliability precondition |
| 2 | **Explanatory Power (EP)** | SHAP leads at *k* ≥ 10 (Cohen's *d* = 3.337); LIME leads at *k* = 5 (*d* = 1.199) |
| 3 | **NIST-Grounded Actionability (ACT)** | First application of NIST SP 800-94 to XAI evaluation for IDS |
| 4 | **Feature Importance Consensus (FIC)** | RF achieves validated consensus (FIC = 0.52–0.56, *ρ* > 0.91 vs MDI); BiLSTM shows near-zero consensus (FIC = 0.061) |
| 5 | **SHAP Interaction Analysis** | Identifies compound attack signatures invisible to single-feature methods |
| 6 | **Computational Feasibility** | DL-SHAP requires 18–30 s/sample (infeasible); TreeSHAP ≤ 0.036 s and IG ≤ 0.001 s are real-time viable |

---

## Four Evaluation Dimensions

### 1. Explanatory Power (EP)
Measures whether XAI-ranked features cause **greater model-confidence degradation** (under training-mean replacement) than randomly ranked features. Uses Cohen's *d* on per-instance confidence-drop scores and 95% bootstrap confidence intervals.

### 2. Actionability (ACT)
Measures whether attributed features translate into **operationally deployable Snort/Suricata rules**, using a NIST SP 800-94 three-tier taxonomy:
- **Tier 1** (weight 1.0): Port, protocol, TCP flag — direct rule match
- **Tier 2** (weight 0.6): Rate, size, inter-arrival time — threshold-based rule
- **Tier 3** (weight 0.0): Statistical properties — not directly configurable

### 3. Explanation Accuracy (EA)
Evaluates prediction-flip faithfulness using **distribution-preserving perturbation** — features are replaced by values sampled from their real marginal distributions (KS-validated, 97.6% pass rate) — and Deletion-AUC via the trapezoidal rule.

### 4. Feature Importance Consensus (FIC) *(Novel)*
A novel metric computing the **mean pairwise Spearman rank correlation** across XAI attribution vectors. Externally validated against RF Mean Decrease Impurity (*ρ* > 0.91, *z* > 3.8 above random).

---

## Validity Classification Layer

Each configuration is classified before reporting any metric:

| Class | Preconditions | Meaning |
|---|---|---|
| **Reliable (R)** | Macro F1 ≥ 90%, *n* ≥ 100, CI lower bound > 0 | Results are trustworthy |
| **Inconclusive (I)** | *n* < 100 or CI crosses zero | Directional estimate only |
| **Base-Model-Weak (W)** | Macro F1 < 90% | Metric measures model fragility, not explanation quality |

> **84% of 38 standard configurations fail at least one precondition.** Only Reliable configurations contribute to cross-method rankings.

---

## Results at a Glance

| Metric | Best Method | Key Value |
|---|---|---|
| EP (*k* = 10, RF/CIC) | SHAP | Cohen's *d* = 3.337 |
| EP (*k* = 5, RF/CIC) | LIME | Cohen's *d* = 1.199 |
| ACT (both datasets) | LIME | Only method above random baseline on both |
| EA flip rate (RF/CIC) | SHAP | 0.821 (6.9× random baseline) |
| FIC (RF/CIC) | RF+SHAP/LIME | 0.555, *ρ*_MDI = 0.952 |
| Timing — real-time viable | TreeSHAP (XGB) | < 0.001 s/sample |
| Timing — infeasible | DL-SHAP (Transformer) | 30.09 s/sample |

---

## Datasets

| Dataset | Records | Features | Classes | Environment |
|---|---|---|---|---|
| [CIC-IIoT-2025](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html) | 383,470 | 68 (preprocessed) | 8 (1 benign + 7 attacks) | IoT |
| [IDS2025-Balanced](https://data.mendeley.com/datasets/pkskt3fv3v/1) | 91,740 | 70 (preprocessed) | 7 (balanced) | General network |

---

## Models

| Model | Type | XAI Support |
|---|---|---|
| Logistic Regression | Classical ML | SHAP LinearExplainer, LIME, Anchors |
| Decision Tree | Classical ML | SHAP TreeExplainer, LIME, Anchors |
| Random Forest | Classical ML | SHAP TreeExplainer, LIME, Anchors |
| XGBoost | Classical ML | SHAP TreeExplainer, LIME, Anchors |
| FT-Transformer | Deep Learning | SHAP KernelExplainer, LIME, IG, Attention |
| BiLSTM | Deep Learning | SHAP KernelExplainer, LIME, IG, Attention |

---

## Repository Structure

```
├── Paper/
│   ├── main.tex                    # IEEE TIFS manuscript
│   ├── references.bib              # Bibliography
│   ├── Cover_Letter.tex            # Submission cover letter
│   └── Figures/                    # All paper figures (PDF + PNG)
│
├── XAI Evalation For IDS datasets/
│   ├── Models/
│   │   ├── Classical_ML/           # LR, DT, RF, XGBoost training
│   │   ├── DeepLearning/           # Transformer, BiLSTM training
│   │   └── model_definitions.py    # Model architectures
│   │
│   ├── XAI_Methods/
│   │   ├── SHAP.py                 # TreeExplainer / LinearExplainer / KernelExplainer
│   │   ├── LIME.py                 # Local surrogate with distribution-aware kernel
│   │   ├── Anchors.py              # Rule-based explanations
│   │   ├── IntegratedGradients.py  # Captum-based IG with zero-vector baseline
│   │   ├── AttentionExplanation.py # Soft attention weights
│   │   ├── Generate_Explanations.py # Unified attribution format
│   │   └── XAI_Config.py           # NIST tier assignments for both datasets
│   │
│   ├── XAI_Evaluation_Metrices/
│   │   ├── Explanatory_Power_2025.py   # EP: confidence-drop + Cohen's d + bootstrap CI
│   │   ├── Actionability_2025.py       # ACT: NIST 3-tier taxonomy
│   │   ├── Explanation_Accuracy_2025.py # EA: distribution-preserving perturbation + D-AUC
│   │   ├── XAI_Consensus_Score.py      # FIC: pairwise Spearman + external MDI validation
│   │   ├── Statistical_Tests_2025.py   # Friedman / Wilcoxon tests
│   │   └── Run_All_Metrics_2025.py     # End-to-end evaluation runner
│   │
│   ├── Analysis/
│   │   ├── SHAP_Interaction_Analysis.py
│   │   ├── Operational_Feasibility.py
│   │   └── Run_All_Analysis_2025.py
│   │
│   └── IDS_Datasets/               # Dataset loading and preprocessing notebooks
│
├── XAI_Evaluation_Metrices/        # Top-level metric scripts (mirror)
└── XAI_Methods/                    # Top-level XAI scripts (mirror)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/PUT-YOUR-LINK-HERE
cd "XAI_Evaluation (Exp_Pow_Action_ExpAcc) IEEE TIFS"

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
```
torch==2.4.0+cu124
scikit-learn>=1.3
shap>=0.43
lime>=0.2
captum>=0.7
scipy>=1.11
pandas>=2.0
numpy>=1.24
```

---

## Quick Start

```python
from XAI_Evaluation_Metrices.Explanatory_Power_2025 import ExplanatoryPowerEvaluator2025
from XAI_Evaluation_Metrices.Explanation_Accuracy_2025 import ExplanationAccuracyEvaluator2025
from XAI_Evaluation_Metrices.Actionability_2025 import ActionabilityEvaluator2025
from XAI_Evaluation_Metrices.XAI_Consensus_Score import FICScoreEvaluator

# 1. Train your model and generate explanation values
#    explanation_dict = {"SHAP": shap_values, "LIME": lime_values, ...}

# 2. Evaluate Explanatory Power
ep_evaluator = ExplanatoryPowerEvaluator2025(model=clf)
ep_results = ep_evaluator.evaluate(shap_values, X_test, method_name="SHAP", top_k=10)
print(f"Cohen's d: {ep_results['cohens_d']:.3f}")
print(f"95% CI: [{ep_results['ci_lower_95']:.3f}, {ep_results['ci_upper_95']:.3f}]")

# 3. Evaluate Actionability
act_evaluator = ActionabilityEvaluator2025(dataset_name="CIC_IIoT_2025", feature_names=feature_names)
act_results = act_evaluator.evaluate(shap_values, X_test, method_name="SHAP")
print(f"Mean Actionability: {act_results['mean_actionability']:.3f}")

# 4. Evaluate Explanation Accuracy
ea_evaluator = ExplanationAccuracyEvaluator2025(model=clf, X_reference=X_train)
ea_results = ea_evaluator.evaluate(shap_values, X_test, method_name="SHAP")
print(f"Flip Rate: {ea_results['flip_rate']:.3f}")
print(f"D-AUC: {ea_results['deletion_auc']:.3f}")

# 5. Evaluate FIC Score
fic_evaluator = FICScoreEvaluator(feature_names=feature_names)
fic_results = fic_evaluator.compute(explanation_dict, dataset_name="CIC_IIoT_2025")
print(f"Global FIC: {fic_results['global_fic']:.3f}")
```

---

## Reproducing Paper Results

```bash
# Step 1: Train all six classifiers on both datasets
python "XAI Evalation For IDS datasets/Models/Classical_ML/Phase_1_Training_ClassicalML_2025.py"
python "XAI Evalation For IDS datasets/Models/DeepLearning/Phase_1_Training_Transformer_2025.py"
python "XAI Evalation For IDS datasets/Models/DeepLearning/Phase_1_Training_LSTM_2025.py"

# Step 2: Generate all XAI explanations
python "XAI Evalation For IDS datasets/XAI_Methods/Generate_Explanations.py"

# Step 3: Run all four evaluation metrics
python "XAI Evalation For IDS datasets/XAI_Evaluation_Metrices/Run_All_Metrics_2025.py"

# Step 4: Run analysis and generate paper figures
python "XAI Evalation For IDS datasets/Analysis/Run_All_Analysis_2025.py"
```

---

## Hardware

All experiments were run on:
- **GPU:** NVIDIA RTX A6000 (48 GB VRAM)
- **Framework:** PyTorch 2.4.0+cu124
- **OS:** Ubuntu 22.04

Classical ML models (scikit-learn) were trained on CPU. Deep learning models required GPU for both training and SHAP KernelExplainer.

---

## Citation

If you use this framework or datasets in your research, please cite:

```bibtex
@article{bibers2025xai,
  author    = {Bibers, Ismail and Khalil, Issa and Abdallah, Mustafa},
  title     = {When Can {XAI} Explanations Be Trusted? A Validity-Aware
               Multi-Metric Evaluation Framework for Network Intrusion
               Detection Systems},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2025},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Ismail Bibers** — ibibers@purdue.edu — Purdue University
- **Mustafa Abdallah** — abdalla0@purdue.edu — Purdue University Indianapolis
- **Issa Khalil** — ikhalil@hbku.edu.qa — Hamad Bin Khalifa University
