"""
XAI_Config.py — Centralized configuration for all XAI methods
==============================================================
Single source of truth for:
  - Hyperparameters (no hardcoding in individual scripts)
  - Dataset paths and metadata
  - Feature tier classifications (NIST 3-tier actionability)
  - Random seeds and device settings

Usage:
    from XAI_Methods.XAI_Config import CONFIG, DATASET_META, FEATURE_TIERS
"""

import os, torch

# ── base paths ────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(_THIS_DIR)
READY_DIR  = os.path.join(ROOT, "IDS_Datasets", "Ready Datasets")
MODELS_DIR = os.path.join(ROOT, "Models")
RESULTS_DIR = os.path.join(ROOT, "XAI_Evaluation_Metrices", "Results")
EXPL_DIR   = os.path.join(ROOT, "explanations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXPL_DIR, exist_ok=True)

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── global config ─────────────────────────────────────────────────────────────
CONFIG = {
    # Reproducibility
    "random_seed": 42,

    # SHAP
    "shap_background_samples": 100,    # background sample size for KernelExplainer
    "shap_explain_samples":    1000,   # per dataset (test set samples to explain)
    "shap_interaction_top_k":  10,     # top-k features for interaction analysis

    # LIME
    "lime_num_samples":    5000,       # perturbed samples for local linear model
    "lime_num_features":   20,         # top features to include in explanation
    "lime_kernel_width":   None,       # None = auto (sqrt(n_features)*0.75)
    "lime_explain_samples": 1000,

    # Integrated Gradients (Captum)
    "ig_steps":          50,           # integration steps
    "ig_baseline":       "zero",       # zero vector baseline
    "ig_explain_samples": 1000,

    # Anchors
    "anchors_precision":   0.95,       # minimum rule precision
    "anchors_max_rule_len": 5,         # max features in one rule
    "anchors_explain_samples": 200,    # fewer samples (slow method)
    "anchors_num_coverage_samples": 10000,

    # Attention (Transformer/LSTM)
    "attn_explain_samples": 1000,
    "attn_layer": "all",               # use all transformer layers

    # Evaluation
    "top_k_features": 10,             # top-k for metric computation
    "n_cv_folds":     5,              # cross-validation folds
    "bootstrap_n":    2000,           # bootstrap iterations for CI
    "alpha":          0.05,           # significance level
    "cohens_d_threshold": 0.5,        # practical significance threshold
    "bonferroni_n_tests": 124,        # total pre-registered tests
}

# ── dataset metadata ──────────────────────────────────────────────────────────
DATASET_META = {
    "CIC_IIoT_2025": {
        "csv_path":    os.path.join(READY_DIR, "CIC_IIoT_2025_consolidated.csv"),
        "split_col":   "split",
        "label_col":   "label",
        "n_samples":   383_470,
        "n_features":  66,
        "n_classes":   8,
        "classes":     ["benign", "bruteforce", "ddos", "dos", "malware", "mitm", "recon", "web"],
        "domain":      "IoT network flows",
        "description": "CIC IIoT 2025 — IoT attack scenarios, temporal 1-second granularity",
        "classical_models_dir": os.path.join(MODELS_DIR, "Classical_ML"),
        "dl_models_dir":        os.path.join(MODELS_DIR, "DeepLearning"),
    },
    "IDS2025_Balanced": {
        "csv_path":    os.path.join(READY_DIR, "IDS2025_Balanced_final_with_split.csv"),
        "split_col":   "split",
        "label_col":   "label",
        "n_samples":   91_740,
        "n_features":  56,
        "n_classes":   7,
        "classes":     ["benign", "botnet ares", "brute force", "dos/ddos",
                        "infiltration", "portscan", "web attack"],
        "domain":      "Balanced network flows",
        "description": "IDS2025 Balanced — perfectly balanced multi-class IDS dataset",
        "classical_models_dir": os.path.join(MODELS_DIR, "Classical_ML"),
        "dl_models_dir":        os.path.join(MODELS_DIR, "DeepLearning"),
    },
}

# ── classical model names ─────────────────────────────────────────────────────
CLASSICAL_MODELS = ["DT", "LR", "RF", "XGB"]
DL_MODELS        = ["Transformer", "LSTM"]
ALL_XAI_METHODS  = ["SHAP", "LIME", "IntegratedGradients", "Anchors", "Attention"]


# ── 3-TIER ACTIONABILITY FRAMEWORK (NIST SP 800-94 grounded) ─────────────────
# Tier 1: Directly actionable — can create immediate IDS/firewall rule
# Tier 2: Semi-actionable — can set threshold/alert but requires analysis
# Tier 3: Non-actionable — inherent traffic property, cannot control
#
# Reference:
#   NIST SP 800-94: Guide to IDS and IPS
#   Suricata/Snort rule documentation
#   Snort content matching: protocol, port, flags, IP
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_TIERS = {
    # ── IDS2025_Balanced features (56 features) ───────────────────────────────
    "IDS2025_Balanced": {
        # TIER 1: Directly actionable — maps to firewall/IDS rule knob
        "tier_1": [
            "source_port",             # → block port range (Snort: dport/sport)
            "destination_port",        # → block port range
            "protocol",                # → block protocol (TCP/UDP/ICMP)
            "fwd_psh_flags",           # → PSH flag pattern rule
            "bwd_psh_flags",           # → PSH flag pattern rule
            "fwd_urg_flags",           # → URG flag rule
            "bwd_urg_flags",           # → URG flag rule
            "fin_flag_count",          # → FIN flood detection rule
            "syn_flag_count",          # → SYN flood detection rule
            "rst_flag_count",          # → RST pattern rule
            "psh_flag_count",          # → PSH flood rule
            "ack_flag_count",          # → ACK flood rule
            "urg_flag_count",          # → URG flag rule
            "cwe_flag_count",          # → CWR flag pattern
            "ece_flag_count",          # → ECE flag pattern
        ],
        # TIER 2: Semi-actionable — threshold/alert configuration
        "tier_2": [
            "total_fwd_packets",       # → rate limiting: packets/flow
            "total_backward_packets",  # → rate limiting
            "total_length_of_fwd_packets",  # → payload size threshold
            "total_length_of_bwd_packets",  # → payload size threshold
            "fwd_packet_length_max",   # → max packet size alert
            "fwd_packet_length_min",   # → min packet size filter
            "fwd_packet_length_mean",  # → mean size threshold
            "fwd_packet_length_std",   # → variance alert
            "bwd_packet_length_max",   # → max packet size
            "bwd_packet_length_min",
            "bwd_packet_length_mean",
            "bwd_packet_length_std",
            "flow_bytes_per_s",        # → bandwidth rate limit
            "flow_packets_per_s",      # → PPS rate limit
            "flow_iat_mean",           # → inter-arrival time alert
            "flow_iat_std",
            "flow_iat_max",
            "flow_iat_min",
            "fwd_iat_total",
            "fwd_iat_mean",
            "fwd_iat_std",
            "fwd_iat_max",
            "fwd_iat_min",
            "bwd_iat_total",
            "bwd_iat_mean",
            "bwd_iat_std",
            "bwd_iat_max",
            "bwd_iat_min",
            "fwd_header_length",       # → header length filter
            "bwd_header_length",
            "fwd_packets_per_s",       # → fwd rate limit
            "bwd_packets_per_s",       # → bwd rate limit
            "min_packet_length",       # → min size filter
            "max_packet_length",       # → max size filter
            "packet_length_mean",
            "packet_length_std",
            "packet_length_variance",
            "down_up_ratio",           # → asymmetry alert
            "average_packet_size",
            "avg_fwd_segment_size",
            "avg_bwd_segment_size",
        ],
        # TIER 3: Non-actionable — emergent statistical properties
        "tier_3": [
            "flow_duration",           # → derived from session, not controllable
            "active_mean",
            "active_std",
            "active_max",
            "active_min",
            "idle_mean",
            "idle_std",
            "idle_max",
            "idle_min",
            "subflow_fwd_packets",
            "subflow_fwd_bytes",
            "subflow_bwd_packets",
            "subflow_bwd_bytes",
            "init_win_bytes_forward",
            "init_win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
        ],
    },

    # ── CIC_IIoT_2025 features (66 features) ─────────────────────────────────
    "CIC_IIoT_2025": {
        # TIER 1: Directly actionable (IoT protocols + flag patterns)
        "tier_1": [
            "network_ports_all_count",         # → block port range
            "network_ports_dst_count",         # → destination port blocking
            "network_ports_src_count",         # → source port pattern
            "network_tcp_flags_fin_count",     # → FIN pattern rule
            "network_tcp_flags_syn_count",     # → SYN flood rule
            "network_tcp_flags_rst_count",     # → RST pattern rule
            "network_tcp_flags_psh_count",     # → PSH flag rule
            "network_tcp_flags_ack_count",     # → ACK flood rule
            "network_tcp_flags_urg_count",     # → URG flag rule (if present)
            "network_fragmented_packets",      # → fragmentation block rule
        ],
        # TIER 2: Semi-actionable (rate limiting, thresholds)
        "tier_2": [
            "network_packets_all_count",       # → PPS rate limit
            "network_packets_src_count",       # → src rate limit
            "network_packets_dst_count",       # → dst rate limit
            "network_fragmentation_score",     # → fragmentation alert
            "network_header_length_avg",       # → header size threshold
            "network_header_length_max",       # → max header alert
            "network_header_length_min",       # → min header filter
            "network_header_length_std",       # → header variance alert
            "log_messages_count",              # → log rate alert
            "log_interval_messages",           # → message frequency threshold
            "log_data_types_count",            # → data type diversity alert
            "log_data_ranges_avg",             # → data range threshold
            "log_data_ranges_max",             # → max data range alert
            "log_data_ranges_min",
            "log_data_ranges_std_deviation",
        ],
        # TIER 3: Non-actionable (statistical/emergent)
        "tier_3": [],  # will be populated dynamically for remaining features
    },
}

# Dynamically assign any CIC_IIoT feature not in tier_1 or tier_2 to tier_3
def get_all_tiers(dataset_name: str, feature_names: list) -> dict:
    """Return complete tier mapping for a dataset, including dynamic tier_3."""
    tiers = FEATURE_TIERS.get(dataset_name, {"tier_1": [], "tier_2": [], "tier_3": []})
    t1 = set(tiers["tier_1"]); t2 = set(tiers["tier_2"])
    t3 = [f for f in feature_names if f not in t1 and f not in t2]
    return {"tier_1": list(t1), "tier_2": list(t2), "tier_3": t3}


def get_tier(feature_name: str, dataset_name: str) -> int:
    """Return 1, 2, or 3 for a feature's actionability tier."""
    tiers = FEATURE_TIERS.get(dataset_name, {})
    if feature_name in tiers.get("tier_1", []):
        return 1
    if feature_name in tiers.get("tier_2", []):
        return 2
    return 3  # default non-actionable


TIER_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.0}  # used in actionability score computation
