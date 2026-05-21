"""
IntegratedGradients.py — Gradient-based XAI for PyTorch DL models
===================================================================
Method: Integrated Gradients (Sundararajan et al., 2017)
Library: Captum (official IG implementation)

Configuration (from XAI_Config):
  - integration_steps: 50
  - baseline: zero vector
  - Compatible with: FeatureTokenizerTransformer, LSTMClassifier

Output format: np.ndarray (n_samples, n_features) — identical to SHAP/LIME

Addresses reviewer criticism:
  - "Only SHAP and LIME — no SOTA methods" (R1, R2, R3)
  - Provides gradient-based explanations (fundamentally different from
    feature-permutation methods, complementary for FIC score)
"""

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from captum.attr import IntegratedGradients as CaptumIG
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("[WARNING] captum not installed. Run: pip install captum")


class IntegratedGradientsExplainer:
    """
    Integrated Gradients for PyTorch tabular models (Transformer, LSTM).
    Uses Captum's implementation with zero-vector baseline.

    The attribution A_i for feature i is:
        A_i = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα
    where x' is the baseline (zero vector) and F is the model's predicted class output.
    """

    def __init__(self, model: nn.Module,
                 device: torch.device = None,
                 n_steps: int = 50,
                 random_state: int = 42):
        """
        Parameters
        ----------
        model       : Trained PyTorch model (must have forward() returning logits)
        device      : torch.device (auto-detected if None)
        n_steps     : Integration approximation steps (50 = standard)
        random_state: Seed (for reproducibility of any stochastic parts)
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Install captum: pip install captum")
        if not isinstance(model, nn.Module):
            raise TypeError("IntegratedGradientsExplainer requires a PyTorch nn.Module")

        self.model        = model.eval()
        self.device       = device or (torch.device("cuda:0") if torch.cuda.is_available()
                                       else torch.device("cpu"))
        self.n_steps      = n_steps
        self.random_state = random_state
        self.model.to(self.device)

        # Wrap model: IG needs a function that returns a single target output
        self._ig = CaptumIG(self._forward_for_ig)

    # ── public API ─────────────────────────────────────────────────────────────

    def explain_batch(self, X: pd.DataFrame, n_samples: int = 1000,
                      batch_size: int = 64) -> np.ndarray:
        """
        Compute IG attributions in mini-batches to avoid CUDA OOM.
        Uses batch_size=64 by default (safe for 48 GB GPU with 50 IG steps).
        """
        # Free any cached GPU memory from previous operations
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        X_sub = self._prepare(X, n_samples)
        n     = len(X_sub)
        was_training = self.model.training
        self.model.train()  # cuDNN RNN requires train mode for backward

        all_attrs = []
        for start in range(0, n, batch_size):
            end     = min(start + batch_size, n)
            x_b     = torch.tensor(X_sub[start:end], dtype=torch.float32).to(self.device)
            base_b  = torch.zeros_like(x_b)

            with torch.no_grad():
                targets = self.model(x_b).argmax(dim=1)

            # Temporarily restore eval for forward in IG path that doesn't need train
            attrs, _ = self._ig.attribute(
                inputs    = x_b,
                baselines = base_b,
                target    = targets,
                n_steps   = self.n_steps,
                return_convergence_delta = True,
            )
            all_attrs.append(attrs.detach().cpu().numpy())

            # Free batch tensors immediately
            del x_b, base_b, targets, attrs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.model.train(was_training)
        return np.concatenate(all_attrs, axis=0)  # (n_samples, n_features)

    def explain_single(self, instance: pd.DataFrame) -> np.ndarray:
        return self.explain_batch(instance, n_samples=1)[0]

    def convergence_delta(self, X: pd.DataFrame, n_samples: int = 100) -> float:
        """
        Compute mean |convergence delta| to validate approximation quality.
        Should be close to 0 (< 0.01 is ideal with 50 steps).
        """
        X_sub    = self._prepare(X, n_samples)
        x_tensor = torch.tensor(X_sub, dtype=torch.float32).to(self.device)
        baseline = torch.zeros_like(x_tensor)
        with torch.no_grad():
            targets = self.model(x_tensor).argmax(dim=1)
        _, delta = self._ig.attribute(
            inputs=x_tensor, baselines=baseline, target=targets,
            n_steps=self.n_steps, return_convergence_delta=True,
        )
        return float(delta.abs().mean().item())

    def timed_explain(self, X: pd.DataFrame, n_samples: int = 100) -> dict:
        X_sub = self._prepare(X, n_samples)
        t0    = time.time()
        self.explain_batch(pd.DataFrame(X_sub), n_samples=n_samples)
        elapsed = time.time() - t0
        return {
            "method":            "IntegratedGradients",
            "n_steps":           self.n_steps,
            "n_samples":         n_samples,
            "total_seconds":     round(elapsed, 3),
            "seconds_per_sample":round(elapsed / max(n_samples, 1), 4),
        }

    # ── internal ───────────────────────────────────────────────────────────────

    def _forward_for_ig(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through that returns raw logits for Captum."""
        return self.model(x)

    def _prepare(self, X, n_samples: int) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:n_samples].values.astype(np.float32)
        return np.asarray(X[:n_samples], dtype=np.float32)


# ── loader utility ────────────────────────────────────────────────────────────

def load_dl_model_for_ig(model_class, pth_path: str, device: torch.device):
    """
    Load a saved DL model checkpoint and return an IntegratedGradientsExplainer.

    Parameters
    ----------
    model_class : Python class (FeatureTokenizerTransformer or LSTMClassifier)
    pth_path    : Path to .pth file
    device      : torch device

    Returns
    -------
    explainer: IntegratedGradientsExplainer
    feature_names: list[str]
    label_classes: list[str]
    """
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    feature_names   = ckpt["feature_names"]
    label_classes   = list(ckpt["label_encoder_classes"])
    n_features      = len(feature_names)
    n_classes       = len(label_classes)

    model = model_class(
        n_features=n_features,
        n_classes=n_classes,
        **{k: cfg[k] for k in cfg if k not in
           ("lr", "weight_decay", "batch_size", "max_epochs", "patience",
            "scheduler", "seed", "bidirectional") and k in
           ("d_model", "n_heads", "n_layers", "dropout",
            "hidden_size", "bidirectional")}
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    explainer = IntegratedGradientsExplainer(model, device=device)
    return explainer, feature_names, label_classes
