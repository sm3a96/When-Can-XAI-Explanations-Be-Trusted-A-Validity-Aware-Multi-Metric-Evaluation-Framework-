"""
AttentionExplanation.py — Native DL explanation via attention weights
=====================================================================
Extracts attention weights from:
  - FeatureTokenizerTransformer (multi-head self-attention across features)
  - LSTMClassifier (soft feature-level attention)

These weights are used as a feature importance proxy — the model's own
internal allocation of "attention" to each feature.

Key advantage over SHAP/LIME:
  - Zero additional computation (weights already computed during forward pass)
  - Captures which features the DL model actually attends to (not post-hoc)
  - Temporal patterns visible across feature positions

Output: np.ndarray (n_samples, n_features) — same format as SHAP/LIME

Addresses reviewer: "No DL-specific XAI methods" (R1, R3)
Used in:
  - Phase 3.4: FIC Score (consensus with SHAP/LIME/IG)
  - Phase 4.2: DL vs Classical interpretability comparison
  - Phase 4.4: SHAP Interaction comparison vs attention
"""

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class AttentionExplainer:
    """
    Extracts feature-level attention weights from PyTorch DL models.
    Supports both Transformer (multi-head attention) and LSTM (soft attention).
    """

    def __init__(self, model: nn.Module,
                 model_arch: str,
                 feature_names: list,
                 device: torch.device = None,
                 layer: str = "last"):
        """
        Parameters
        ----------
        model        : Trained PyTorch model (in eval mode)
        model_arch   : 'transformer' or 'lstm'
        feature_names: List of feature column names
        device       : torch.device
        layer        : 'last' (last attention layer) | 'mean' (mean across layers)
        """
        if model_arch not in ("transformer", "lstm"):
            raise ValueError("model_arch must be 'transformer' or 'lstm'")

        self.model         = model.eval()
        self.model_arch    = model_arch
        self.feature_names = feature_names
        self.device        = device or (torch.device("cuda:0") if torch.cuda.is_available()
                                        else torch.device("cpu"))
        self.layer         = layer
        self.model.to(self.device)

    # ── public API ─────────────────────────────────────────────────────────────

    def explain_batch(self, X: pd.DataFrame, n_samples: int = 1000) -> np.ndarray:
        """
        Extract attention weights for up to n_samples instances.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Attention scores per feature (aggregated across heads for Transformer).
        """
        X_sub    = self._prepare(X, n_samples)
        x_tensor = torch.tensor(X_sub, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if self.model_arch == "transformer":
                return self._transformer_attention(x_tensor)
            else:
                return self._lstm_attention(x_tensor)

    def explain_single(self, instance: pd.DataFrame) -> np.ndarray:
        return self.explain_batch(instance, n_samples=1)[0]

    def timed_explain(self, X: pd.DataFrame, n_samples: int = 1000) -> dict:
        X_sub = self._prepare(X, n_samples)
        t0    = time.time()
        self.explain_batch(pd.DataFrame(X_sub, columns=self.feature_names), n_samples=n_samples)
        elapsed = time.time() - t0
        return {
            "method":            "Attention",
            "model_arch":        self.model_arch,
            "n_samples":         n_samples,
            "total_seconds":     round(elapsed, 3),
            "seconds_per_sample":round(elapsed / max(n_samples, 1), 4),
        }

    def per_class_attention(self, X: pd.DataFrame, y_pred: np.ndarray,
                            class_names: list, n_samples: int = 1000) -> dict:
        """
        Compute mean attention weights per predicted class.
        Used in Phase 4.3 (attack-type-specific XAI profiles).

        Returns
        -------
        dict: {class_name: np.ndarray (n_features,)} — mean attention per class
        """
        attn    = self.explain_batch(X, n_samples=n_samples)
        y_pred  = np.asarray(y_pred[:n_samples])
        result  = {}

        for cls_idx, cls_name in enumerate(class_names):
            mask = (y_pred == cls_idx)
            if mask.sum() > 0:
                result[cls_name] = attn[mask].mean(axis=0)
            else:
                result[cls_name] = np.zeros(len(self.feature_names))

        return result

    # ── internal ───────────────────────────────────────────────────────────────

    def _transformer_attention(self, x_tensor: torch.Tensor) -> np.ndarray:
        """
        For FeatureTokenizerTransformer:
        - Each feature is a token (including [CLS])
        - We get attention FROM [CLS] TO each feature token
        - Multi-head: average across heads
        - Multi-layer: use last layer or average (per config)
        """
        B, n_feats = x_tensor.shape[0], x_tensor.shape[1]
        x_norm = self.model.input_norm(x_tensor)

        # Embed features
        tokens = torch.stack(
            [emb(x_norm[:, i:i+1]) for i, emb in enumerate(self.model.feature_embeddings)],
            dim=1
        )
        cls    = self.model.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_feats+1, d_model)

        # Collect attention weights from each layer
        layer_attentions = []
        src = tokens
        for layer in self.model.transformer.layers:
            _, attn_weights = layer.self_attn(
                src, src, src,
                need_weights=True,
                average_attn_weights=True  # average across heads
            )
            # attn_weights: (B, seq_len, seq_len)
            # CLS attention to features: row 0 (CLS) to columns 1: (feature tokens)
            cls_to_feat = attn_weights[:, 0, 1:]  # (B, n_feats)
            layer_attentions.append(cls_to_feat)
            src = layer(src)

        # Aggregate across layers
        if self.layer == "last":
            feature_attn = layer_attentions[-1]
        else:  # mean
            feature_attn = torch.stack(layer_attentions, dim=0).mean(dim=0)

        return feature_attn.cpu().numpy()  # (B, n_feats)

    def _lstm_attention(self, x_tensor: torch.Tensor) -> np.ndarray:
        """
        For LSTMClassifier:
        - The model's FeatureAttention layer computes soft weights over feature positions
        - We directly retrieve self.model.last_attn_weights after a forward pass
        """
        # Run forward pass to populate last_attn_weights
        _ = self.model(x_tensor)

        if self.model.last_attn_weights is None:
            raise RuntimeError("LSTM attention weights not found. "
                               "Ensure model has FeatureAttention layer.")

        attn = self.model.last_attn_weights.cpu().numpy()  # (B, n_features)
        return attn

    def _prepare(self, X, n_samples: int) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
            return X.iloc[:n_samples].values.astype(np.float32)
        return np.asarray(X[:n_samples], dtype=np.float32)


# ── loader utility ────────────────────────────────────────────────────────────

def load_attention_explainer(model_class, pth_path: str,
                             model_arch: str,
                             device: torch.device = None) -> tuple:
    """
    Load a DL model and return an AttentionExplainer.

    Returns
    -------
    explainer     : AttentionExplainer
    feature_names : list[str]
    label_classes : list[str]
    """
    device = device or (torch.device("cuda:0") if torch.cuda.is_available()
                        else torch.device("cpu"))
    ckpt   = torch.load(pth_path, map_location=device, weights_only=False)
    cfg    = ckpt["config"]
    feature_names = ckpt["feature_names"]
    label_classes = list(ckpt["label_encoder_classes"])
    n_features    = len(feature_names)
    n_classes     = len(label_classes)

    # Build model with saved config
    init_kwargs = {k: cfg[k] for k in cfg
                   if k in ("d_model", "n_heads", "n_layers", "dropout",
                             "hidden_size", "bidirectional")}
    model = model_class(n_features=n_features, n_classes=n_classes, **init_kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    explainer = AttentionExplainer(
        model=model, model_arch=model_arch,
        feature_names=feature_names, device=device
    )
    return explainer, feature_names, label_classes
