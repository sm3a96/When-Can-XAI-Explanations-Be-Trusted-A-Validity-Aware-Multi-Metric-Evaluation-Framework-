"""
SHAP.py — Production-ready SHAP Explainer
==========================================
Class-based implementation supporting:
  - All model types: TreeExplainer (DT/RF/XGB), LinearExplainer (LR),
    DeepExplainer (PyTorch DL), KernelExplainer (fallback)
  - Batch explanation generation for 1000 instances
  - SHAP interaction values (novel contribution — Phase 4.4)
  - Multi-class: extracts per-predicted-class SHAP values
  - Consistent output format: np.ndarray (n_samples, n_features)

Usage:
    explainer = SHAPExplainer(model, X_train, model_type="tree")
    values    = explainer.explain_batch(X_test, n_samples=1000)
    inter     = explainer.interaction_values(X_test, n_samples=200)
"""

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """
    Unified SHAP wrapper for classical ML and DL models.
    Addresses Reviewer 2 criticism: consistent multi-class handling,
    no arbitrary class selection, validated output format.
    """

    SUPPORTED_TYPES = ("tree", "linear", "kernel", "deep")

    def __init__(self, model, X_train: pd.DataFrame,
                 model_type: str = "auto",
                 random_state: int = 42,
                 n_background: int = 100):
        """
        Parameters
        ----------
        model       : Trained model (sklearn, xgboost, or torch.nn.Module)
        X_train     : Training data (DataFrame with feature names)
        model_type  : 'auto' | 'tree' | 'linear' | 'kernel' | 'deep'
        random_state: Seed for background sample selection
        n_background: Background sample size for KernelExplainer
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame with feature names")

        self.model        = model
        self.feature_names = list(X_train.columns)
        self.random_state  = random_state
        self.model_type    = self._detect_type(model) if model_type == "auto" else model_type
        self._is_classifier = hasattr(model, "predict_proba") or (
            hasattr(model, "forward")  # PyTorch
        )

        np.random.seed(random_state)
        self.background = X_train.sample(
            min(n_background, len(X_train)), random_state=random_state
        )
        self.explainer  = self._build_explainer()

    # ── public API ─────────────────────────────────────────────────────────────

    def explain_batch(self, X: pd.DataFrame, n_samples: int = 1000) -> np.ndarray:
        """
        Generate SHAP values for up to n_samples instances.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Absolute mean SHAP values per sample for the predicted class.
        """
        X_sub = self._prepare(X, n_samples)
        np.random.seed(self.random_state)

        raw = self._compute_shap_values(X_sub)
        return self._to_per_class_matrix(raw, X_sub)

    def explain_single(self, instance: pd.DataFrame) -> np.ndarray:
        """Explain one instance. Returns (n_features,) array."""
        return self.explain_batch(instance, n_samples=1)[0]

    def interaction_values(self, X: pd.DataFrame, n_samples: int = 200) -> np.ndarray:
        """
        Compute SHAP interaction values (pairwise feature interactions).
        Only available for TreeExplainer (DT, RF, XGB).
        Novel contribution: reveals compound attack signatures.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features, n_features)
        """
        if self.model_type != "tree":
            raise NotImplementedError(
                "SHAP interaction values require TreeExplainer (tree models only). "
                f"Current model_type='{self.model_type}'"
            )
        X_sub = self._prepare(X, n_samples)
        np.random.seed(self.random_state)
        inter = self.explainer.shap_interaction_values(X_sub)

        if isinstance(inter, list):
            # Multi-class: select per-predicted-class interaction matrix
            classes  = self._get_predicted_classes(X_sub)
            result   = np.stack([inter[classes[i]][i] for i in range(len(X_sub))], axis=0)
        elif inter.ndim == 4:
            # (n_samples, n_classes, n_features, n_features) — select predicted class
            classes = self._get_predicted_classes(X_sub)
            result  = inter[np.arange(len(X_sub)), classes]
        else:
            result = inter  # binary or regression

        return result  # (n_samples, n_features, n_features)

    def timed_explain(self, X: pd.DataFrame, n_samples: int = 100) -> dict:
        """Explain n_samples and return timing info for operational feasibility analysis."""
        X_sub = self._prepare(X, n_samples)
        t0    = time.time()
        self.explain_batch(X_sub, n_samples=n_samples)
        elapsed = time.time() - t0
        return {
            "method":            "SHAP",
            "model_type":        self.model_type,
            "n_samples":         n_samples,
            "total_seconds":     round(elapsed, 3),
            "seconds_per_sample":round(elapsed / max(n_samples, 1), 4),
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _detect_type(self, model) -> str:
        name = type(model).__name__.lower()
        if any(k in name for k in ("tree", "forest", "boost", "xgb", "lgbm", "gbm")):
            return "tree"
        if any(k in name for k in ("logistic", "linear", "regression", "ridge", "lasso")):
            return "linear"
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return "deep"
        except ImportError:
            pass
        return "kernel"

    def _build_explainer(self):
        # For DL (deep) models: skip DeepExplainer — BatchNorm + attention break additivity.
        # Use KernelExplainer directly with device-aware predict function.
        if self.model_type not in ("tree", "linear"):
            return self._build_kernel_explainer()
        try:
            if self.model_type == "tree":
                return shap.TreeExplainer(self.model)
            if self.model_type == "linear":
                return shap.LinearExplainer(self.model, self.background)
        except Exception as e:
            print(f"[SHAP] {self.model_type} explainer failed ({e}). Falling back to KernelExplainer.")
        return self._build_kernel_explainer()

    def _build_kernel_explainer(self):
        """KernelExplainer that handles both sklearn and PyTorch models (device-aware)."""
        import torch as _torch
        _device = None
        if isinstance(self.model, _torch.nn.Module):
            try:
                _device = next(self.model.parameters()).device
            except StopIteration:
                pass

        def predict_fn(X):
            if self._is_classifier:
                if hasattr(self.model, "predict_proba"):
                    df = pd.DataFrame(X, columns=self.feature_names)
                    return self.model.predict_proba(df)
                # PyTorch — move to model device
                t = _torch.tensor(np.asarray(X), dtype=_torch.float32)
                if _device is not None:
                    t = t.to(_device)
                with _torch.no_grad():
                    return _torch.softmax(self.model(t), dim=1).cpu().numpy()
            df = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict(df)

        return shap.KernelExplainer(predict_fn, self.background)

    def _prepare(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            X = X[self.feature_names]
        return X.iloc[:n_samples].reset_index(drop=True)

    def _compute_shap_values(self, X_sub: pd.DataFrame):
        """Raw SHAP values — KernelExplainer works on DataFrames/numpy; tree/linear also."""
        return self.explainer.shap_values(X_sub)

    def _get_predicted_classes(self, X_sub: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X_sub)
            classes = getattr(self.model, "classes_", None)
            if classes is not None and not np.issubdtype(np.array(preds).dtype, np.integer):
                return np.array([np.where(classes == p)[0][0] for p in preds], dtype=int)
            return np.asarray(preds, dtype=int)
        # PyTorch
        import torch
        t = torch.tensor(X_sub.values, dtype=torch.float32)
        with torch.no_grad():
            return self.model(t).argmax(1).numpy()

    def _to_per_class_matrix(self, raw, X_sub: pd.DataFrame) -> np.ndarray:
        """
        Convert raw SHAP values (any format) to (n_samples, n_features).
        For multi-class: selects the predicted class values.
        """
        n = len(X_sub)

        # New API: shap.Explanation object
        if hasattr(raw, "values"):
            vals = np.asarray(raw.values)
            if vals.ndim == 3:
                classes = self._get_predicted_classes(X_sub)
                return np.stack([vals[i, :, classes[i]] for i in range(n)], axis=0)
            return vals

        # Old API: list of arrays (one per class)
        if isinstance(raw, list):
            classes = self._get_predicted_classes(X_sub)
            return np.stack(
                [np.asarray(raw[classes[i]])[i] for i in range(n)], axis=0
            )

        raw = np.asarray(raw)
        if raw.ndim == 3:
            # (n_samples, n_features, n_classes)
            classes = self._get_predicted_classes(X_sub)
            return raw[np.arange(n), :, classes]
        return raw  # (n_samples, n_features) already
