"""
LIME.py — Production-ready LIME Explainer
==========================================
Class-based implementation supporting:
  - Batch explanation generation
  - Multi-class: extracts per-predicted-class LIME weights
  - Adaptive kernel_width (sqrt(n_features) × 0.75)
  - Consistent output format matching SHAP: np.ndarray (n_samples, n_features)
  - Operational timing for feasibility analysis

Addresses reviewer criticism:
  - No arbitrary class selection (uses predicted class)
  - Consistent feature ordering across instances
  - num_samples=5000 (sufficient local approximation)
"""

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
from lime import lime_tabular


class LIMEExplainer:
    """
    Unified LIME wrapper for any sklearn-compatible or PyTorch model.
    Output format is intentionally identical to SHAPExplainer.explain_batch().
    """

    def __init__(self, model, X_train: pd.DataFrame,
                 mode: str = "classification",
                 random_state: int = 42,
                 num_samples: int = 5000,
                 kernel_width: float = None,
                 discretize_continuous: bool = True):
        """
        Parameters
        ----------
        model      : Trained model
        X_train    : Training data (DataFrame with feature names)
        mode       : 'classification' or 'regression'
        num_samples: Perturbed samples for local linear fit (default 5000)
        kernel_width: Controls locality; None = auto (sqrt(n_features) × 0.75)
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame with feature names")

        np.random.seed(random_state); random.seed(random_state)

        self.model         = model
        self.feature_names = list(X_train.columns)
        self.mode          = mode
        self.random_state  = random_state
        self.num_samples   = num_samples

        if kernel_width is None:
            kernel_width = np.sqrt(len(self.feature_names)) * 0.75

        self._class_names = None
        if mode == "classification" and hasattr(model, "classes_"):
            self._class_names = [str(c) for c in model.classes_]

        self._explainer = lime_tabular.LimeTabularExplainer(
            training_data         = X_train.values,
            feature_names         = self.feature_names,
            class_names           = self._class_names,
            mode                  = mode,
            discretize_continuous = discretize_continuous,
            kernel_width          = kernel_width,
            random_state          = random_state,
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def explain_batch(self, X: pd.DataFrame, n_samples: int = 1000,
                      num_features: int = None) -> np.ndarray:
        """
        Explain up to n_samples instances.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            LIME weight for each feature for the predicted class.
        """
        X_sub = self._prepare(X, n_samples)
        if num_features is None:
            num_features = len(self.feature_names)

        predict_fn = self._build_predict_fn(X_sub)
        result = np.zeros((len(X_sub), len(self.feature_names)), dtype=np.float64)

        for i, (_, row) in enumerate(X_sub.iterrows()):
            label_idx = self._get_predicted_class(row.values.reshape(1, -1))
            exp = self._explainer.explain_instance(
                data_row   = row.values.astype(np.double),
                predict_fn = predict_fn,
                num_features = num_features,
                num_samples  = self.num_samples,
                labels       = [label_idx],
            )
            weights_map = dict(exp.local_exp.get(label_idx, []))
            for feat_idx, weight in weights_map.items():
                if 0 <= feat_idx < len(self.feature_names):
                    result[i, feat_idx] = weight

        return result

    def explain_single(self, instance: pd.DataFrame) -> np.ndarray:
        """Explain one instance. Returns (n_features,) array."""
        return self.explain_batch(instance, n_samples=1)[0]

    def timed_explain(self, X: pd.DataFrame, n_samples: int = 50) -> dict:
        """Timing benchmark for SOC operational feasibility."""
        X_sub = self._prepare(X, n_samples)
        t0    = time.time()
        self.explain_batch(X_sub, n_samples=n_samples)
        elapsed = time.time() - t0
        return {
            "method":            "LIME",
            "n_samples":         n_samples,
            "total_seconds":     round(elapsed, 3),
            "seconds_per_sample":round(elapsed / max(n_samples, 1), 4),
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            X = X[self.feature_names]
        return X.iloc[:n_samples].reset_index(drop=True)

    def _build_predict_fn(self, X_ref: pd.DataFrame):
        feature_names = self.feature_names
        model = self.model

        if self.mode == "classification":
            if hasattr(model, "predict_proba"):
                def fn(x):
                    return model.predict_proba(pd.DataFrame(x, columns=feature_names))
            else:
                import torch as _torch
                _dev = next(iter(model.parameters()), _torch.tensor(0)).device \
                       if isinstance(model, _torch.nn.Module) else _torch.device("cpu")
                def fn(x):
                    t = _torch.tensor(x, dtype=_torch.float32).to(_dev)
                    with _torch.no_grad():
                        return _torch.softmax(model(t), dim=1).cpu().numpy()
        else:
            def fn(x):
                return model.predict(pd.DataFrame(x, columns=feature_names))

        return fn

    def _get_predicted_class(self, x_arr: np.ndarray) -> int:
        df = pd.DataFrame(x_arr, columns=self.feature_names)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[0]
            return int(np.argmax(proba))
        if hasattr(self.model, "predict"):
            pred = self.model.predict(df)[0]
            classes = getattr(self.model, "classes_", None)
            if classes is not None and not isinstance(pred, (int, np.integer)):
                return int(np.where(classes == pred)[0][0])
            return int(pred)
        # PyTorch
        import torch
        t = torch.tensor(x_arr, dtype=torch.float32)
        with torch.no_grad():
            return int(self.model(t).argmax(1).item())
