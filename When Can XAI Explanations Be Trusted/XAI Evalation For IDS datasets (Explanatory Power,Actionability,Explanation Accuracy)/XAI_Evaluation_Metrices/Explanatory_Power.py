# Explanatory Power measures how much of the model's output can be "explained" by the selected features according to the explainer.
# A high explanatory power means the explanation covers most of the model's prediction logic.

"""
Interpretation Guide:

Metric	           Good Value	            Interpretation
Raw Power	       Close to model outputs	Explanations capture true effect sizes
Normalized Power   80-120%	                Explanations fully account for outputs
R² Score	       >0.8	                    Explanations match model behavior well
"""


from __future__ import annotations

import inspect
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import shap
from lime.lime_tabular import LimeTabularExplainer


def _safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class XAIExplanatoryPowerEvaluator:
    """
    explanatory power evaluator for SHAP and LIME.

    For instance i and target t:

      y_i: model output in the explainer's space (prob/logit/margin)
      b_i: explainer baseline/intercept
      s_i: sum_j φ_ij (signed sum)

      delta d_i = y_i - b_i
      normalized_power = s_i / d_i
      R² = R²(signed_sums, deltas)
    """

    def __init__(self, model, n_jobs: int = -1, random_state: Optional[int] = None):
        self.model = model
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._is_classifier = hasattr(model, "predict_proba")

    def evaluate(
        self,
        explainer: Union[shap.Explainer, LimeTabularExplainer, object],
        X: pd.DataFrame,
        method: str = "auto",
        normalization: str = "output",
        class_idx: Optional[int] = None,
        lime_num_samples: int = 5000,
        feature_subset: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, int, np.ndarray, str, np.floating, Tuple]]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if feature_subset is not None:
            missing = [c for c in feature_subset if c not in X.columns]
            if missing:
                raise ValueError(f"feature_subset contains unknown columns: {missing}")
            X = X[feature_subset].copy()

        X = X.reset_index(drop=True)

        # Auto-detect method
        if method == "auto":
            method = "lime" if isinstance(explainer, LimeTabularExplainer) or hasattr(explainer, "explain_instance") else "shap"

        # Prepare targets
        class_indices = None
        y_prob_cls = None
        y_logit_cls = None
        y_margin_cls = None

        if self._is_classifier:
            proba = self.model.predict_proba(X)
            y_pred = self.model.predict(X)
            model_classes = getattr(self.model, "classes_", None)

            if class_idx is None:
                if model_classes is not None and not np.issubdtype(np.asarray(y_pred).dtype, np.integer):
                    class_indices = np.array([int(np.where(model_classes == p)[0][0]) for p in y_pred])
                else:
                    class_indices = np.asarray(y_pred, dtype=int)
            else:
                class_indices = np.full(len(X), int(class_idx), dtype=int)

            y_prob_cls = proba[np.arange(len(X)), class_indices]

            # Binary logit candidate
            if proba.shape[1] == 2:
                y_logit_cls = _safe_logit(y_prob_cls)

            # Margins/logits if available (LogReg/XGBoost)
            y_margin_cls = self._get_margins_for_selected_class(X, class_indices)
        else:
            y_prob_cls = self.model.predict(X).astype(float)

        if method == "shap":
            signed_sums, abs_sums, baselines, target_outputs = self._compute_shap_sums_and_baselines(
                explainer,
                X,
                candidates={"margin": y_margin_cls, "logit": y_logit_cls, "prob": y_prob_cls},
            )
        elif method == "lime":
            signed_sums, abs_sums, baselines, target_outputs = self._compute_lime_sums_and_baselines(
                explainer, X, class_indices, lime_num_samples
            )
        else:
            raise ValueError("method must be 'shap', 'lime', or 'auto'")

        signed_sums = np.asarray(signed_sums).reshape(-1)
        abs_sums = np.asarray(abs_sums).reshape(-1)
        baselines = np.asarray(baselines).reshape(-1)
        target_outputs = np.asarray(target_outputs).reshape(-1)

        deltas = target_outputs - baselines

        # Normalized power
        eps = 1e-9
        denom = np.where(np.abs(deltas) > eps, deltas, np.nan)
        normalized = signed_sums / denom

        # Fidelity metrics
        valid_mask = ~np.isnan(normalized) & np.isfinite(signed_sums) & np.isfinite(deltas)
        if valid_mask.sum() > 1 and np.var(signed_sums[valid_mask]) > 0:
            r2 = r2_score(deltas[valid_mask], signed_sums[valid_mask])
        else:
            r2 = np.nan

        try:
            pearson_corr = pearsonr(np.abs(deltas[valid_mask]), abs_sums[valid_mask])[0] if valid_mask.any() else np.nan
        except Exception:
            pearson_corr = np.nan
        try:
            spearman_corr = spearmanr(np.abs(deltas[valid_mask]), abs_sums[valid_mask])[0] if valid_mask.any() else np.nan
        except Exception:
            spearman_corr = np.nan

        def safe_stat(arr, func, default=np.nan):
            try:
                return func(arr)
            except Exception:
                return default

        return {
            "model_outputs": target_outputs,
            "baselines": baselines,
            "deltas": deltas,
            "signed_powers": signed_sums,
            "raw_powers": abs_sums,
            "normalized_powers": normalized,
            "mean_raw_power": safe_stat(abs_sums, np.mean),
            "std_raw_power": safe_stat(abs_sums, np.std),
            "min_raw_power": safe_stat(abs_sums, np.min),
            "max_raw_power": safe_stat(abs_sums, np.max),
            "mean_normalized_power": safe_stat(normalized[~np.isnan(normalized)], np.mean)
            if np.any(~np.isnan(normalized)) else np.nan,
            "std_normalized_power": safe_stat(normalized[~np.isnan(normalized)], np.std)
            if np.any(~np.isnan(normalized)) else np.nan,
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            "r2_score": r2,
            "method": method,
            "normalization": normalization,
            "num_instances": len(X),
        }

    # ------------------------------ Internals ------------------------------

    def _get_margins_for_selected_class(
        self, X: pd.DataFrame, class_indices: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Obtain raw margins/logits for the selected class when available.
        Handles:
          - sklearn LogisticRegression via decision_function
          - XGBoost via predict(..., output_margin=True) or booster API
        """
        # sklearn linear/logistic: decision_function
        if hasattr(self.model, "decision_function"):
            try:
                margins = self.model.decision_function(X)
                margins = np.asarray(margins)
                if margins.ndim == 1:
                    return margins  # binary
                if margins.ndim == 2 and class_indices is not None:
                    idx = np.arange(len(X))
                    return margins[idx, class_indices]
            except Exception:
                pass

        # Generic predict(output_margin=True)
        try:
            sig = inspect.signature(self.model.predict)
            if "output_margin" in sig.parameters:
                margins = self.model.predict(X, output_margin=True)
            else:
                margins = None
        except Exception:
            margins = None

        # XGBoost booster fallback
        if margins is None and hasattr(self.model, "get_booster"):
            try:
                import xgboost as xgb
                dmat = xgb.DMatrix(X, feature_names=X.columns.tolist())
                margins = self.model.get_booster().predict(dmat, output_margin=True)
            except Exception:
                margins = None

        if margins is None:
            return None

        margins = np.asarray(margins)
        if margins.ndim == 1:
            return margins
        if margins.ndim == 2 and class_indices is not None:
            idx = np.arange(len(X))
            return margins[idx, class_indices]
        return None

    def _compute_shap_sums_and_baselines(
        self,
        explainer: Union[shap.Explainer, object],
        X: pd.DataFrame,
        candidates: Dict[str, Optional[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SHAP signed/abs sums and baselines, then select the best-matching target
        among margin/logit/prob by minimum reconstruction error.
        """
        values, bases = self._get_shap_values_and_bases(explainer, X)

        signed_sums = np.sum(values, axis=1)
        abs_sums = np.sum(np.abs(values), axis=1)
        bases = np.asarray(bases).reshape(-1)

        best_target = None
        best_mse = np.inf

        for key in ["margin", "logit", "prob"]:
            y = candidates.get(key)
            if y is None:
                continue
            y = np.asarray(y).reshape(-1)
            mse = np.nanmean((bases + signed_sums - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_target = y

        if best_target is None:
            best_target = np.asarray(candidates.get("prob"))

        return signed_sums, abs_sums, bases, best_target

    def _get_shap_values_and_bases(
        self,
        explainer: Union[shap.Explainer, object],
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return SHAP values (n_samples, n_features) for the predicted class (if multiclass)
        and per-sample baselines.
        Supports new API (explainer(X)) and older explainer.shap_values(X).
        """
        # New API
        try:
            ex = explainer(X)
            vals = np.asarray(ex.values)
            base_vals = np.asarray(ex.base_values)

            if vals.ndim == 3:
                y_pred = self.model.predict(X)
                model_classes = getattr(self.model, "classes_", None)
                if model_classes is not None and not np.issubdtype(np.asarray(y_pred).dtype, np.integer):
                    class_indices = np.array([int(np.where(model_classes == p)[0][0]) for p in y_pred])
                else:
                    class_indices = np.asarray(y_pred, dtype=int)

                values = np.stack([vals[i, :, class_indices[i]] for i in range(vals.shape[0])], axis=0)

                if base_vals.ndim == 2:
                    bases = base_vals[np.arange(base_vals.shape[0]), class_indices]
                elif base_vals.ndim == 1:
                    bases = base_vals[class_indices]
                else:
                    raise ValueError("Unsupported SHAP base_values shape for multiclass.")
            else:
                values = vals
                if np.ndim(base_vals) == 0:
                    bases = np.full(values.shape[0], float(base_vals))
                else:
                    bases = base_vals.reshape(-1)

            return values, bases
        except Exception:
            pass

        # Old API
        try:
            raw = explainer.shap_values(X)
        except Exception as e:
            raise RuntimeError(f"Could not compute SHAP values: {e}")

        expected_value = getattr(explainer, "expected_value", 0.0)

        if isinstance(raw, list):
            y_pred = self.model.predict(X)
            model_classes = getattr(self.model, "classes_", None)
            if model_classes is not None and not np.issubdtype(np.asarray(y_pred).dtype, np.integer):
                class_indices = np.array([int(np.where(model_classes == p)[0][0]) for p in y_pred])
            else:
                class_indices = np.asarray(y_pred, dtype=int)

            values = np.stack([raw[class_indices[i]][i] for i in range(len(X))], axis=0)

            if isinstance(expected_value, (list, np.ndarray)):
                bases = np.array([expected_value[class_indices[i]] for i in range(len(X))], dtype=float)
            else:
                bases = np.full(len(X), float(expected_value))
        else:
            values = np.asarray(raw)
            if isinstance(expected_value, (list, np.ndarray)):
                ev = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                bases = np.full(len(X), float(ev))
            else:
                bases = np.full(len(X), float(expected_value))

        return values, bases

    def _compute_lime_sums_and_baselines(
        self,
        explainer: LimeTabularExplainer,
        X: pd.DataFrame,
        class_indices: Optional[np.ndarray],
        num_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use LIME's local surrogate. For classification, target is probability space.
        Returns signed sums, abs sums, intercepts, and model outputs.
        """
        feature_names = list(X.columns)

        if self._is_classifier:
            predict_fn = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=feature_names))
            preds = self.model.predict(X)
            model_classes = getattr(self.model, "classes_", None)
            if class_indices is None:
                if model_classes is not None and not np.issubdtype(np.asarray(preds).dtype, np.integer):
                    class_indices = np.array([int(np.where(model_classes == p)[0][0]) for p in preds])
                else:
                    class_indices = np.asarray(preds, dtype=int)
            model_outputs = self.model.predict_proba(X)[np.arange(len(X)), class_indices]
        else:
            predict_fn = lambda x: self.model.predict(pd.DataFrame(x, columns=feature_names))
            class_indices = None
            model_outputs = self.model.predict(X).astype(float)

        def process_row(i: int, row: pd.Series) -> Tuple[float, float, float]:
            instance = row.values.astype(np.double)
            if class_indices is not None:
                label_idx = int(class_indices[i])
                exp = explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=len(instance),
                    num_samples=num_samples,
                    labels=[label_idx],
                )
                weights = exp.local_exp[label_idx]
                signed_sum = float(np.sum([w for _, w in weights])) if len(weights) else 0.0
                abs_sum = float(np.sum([abs(w) for _, w in weights])) if len(weights) else 0.0
                intercept = float(exp.intercept[label_idx])
            else:
                exp = explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=len(instance),
                    num_samples=num_samples,
                )
                k = list(exp.local_exp.keys())[0]
                weights = exp.local_exp[k]
                signed_sum = float(np.sum([w for _, w in weights])) if len(weights) else 0.0
                abs_sum = float(np.sum([abs(w) for _, w in weights])) if len(weights) else 0.0
                intercept = float(exp.intercept[k])
            return signed_sum, abs_sum, intercept

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(i, row) for i, (_, row) in enumerate(X.iterrows())
        )
        signed_sums, abs_sums, intercepts = map(np.asarray, zip(*results))
        return signed_sums, abs_sums, intercepts, np.asarray(model_outputs)






