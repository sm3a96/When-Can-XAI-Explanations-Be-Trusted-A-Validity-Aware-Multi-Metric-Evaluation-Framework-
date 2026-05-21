"""
Actionability metric for XAI explanations

 this fitted classifier exposing predict and predict_proba
This codce used for explainer methdos like (LIME or SHAP explainer)

also and imporntat thing (X) pandas dataframe with the same columns used to train the model
the actionable features are list of feature you can back to our paper to see more about it

Output:
- Either a float mean score or a dict with:
  {'mean_score','std_score','median_score','instance_scores','feature_counts'}

"""


import numpy as np
import pandas as pd


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches

import warnings
import logging
warnings.filterwarnings('ignore')


# The visualization setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.figsize'] = (12, 8)

logger = logging.getLogger("XAIActionabilityEvaluator")
logging.basicConfig(level=logging.INFO)

class XAIActionabilityEvaluator:
    def __init__(self, model: Any):
        self.model = model
        if not callable(getattr(self.model, 'predict', None)):
            raise ValueError("Model must have a callable 'predict' method.")
        if not callable(getattr(self.model, 'predict_proba', None)):
            warnings.warn("Model doesn't have 'predict_proba' method, some XAI methods may fail.")


# Evaluate actionability 
    def evaluate_actionability(
            self,
            explainer: Any,
            X: pd.DataFrame,
            method: str = 'lime',
            actionable_features: Optional[List[str]] = None,
            top_k: Optional[int] = None,
            verbose: bool = False,
            sample_size: Optional[int] = None,
            random_state: int = 42,
            return_details: bool = False  # <-- Added parameter
        ) -> Union[float, Dict[str, Any]]: 
            """
            explainer is the object for LIME or SHAP .
            X is thje  pd dataframe from the dataset.
            
            actionable_features is List of feature names that are actionable.
            top_k show the number of top features to consider.

            Returns
            float or Dict[str, Any]
            """
            if actionable_features is None or not actionable_features:
                raise ValueError("You must provide a non-empty list of actionable feature names.")
            if not isinstance(X, pd.DataFrame) or X.empty:
                raise ValueError("Input X must be a non-empty pandas DataFrame.")
            invalid_features = [f for f in actionable_features if f not in X.columns]
            if invalid_features:
                raise ValueError(f"Actionable features not in dataset: {invalid_features}")
            if top_k is None:
                top_k = min(10, X.shape[1])
            elif top_k > X.shape[1]:
                warnings.warn(f"top_k ({top_k}) > number of features ({X.shape[1]}). Setting top_k to {X.shape[1]}.")
                top_k = X.shape[1]
            if sample_size is not None and sample_size < len(X):
                X = X.sample(sample_size, random_state=random_state)
                if verbose:
                    logger.info(f"Using a sample of {sample_size} instances for evaluation.")

            actionability_scores: List[float] = []
            feature_occurrence: Dict[str, int] = {f: 0 for f in X.columns}
            actionable_occurrence: Dict[str, int] = {f: 0 for f in actionable_features}

            for idx, (_, instance) in enumerate(X.iterrows()):
                instance_df = instance.to_frame().T
                try:
                    feature_importances = self._get_feature_importances(
                        instance_df, explainer, method, verbose=(verbose and idx == 0)
                    )
                    if feature_importances is None:
                        if verbose and idx < 5:
                            logger.warning(f"Failed to get explanation for instance {idx}")
                        continue
                    top_features = self._get_top_features(
                        feature_importances, top_k, list(instance_df.columns)
                    )
                    for feature, _ in top_features:
                        feature_occurrence[feature] += 1
                        if feature in actionable_features:
                            actionable_occurrence[feature] += 1
                    actionability_score = self._compute_actionability_score(
                        top_features, actionable_features
                    )
                    actionability_scores.append(actionability_score)
                    if verbose and idx < 3:
                        logger.info(f"Instance {idx} actionability: {actionability_score:.4f}")
                        logger.info(f"Top {min(5, len(top_features))} features: {top_features[:5]}")
                        logger.info("-" * 40)
                except Exception as e:
                    if verbose:
                        logger.error(f"Error processing instance {idx}: {str(e)}")
                    continue

            if not actionability_scores:
                raise RuntimeError("Could not compute actionability scores for any instances.")

            if verbose:
                self._print_summary(feature_occurrence, actionable_occurrence, actionability_scores, actionable_features)

            if return_details:
                # Return detailed results as a dictionary
                return {
                    'mean_score': float(np.mean(actionability_scores)),
                    'instance_scores': np.array(actionability_scores),
                    'feature_counts': feature_occurrence,
                    'actio*nable_counts': actionable_occurrence,
                    'top_features': sorted(feature_occurrence.items(), key=lambda x: x[1], reverse=True)[:20]
                }
            return float(np.mean(actionability_scores))


    def _get_feature_importances(
        self,
        instance: pd.DataFrame,
        explainer: Any,
        method: str,
        verbose: bool = False
    ) -> Optional[Dict[str, float]]:
        method = method.lower()
        if method == 'lime':
            return self._get_lime_feature_importances(instance, explainer, verbose)
        elif method == 'shap':
            return self._get_shap_feature_importances(instance, explainer, verbose)
        else:
            raise ValueError(f"Unsupported XAI method: {method}. Use 'lime' or 'shap'.")

    def _get_lime_feature_importances(
        self,
        instance: pd.DataFrame,
        explainer: Any,
        verbose: bool = False
    ) -> Optional[Dict[str, float]]:
        try:
            explanation = explainer.explain_instance(
                instance.values[0],
                self.model.predict_proba,
                num_features=instance.shape[1]
            )
            feature_importances: Dict[str, float] = {}
            for feature_condition, importance in explanation.as_list():
                feature_name = self._clean_lime_feature_name(feature_condition)
                if feature_name in feature_importances:
                    if abs(importance) > abs(feature_importances[feature_name]):
                        feature_importances[feature_name] = importance
                else:
                    feature_importances[feature_name] = importance
            if verbose:
                logger.info(f"LIME identified {len(feature_importances)} features with non-zero importance.")
            return feature_importances
        except Exception as e:
            if verbose:
                logger.error(f"LIME explanation error: {str(e)}")
            return None

    def _clean_lime_feature_name(self, feature_condition: str) -> str:
        for delimiter in [" <= ", " > ", " < ", " >= "]:
            if delimiter in feature_condition:
                return feature_condition.split(delimiter)[0].strip()
        return feature_condition.strip()

    def _get_shap_feature_importances(
        self,
        instance: pd.DataFrame,
        explainer: Any,
        verbose: bool = False
    ) -> Optional[Dict[str, float]]:
        try:
            shap_values = explainer.shap_values(instance)
            feature_values = None
            if isinstance(shap_values, list):
                pred_class = np.argmax(self.model.predict_proba(instance))
                if pred_class < len(shap_values):
                    feature_values = shap_values[pred_class][0]
                else:
                    sums = [np.sum(np.abs(s[0])) for s in shap_values]
                    max_class = np.argmax(sums)
                    feature_values = shap_values[max_class][0]
            elif isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) > 2:
                    feature_values = shap_values[0][0]
                else:
                    feature_values = shap_values[0]
            if feature_values is None:
                if verbose:
                    logger.warning(f"Unrecognized SHAP values format: {type(shap_values)}")
                return None
            feature_importances = {
                feature_name: feature_values[idx]
                for idx, feature_name in enumerate(instance.columns)
                if idx < len(feature_values)
            }
            if verbose:
                logger.info(f"SHAP identified {len(feature_importances)} features.")
            return feature_importances
        except Exception as e:
            if verbose:
                logger.error(f"SHAP explanation error: {str(e)}")
            return None

    def _get_top_features(
        self,
        feature_importances: Dict[str, float],
        top_k: int,
        feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        valid_importances = {
            f: imp for f, imp in feature_importances.items()
            if f in feature_names
        }
        sorted_features = sorted(
            valid_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:min(top_k, len(sorted_features))]

    def _compute_actionability_score(
        self,
        top_features: List[Tuple[str, float]],
        actionable_features: List[str]
    ) -> float:
        if not top_features:
            return 0.0
        _, importance_values = zip(*top_features)
        abs_importance = np.abs(importance_values)
        total_importance = np.sum(abs_importance)
        if total_importance == 0:
            return 0.0
        weighted_score = sum(
            abs(importance) / total_importance
            for feature, importance in top_features
            if feature in actionable_features
        )
        return float(weighted_score)

    def _print_summary(
        self,
        feature_occurrence: Dict[str, int],
        actionable_occurrence: Dict[str, int],
        actionability_scores: List[float],
        actionable_features: List[str]
    ):
        print("\n===== ACTIONABILITY EVALUATION SUMMARY =====")
        print("\nMost frequently important features:")
        for feature, count in sorted(
            feature_occurrence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            is_actionable = "Ok" if feature in actionable_features else "Not Ok"
            print(f"  {feature}: {count} times {is_actionable}")

        print("\nMost frequently important actionable features:")
        for feature, count in sorted(
            actionable_occurrence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            if count > 0:
                print(f"  {feature}: {count} times")

        print("\nActionability score distribution:")
        print(f"  Mean:   {np.mean(actionability_scores):.4f}")
        print(f"  Median: {np.median(actionability_scores):.4f}")
        print(f"  Min:    {np.min(actionability_scores):.4f}")
        print(f"  Max:    {np.max(actionability_scores):.4f}")
        print(f"  Std:    {np.std(actionability_scores):.4f}")

        mean_score = np.mean(actionability_scores)
        print("\nInterpretation:")
        if mean_score < 0.3:
            print("  Low actionability: Explanations mostly focus on non-actionable features")
        elif mean_score > 0.7:
            print("  High actionability: Explanations suggest features that can be reasonably modified")
        else:
            print("  Medium actionability: Explanations partially focus on actionable features")

    @staticmethod
    def visualize_actionability_results(
        lime_results,
        shap_results,
        actionable_features,
        model_name,
        y_pred=None,
        class_names=None,
        compact=True,           
        top_n=10,               
        figsize=(10, 6),        
 
    ):
        """
        Create actionability visualization.
        If compact=True (default), produces a 2x2 grid suitable for IEEE S&P:
          - Histogram (LIME vs SHAP distributions)
          - Correlation scatter
          - Top-N features (stacked bars, hatched for actionable)
          - Per-class heatmap (limited to top 10 classes)
        """
        lime_scores = np.asarray(lime_results['instance_scores'], dtype=float)
        shap_scores = np.asarray(shap_results['instance_scores'], dtype=float)
        lime_feature_counts = lime_results.get('feature_counts', {})
        shap_feature_counts = shap_results.get('feature_counts', {})
        lime_mean = float(lime_results['mean_score'])
        shap_mean = float(shap_results['mean_score'])
     

        if not compact:
          
            pass

        # 2x2 grid
        sns.set_context('paper', font_scale=1.0)
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        # Panel A ==> Histogram
        bins = np.linspace(0, 1, 21)
        axes[0, 0].hist(lime_scores, bins=bins, alpha=0.7, label='LIME', color='#3498db', edgecolor='k', linewidth=0.5)
        axes[0, 0].hist(shap_scores, bins=bins, alpha=0.7, label='SHAP', color='#e74c3c', edgecolor='k', linewidth=0.5)
        axes[0, 0].set_xlabel('Actionability Score', fontsize=9)
        axes[0, 0].set_ylabel('Count', fontsize=9)
        axes[0, 0].set_title('Distribution', fontsize=10)
        axes[0, 0].legend(fontsize=8, frameon=False)
        axes[0, 0].tick_params(labelsize=8)

        # Panel B ==> Correlation 
        n = min(len(lime_scores), len(shap_scores))
        x, y = lime_scores[:n], shap_scores[:n]
        r_corr, p_corr = stats.pearsonr(x, y)
        axes[0, 1].scatter(x, y, alpha=0.4, s=10, color='#2ecc71', edgecolor='none')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1)
        axes[0, 1].set_xlim(0, 1); axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xlabel('LIME', fontsize=9)
        axes[0, 1].set_ylabel('SHAP', fontsize=9)
        axes[0, 1].set_title(f'Correlation: r={r_corr:.3f}', fontsize=10)
        axes[0, 1].tick_params(labelsize=8)

        # Panel C ==> TopN features 
        feature_importances = {}
        for feat, c in lime_feature_counts.items():
            feature_importances[feat] = {
                'lime': c, 'shap': shap_feature_counts.get(feat, 0),
                'is_actionable': feat in actionable_features
            }
        for feat, c in shap_feature_counts.items():
            feature_importances.setdefault(feat, {'lime': 0, 'shap': 0, 'is_actionable': feat in actionable_features})
            feature_importances[feat]['shap'] = c

        fi_df = (pd.DataFrame.from_dict(feature_importances, orient='index')
                   .assign(total=lambda d: d['lime'] + d['shap'])
                   .sort_values('total', ascending=False)
                   .head(top_n)
                   .sort_values('total'))  

        y_pos = np.arange(len(fi_df))
        bar_width = 0.35
        bars_lime = axes[1, 0].barh(y_pos - bar_width/2, fi_df['lime'], bar_width, color='#3498db', label='LIME')
        bars_shap = axes[1, 0].barh(y_pos + bar_width/2, fi_df['shap'], bar_width, color='#e74c3c', label='SHAP')

        # Hatch actionable features
        for j, (feat, row) in enumerate(fi_df.iterrows()):
            if bool(row['is_actionable']):
                bars_lime[j].set_hatch('///')
                bars_shap[j].set_hatch('///')

        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(fi_df.index, fontsize=7)
        axes[1, 0].set_xlabel('Count', fontsize=9)
        axes[1, 0].set_title(f'Top {top_n} features', fontsize=10)
        axes[1, 0].tick_params(axis='x', labelsize=8)
        #  legend
        actionable_patch = mpatches.Patch(facecolor='lightgrey', hatch='///', edgecolor='k', label='Actionable')
        handles = [mpatches.Patch(color='#3498db', label='LIME'),
                   mpatches.Patch(color='#e74c3c', label='SHAP'),
                   actionable_patch]
        axes[1, 0].legend(handles=handles, fontsize=7, frameon=False, loc='lower right')

        # Panel D ==> Per-class heatmap 
        if y_pred is not None and class_names is not None:
            unique = np.unique(y_pred)
            if len(unique) > 10:
                from collections import Counter
                top_classes = [c for c, _ in Counter(y_pred).most_common(10)]
            else:
                top_classes = list(unique)

            class_data = []
            for c in top_classes:
                idx = np.where(y_pred == c)[0]
                if len(idx) == 0:
                    continue
                class_name = class_names[c] if c < len(class_names) else str(c)
                lime_val = float(np.mean(lime_scores[idx[idx < len(lime_scores)]])) if len(idx) else np.nan
                shap_val = float(np.mean(shap_scores[idx[idx < len(shap_scores)]])) if len(idx) else np.nan
                class_data.append({'Class': class_name, 'LIME': lime_val, 'SHAP': shap_val})

            if class_data:
                cdf = pd.DataFrame(class_data).set_index('Class')[['LIME', 'SHAP']].T
                sns.heatmap(cdf, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True,
                            vmin=0, vmax=1, ax=axes[1, 1], annot_kws={'fontsize': 7})
                axes[1, 1].set_title('By class (top 10)', fontsize=10)
                axes[1, 1].tick_params(labelsize=7)
            else:
                axes[1, 1].axis('off')
        else:
            axes[1, 1].axis('off')

        fig.suptitle(f'XAI Actionability – {model_name}', fontsize=11, y=0.99)

        return fig







# #     def _get_feature_importances(
# #         self,
# #         instance: pd.DataFrame,
# #         explainer: Any,
# #         method: str,
# #         verbose: bool = False
# #     ) -> Optional[Dict[str, float]]:
# #         method = method.lower()
# #         if method == 'lime':
# #             return self._get_lime_feature_importances(instance, explainer, verbose)
# #         elif method == 'shap':
# #             return self._get_shap_feature_importances(instance, explainer, verbose)
# #         else:
# #             raise ValueError(f"Unsupported XAI method: {method}. Use 'lime' or 'shap'.")





# #     def _get_lime_feature_importances(
# #         self,
# #         instance: pd.DataFrame,
# #         explainer: Any,
# #         verbose: bool = False
# #     ) -> Optional[Dict[str, float]]:
# #         try:
# #             # Wrap predict_proba to ensure consistent DataFrame columns ordering for models that expect DataFrame
# #             def predict_fn(X):
# #                 X_df = pd.DataFrame(X, columns=instance.columns)
# #                 return self.model.predict_proba(X_df)

# #             explanation = explainer.explain_instance(
# #                 instance.values[0],
# #                 predict_fn,
# #                 num_features=instance.shape[1]
# #             )
            
# #             feature_importances: Dict[str, float] = {}
# #             feature_names = instance.columns.tolist()
            
# #             # Parse LIME condition strings into feature names robustly
# #             for feature_condition, importance in explanation.as_list():
# #                 feature_name = self._clean_lime_feature_name(feature_condition, feature_names)
                
# #                 if feature_name in feature_importances:
# #                     # Keep the magnitude-dominant importance per feature
# #                     if abs(importance) > abs(feature_importances[feature_name]):
# #                         feature_importances[feature_name] = float(importance)
# #                 else:
# #                     feature_importances[feature_name] = float(importance)
            
# #             if verbose:
# #                 logger.info(f"LIME identified {len(feature_importances)} features with non-zero importance.")
# #                 non_zero = sum(1 for v in feature_importances.values() if abs(v) > 1e-10)
# #                 logger.info(f"  Non-zero importances: {non_zero}")
            
# #             return feature_importances
            
# #         except Exception as e:
# #             if verbose:
# #                 logger.error(f"LIME explanation error: {str(e)}")
# #             return None



# #     def _compute_actionability_score(
# #         self,
# #         top_features: List[Tuple[str, float]],
# #         actionable_features: List[str]
# #     ) -> float:
# #         """
# #         Score = (sum of absolute importance of actionable features) / (sum of all absolute importance)
# #         Guaranteed in [0, 1] (clipped to protect against rare numeric issues).
# #         """
# #         if not top_features:
# #             return 0.0
        
# #         total_importance = float(sum(abs(imp) for _, imp in top_features))
# #         if not np.isfinite(total_importance) or total_importance < 1e-12:
# #             return 0.0
        
# #         actionable_importance = float(sum(
# #             abs(importance) 
# #             for feature, importance in top_features
# #             if feature in actionable_features
# #         ))
        
# #         score = actionable_importance / total_importance




