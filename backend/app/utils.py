import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

logger = logging.getLogger(__name__)


def compute_uncertainty(model, X_df: pd.DataFrame) -> float:
    """
    For tree ensemble models like RandomForest, compute std across trees' predictions as a proxy for uncertainty.
    Returns a single float (std dev) per-row; here we return mean of row stds as a simple confidence measure.
    """
    try:
        # If pipeline, extract reg and preprocessor
        preprocessed = None
        if hasattr(model, 'named_steps') and 'reg' in model.named_steps:
            reg = model.named_steps['reg']
            preproc = model.named_steps.get('preprocessor')
            if preproc is not None:
                preprocessed = preproc.transform(X_df)
        else:
            reg = model

        if hasattr(reg, 'estimators_'):
            # For ensembles, get per-estimator predictions on transformed data
            X_for_pred = preprocessed if preprocessed is not None else X_df
            preds = []
            for estimator in reg.estimators_:
                try:
                    preds.append(estimator.predict(X_for_pred))
                except Exception:
                    # for some implementations (GradientBoosting), estimators_ is 2d array
                    try:
                        preds.append(estimator[0].predict(X_for_pred))
                    except Exception:
                        preds.append(np.zeros((len(X_df),)))
            preds = np.stack(preds, axis=1)
            row_std = preds.std(axis=1)
            return float(np.mean(row_std))
    except Exception as e:
        logger.warning("Uncertainty computation failed: %s", e)
    return None


def explain_prediction(model, X_df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float, float]]:
    """
    Return top_k contributing features as tuples (feature_name, value, contribution)
    If SHAP is available and model is tree-based, use TreeExplainer.
    Otherwise, use permutation of model feature importance or coefficients when available.
    """
    cols = X_df.columns.tolist()
    # Try SHAP explanations using the whole pipeline (SHAP supports pipelines in many cases)
    if _HAS_SHAP:
        try:
            explainer = shap.Explainer(model, X_df, algorithm='auto')
            shap_vals = explainer(X_df)
            # try to map shap values back to original features by averaging over one-hot groups if needed
            # Here we simply compute mean absolute shap across the sample axis and choose top_k indices
            mean_abs = np.abs(shap_vals.values).mean(axis=0)
            # If shap returns values matching the input columns length, map directly
            if mean_abs.shape[0] == len(cols):
                idx = np.argsort(mean_abs)[-top_k:][::-1]
                out = []
                for i in idx:
                    out.append((cols[i], float(X_df.iloc[0, i]), float(np.sign(shap_vals.values[0, i]) * mean_abs[i])))
                return out
            # Otherwise, fall back to perturbation method below
        except Exception as e:
            logger.warning('SHAP explanation failed: %s', e)

    # Fallback: perturb each original feature one at a time and measure change in prediction
    try:
        base_pred = float(model.predict(X_df)[0])
        contributions = []
        for c in cols:
            X2 = X_df.copy()
            # replace with median/typical value
            try:
                med = X2[c].median()
            except Exception:
                med = 0
            X2[c] = med
            try:
                p = float(model.predict(X2)[0])
                contrib = base_pred - p
                contributions.append((c, float(X_df.iloc[0][c]), contrib))
            except Exception:
                continue
        contributions = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)[:top_k]
        return contributions
    except Exception as e:
        logger.warning('Perturbation fallback failed: %s', e)

    # as a final fallback return the largest absolute raw inputs
    vals = X_df.iloc[0].abs().sort_values(ascending=False).head(top_k)
    return [(c, float(X_df.iloc[0][c]), float(vals[c])) for c in vals.index]
