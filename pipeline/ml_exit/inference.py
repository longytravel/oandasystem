"""
ML Exit Inference - Predict exit scores from trained models.

Takes trained MLExitModels and a feature DataFrame, returns per-row predictions
with hold_value, adverse_risk, and combined confidence scores.
"""
import numpy as np
import pandas as pd
from typing import Optional

from pipeline.ml_exit.features import get_feature_names


def predict_exit_scores(
    models,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate exit predictions from trained models.

    Args:
        models: MLExitModels (from train.py) with hold_value_model and adverse_risk_model.
        features_df: DataFrame with columns matching get_feature_names().

    Returns:
        DataFrame with columns:
        - hold_value_pred: predicted future R change (higher = more value in holding)
        - adverse_risk_pred: probability of hitting SL before TP (0-1)
        - confidence: combined confidence score (higher = more confident to exit)
    """
    feature_names = get_feature_names()

    missing = [f for f in feature_names if f not in features_df.columns]
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')

    X = features_df[feature_names].values.astype(np.float64)

    # Handle NaN/inf by replacing with 0
    finite_mask = np.isfinite(X)
    if not finite_mask.all():
        X = np.where(finite_mask, X, 0.0)

    # Predict hold value (regression)
    hold_value_pred = models.hold_value_model.predict(X)

    # Predict adverse risk (classification probability)
    adverse_risk_pred = models.adverse_risk_model.predict_proba(X)[:, 1]

    # Combined confidence: high risk AND low hold value = high exit confidence
    # Normalized to [0, 1] range
    # hold_value_pred is unbounded, so clip to [-3, 3] then normalize
    hold_norm = np.clip(hold_value_pred, -3.0, 3.0)
    hold_norm = (hold_norm - (-3.0)) / 6.0  # Map [-3,3] -> [0,1], where 0 = worst (exit)
    hold_exit_signal = 1.0 - hold_norm  # Invert: low hold value = high exit signal

    # Combine: average of risk and inverted hold value
    confidence = 0.5 * adverse_risk_pred + 0.5 * hold_exit_signal

    return pd.DataFrame({
        'hold_value_pred': hold_value_pred,
        'adverse_risk_pred': adverse_risk_pred,
        'confidence': confidence,
    }, index=features_df.index)
