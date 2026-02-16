"""
ML Exit Policy - Convert model predictions into actionable exit signals.

Two functions:
1. apply_exit_policy(): Converts predictions into per-row exit decisions
2. generate_ml_score_arrays(): Converts exit decisions into (n_bars,) arrays
   for the numba backtest engine
"""
import numpy as np
import pandas as pd
from typing import Tuple


def apply_exit_policy(
    predictions: pd.DataFrame,
    hold_value_threshold: float = 0.0,
    adverse_risk_threshold: float = 0.5,
    min_confidence: float = 0.3,
    policy_mode: str = 'dual_model',
) -> np.ndarray:
    """
    Apply exit policy rules to model predictions.

    Three policy modes:
    - dual_model: EXIT when hold_value AND adverse_risk both signal (original)
    - risk_only: EXIT when adverse_risk signals (ignores hold_value model)
    - hold_only: EXIT when hold_value signals (ignores adverse_risk model)

    All modes require minimum confidence threshold.

    The output is a binary score per row:
    - 1.0 means "exit"
    - 0.0 means "hold"

    Args:
        predictions: DataFrame with columns:
            - hold_value_pred: predicted future R change
            - adverse_risk_pred: P(hit SL before TP)
            - confidence: combined confidence (0-1)
        hold_value_threshold: Exit if hold_value < this (default 0.0)
        adverse_risk_threshold: Exit if adverse_risk > this (default 0.5)
        min_confidence: Minimum confidence to act on prediction (default 0.3)
        policy_mode: 'dual_model', 'risk_only', or 'hold_only'

    Returns:
        np.ndarray of shape (n_rows,) with binary exit scores (1.0 = exit, 0.0 = hold)
    """
    hold_value = predictions['hold_value_pred'].values
    adverse_risk = predictions['adverse_risk_pred'].values
    confidence = predictions['confidence'].values

    if policy_mode == 'risk_only':
        exit_mask = (adverse_risk > adverse_risk_threshold) & (confidence >= min_confidence)
    elif policy_mode == 'hold_only':
        exit_mask = (hold_value < hold_value_threshold) & (confidence >= min_confidence)
    else:  # dual_model (default, current behavior)
        exit_mask = (
            (hold_value < hold_value_threshold)
            & (adverse_risk > adverse_risk_threshold)
            & (confidence >= min_confidence)
        )

    # Binary output: policy is the single decision authority (Fix 4)
    scores = np.where(exit_mask, 1.0, 0.0)
    return scores


def generate_ml_score_arrays(
    policy_scores: np.ndarray,
    bar_indices: np.ndarray,
    n_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map per-decision-point policy scores into per-bar arrays for the numba engine.

    The numba backtest engine reads ml_long_scores[bar] and ml_short_scores[bar]
    to decide whether to exit. This function fills two (n_bars,) arrays where
    each bar's score is the maximum policy score assigned to it.

    Since the policy already accounts for direction (via model predictions),
    both long and short arrays receive the same scores.

    Args:
        policy_scores: Per-decision-point scores from apply_exit_policy()
        bar_indices: Bar index corresponding to each score entry
        n_bars: Total number of bars in the dataset

    Returns:
        (ml_long_scores, ml_short_scores) each of shape (n_bars,) float64
    """
    ml_long_scores = np.zeros(n_bars, dtype=np.float64)
    ml_short_scores = np.zeros(n_bars, dtype=np.float64)

    for i in range(len(policy_scores)):
        bar = int(bar_indices[i])
        if 0 <= bar < n_bars:
            score = policy_scores[i]
            # Take max if multiple decisions map to same bar
            if score > ml_long_scores[bar]:
                ml_long_scores[bar] = score
            if score > ml_short_scores[bar]:
                ml_short_scores[bar] = score

    return ml_long_scores, ml_short_scores
