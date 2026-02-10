"""
ML Exit Labeling - Supervised targets for the exit model.

For each (trade, decision_bar) pair, computes future-looking labels that serve
as regression and classification targets during training.

Labels intentionally use FUTURE data — they are what the model learns to predict.

Targets:
- future_r_change_5bar:  regression target for hold-value model
- future_r_change_10bar: same, longer horizon
- hit_sl_before_tp:      binary classification target for adverse-risk model
- optimal_exit_bar:      bar with highest unrealized R (for analysis)
- bars_to_exit:          bars remaining until actual trade exit
"""
import numpy as np
from typing import List, Dict


LABEL_NAMES = [
    'future_r_change_5bar',
    'future_r_change_10bar',
    'hit_sl_before_tp',
    'optimal_exit_bar',
    'bars_to_exit',
    'remaining_pnl_r',
]


def get_label_names() -> List[str]:
    """Return the ordered list of label names."""
    return list(LABEL_NAMES)


def _unrealized_r_at_bar(
    bar: int,
    closes: np.ndarray,
    entry_price: float,
    direction: int,
    initial_sl_dist: float,
) -> float:
    """Compute unrealized R at a specific bar."""
    safe_sl = max(initial_sl_dist, 1e-10)
    if direction == 1:
        return (closes[bar] - entry_price) / safe_sl
    else:
        return (entry_price - closes[bar]) / safe_sl


def compute_labels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    direction: int,
    entry_bar: int,
    exit_bar: int,
    current_bar: int,
    initial_sl_dist: float,
) -> Dict[str, float]:
    """
    Compute supervised labels for a single (trade, decision_bar) point.

    Uses future data from current_bar to exit_bar to compute targets.

    Args:
        highs: High prices array for entire dataset
        lows: Low prices array
        closes: Close prices array
        entry_price: Trade entry price
        sl_price: Stop loss price at entry
        tp_price: Take profit price
        direction: 1 for long, -1 for short
        entry_bar: Bar index of trade entry
        exit_bar: Bar index of trade exit (inclusive)
        current_bar: Bar index of the decision point
        initial_sl_dist: abs(entry_price - sl_price) at entry

    Returns:
        Dict with label values (see LABEL_NAMES for keys)
    """
    safe_sl = max(initial_sl_dist, 1e-10)

    # Edge case: trade exits on current bar
    if current_bar >= exit_bar:
        return {
            'future_r_change_5bar': 0.0,
            'future_r_change_10bar': 0.0,
            'hit_sl_before_tp': 0.0,
            'optimal_exit_bar': float(current_bar),
            'bars_to_exit': 0.0,
            'remaining_pnl_r': 0.0,
        }

    current_r = _unrealized_r_at_bar(
        current_bar, closes, entry_price, direction, initial_sl_dist
    )

    # --- future_r_change_5bar ---
    future_bar_5 = min(current_bar + 5, exit_bar)
    future_r_5 = _unrealized_r_at_bar(
        future_bar_5, closes, entry_price, direction, initial_sl_dist
    )
    future_r_change_5 = future_r_5 - current_r

    # --- future_r_change_10bar ---
    future_bar_10 = min(current_bar + 10, exit_bar)
    future_r_10 = _unrealized_r_at_bar(
        future_bar_10, closes, entry_price, direction, initial_sl_dist
    )
    future_r_change_10 = future_r_10 - current_r

    # --- hit_sl_before_tp ---
    # Check bars from current_bar+1 to exit_bar: does price hit SL before TP?
    hit_sl = 0.0
    for bar in range(current_bar + 1, exit_bar + 1):
        if direction == 1:  # Long
            if lows[bar] <= sl_price:
                hit_sl = 1.0
                break
            if highs[bar] >= tp_price:
                break  # hit TP first
        else:  # Short
            if highs[bar] >= sl_price:
                hit_sl = 1.0
                break
            if lows[bar] <= tp_price:
                break  # hit TP first

    # --- optimal_exit_bar ---
    # Bar with highest unrealized R from entry to exit
    best_r = -1e10
    best_bar = entry_bar
    for bar in range(entry_bar, exit_bar + 1):
        r = _unrealized_r_at_bar(bar, closes, entry_price, direction, initial_sl_dist)
        if r > best_r:
            best_r = r
            best_bar = bar

    # --- bars_to_exit ---
    bars_remaining = exit_bar - current_bar

    # --- remaining_pnl_r (V2 optimal stopping label) ---
    # Value of holding from current_bar to exit: final_r - current_r
    # Positive = holding improves trade, Negative = holding makes it worse
    final_r = _unrealized_r_at_bar(
        exit_bar, closes, entry_price, direction, initial_sl_dist
    )
    remaining_pnl = final_r - current_r

    return {
        'future_r_change_5bar': future_r_change_5,
        'future_r_change_10bar': future_r_change_10,
        'hit_sl_before_tp': hit_sl,
        'optimal_exit_bar': float(best_bar),
        'bars_to_exit': float(bars_remaining),
        'remaining_pnl_r': remaining_pnl,
    }
