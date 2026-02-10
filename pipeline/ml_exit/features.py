"""
ML Exit Feature Engineering.

Computes 16 features per (trade, bar) decision point for the ML exit model.
ALL features use data at or before the decision bar (no look-ahead bias).

Three feature groups:
- Trade-state features (9): position-specific metrics (unrealized R, MFE/MAE, drawdown, age ratio, etc.)
- Market features (5): ATR, trend slope, momentum — all normalized
- Session features (2): hour_of_day, day_of_week

Uses numpy (not numba) since this runs once per dataset build, not in a hot loop.
"""
import numpy as np
from typing import List, Dict, Optional


# --- Feature names (must match compute_decision_features output order) ---

TRADE_STATE_FEATURES = [
    'direction',           # 1=long, -1=short
    'age_bars',            # bars since entry
    'unrealized_r',        # (close - entry) / sl_dist, sign-adjusted
    'distance_to_sl_r',    # abs(close - current_sl) / initial_sl_dist
    'distance_to_tp_r',    # abs(tp - close) / initial_sl_dist
    'mfe_r_running',       # running max favorable excursion in R
    'mae_r_running',       # running max adverse excursion in R
    'mfe_drawdown_r',      # V2: mfe_r - unrealized_r (pullback from peak)
    'age_ratio',           # V2: age_bars / max_hold_bars (trade progress 0-1)
]

MARKET_FEATURES = [
    'atr_norm',            # current ATR / entry ATR (vol expansion/contraction)
    'trend_slope_short',   # (close - close[5 bars ago]) / ATR, clipped [-3,3]
    'trend_slope_long',    # (close - close[20 bars ago]) / ATR, clipped [-3,3]
    'momentum_short',      # RSI change over 3 bars / 20, clipped [-1,1]
    'momentum_long',       # RSI change over 10 bars / 20, clipped [-1,1]
]

SESSION_FEATURES = [
    'hour_of_day',         # 0-23
    'day_of_week',         # 0-4 (Mon-Fri)
]

ALL_FEATURE_NAMES = TRADE_STATE_FEATURES + MARKET_FEATURES + SESSION_FEATURES
N_FEATURES = len(ALL_FEATURE_NAMES)  # 16 (9 trade + 5 market + 2 session)


def get_feature_names() -> List[str]:
    """Return the ordered list of 16 feature names."""
    return list(ALL_FEATURE_NAMES)


def precompute_market_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Pre-calculate market-level indicators for the entire dataset.

    Computes RSI(14), ATR(14), and stores closes for later lookback.
    Called once per dataset — individual bar lookups are O(1) after this.

    Args:
        highs: High prices, shape (n_bars,)
        lows: Low prices, shape (n_bars,)
        closes: Close prices, shape (n_bars,)
        opens: Open prices (unused, reserved for future features)

    Returns:
        Dict with pre-computed arrays:
        - 'rsi': RSI(14) array, shape (n_bars,)
        - 'atr': ATR(14) array, shape (n_bars,)
        - 'closes': reference to closes array
    """
    n = len(closes)

    # --- RSI(14) via Wilder's smoothing ---
    rsi_period = 14
    rsi = np.full(n, 50.0, dtype=np.float64)

    if n > rsi_period + 1:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:rsi_period])
        avg_loss = np.mean(losses[:rsi_period])

        if avg_loss > 0:
            rsi[rsi_period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[rsi_period] = 100.0

        for i in range(rsi_period, n - 1):
            avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
            if avg_loss > 0:
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            else:
                rsi[i + 1] = 100.0

    # --- ATR(14) via Wilder's smoothing ---
    atr_period = 14
    atr = np.zeros(n, dtype=np.float64)

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    if n > atr_period:
        atr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    return {
        'rsi': rsi,
        'atr': atr,
        'closes': closes,
    }


def compute_decision_features(
    trade_state: dict,
    market_data: dict,
    bar_idx: int,
) -> np.ndarray:
    """
    Compute the 16-feature vector for a single (trade, bar) decision point.

    Args:
        trade_state: Dict with trade-specific info:
            - direction: int (1=long, -1=short)
            - entry_bar: int (bar index of entry)
            - entry_price: float
            - sl_price: float (current stop loss)
            - tp_price: float (take profit)
            - initial_sl_dist: float (abs(entry - original_sl))
            - mfe_r: float (running MFE in R-multiples)
            - mae_r: float (running MAE in R-multiples)
            - hour: int (0-23, hour of decision bar)
            - day_of_week: int (0-4, day of decision bar)
        market_data: Dict from precompute_market_features():
            - rsi: array (n_bars,)
            - atr: array (n_bars,)
            - closes: array (n_bars,)
        bar_idx: Current bar index

    Returns:
        np.ndarray of shape (16,), dtype float64
    """
    features = np.zeros(N_FEATURES, dtype=np.float64)

    direction = trade_state['direction']
    entry_bar = trade_state['entry_bar']
    entry_price = trade_state['entry_price']
    sl_price = trade_state['sl_price']
    tp_price = trade_state['tp_price']
    initial_sl_dist = trade_state['initial_sl_dist']

    closes = market_data['closes']
    rsi = market_data['rsi']
    atr = market_data['atr']

    close = closes[bar_idx]
    cur_atr = max(atr[bar_idx], 1e-10)
    safe_sl_dist = max(initial_sl_dist, 1e-10)

    # === Trade-state features (indices 0-6) ===

    # 0: direction
    features[0] = float(direction)

    # 1: age_bars
    features[1] = float(bar_idx - entry_bar)

    # 2: unrealized_r — sign-adjusted PnL in R-multiples
    if direction == 1:
        features[2] = (close - entry_price) / safe_sl_dist
    else:
        features[2] = (entry_price - close) / safe_sl_dist

    # 3: distance_to_sl_r
    features[3] = abs(close - sl_price) / safe_sl_dist

    # 4: distance_to_tp_r
    features[4] = abs(tp_price - close) / safe_sl_dist

    # 5: mfe_r_running (passed in from caller who tracks it bar-by-bar)
    features[5] = trade_state['mfe_r']

    # 6: mae_r_running
    features[6] = trade_state['mae_r']

    # 7: mfe_drawdown_r (V2) — how far pulled back from peak
    mfe_r = trade_state['mfe_r']
    unrealized_r = features[2]
    features[7] = max(mfe_r - unrealized_r, 0.0)

    # 8: age_ratio (V2) — trade progress (0-1)
    max_hold = trade_state.get('max_hold_bars', 0)
    age = float(bar_idx - entry_bar)
    features[8] = min(age / max(max_hold, 1), 1.0) if max_hold > 0 else 0.0

    # === Market features (indices 9-13) ===

    # 9: atr_norm — current ATR / ATR at entry
    entry_atr = max(atr[entry_bar], 1e-10)
    features[9] = cur_atr / entry_atr

    # 10: trend_slope_short — (close - close[5 ago]) / ATR, clipped [-3, 3]
    if bar_idx >= 5:
        val = (close - closes[bar_idx - 5]) / cur_atr
        features[10] = np.clip(val, -3.0, 3.0)

    # 11: trend_slope_long — (close - close[20 ago]) / ATR, clipped [-3, 3]
    if bar_idx >= 20:
        val = (close - closes[bar_idx - 20]) / cur_atr
        features[11] = np.clip(val, -3.0, 3.0)

    # 12: momentum_short — RSI change over 3 bars / 20, clipped [-1, 1]
    if bar_idx >= 3:
        val = (rsi[bar_idx] - rsi[bar_idx - 3]) / 20.0
        features[12] = np.clip(val, -1.0, 1.0)

    # 13: momentum_long — RSI change over 10 bars / 20, clipped [-1, 1]
    if bar_idx >= 10:
        val = (rsi[bar_idx] - rsi[bar_idx - 10]) / 20.0
        features[13] = np.clip(val, -1.0, 1.0)

    # === Session features (indices 14-15) ===

    # 14: hour_of_day
    features[14] = float(trade_state['hour'])

    # 15: day_of_week
    features[15] = float(trade_state['day_of_week'])

    return features
