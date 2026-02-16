"""
ML Feature Computation for V6 EMA Cross + ML Exit Strategy.

Pre-computes 8 per-bar market features from OHLC data, then applies
Optuna-optimized weights to produce direction-aware exit scores.

Architecture:
- compute_ml_features(): @njit, returns (n_bars, 8) float64 array
- compute_ml_scores(): numpy, returns (long_scores, short_scores) each (n_bars,)

Features are computed ONCE per dataset. Scores are recomputed per trial
(cheap vectorized multiply). Numba reads one score per bar — simple lookup.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def compute_ml_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
) -> np.ndarray:
    """
    Compute 8 per-bar market features from OHLC data.

    All backward-looking only (no look-ahead bias).
    First ~200 bars have warm-up artifacts — filled with 0.

    Features:
        0: rsi_momentum   - RSI(14)[i] - RSI(14)[i-5], /20       [-1,1]
        1: atr_ratio       - ATR(14)/SMA(ATR,50) - 1.0            [-2,2]
        2: range_position  - (close - 20bar_low)/(20bar_high - 20bar_low)  [0,1]
        3: trend_alignment - sign(MA50-MA200) * min(1, |diff|/ATR) [-1,1]
        4: momentum_exh    - (RSI-50)/50                           [-1,1]
        5: bar_rejection   - 1 - |close-open|/(high-low+1e-10)    [0,1]
        6: cc_momentum     - (close[i]-close[i-3])/ATR             [-3,3]
        7: range_contract  - (high-low)/SMA(high-low,20)           [0.3,3]

    Args:
        highs: High prices array
        lows: Low prices array
        closes: Close prices array
        opens: Open prices array

    Returns:
        (n_bars, 8) float64 array of features
    """
    n = len(closes)
    features = np.zeros((n, 8), dtype=np.float64)

    # === Pre-compute RSI(14) ===
    rsi_period = 14
    rsi = np.full(n, 50.0, dtype=np.float64)

    if n > rsi_period + 1:
        # Initial average gain/loss
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(1, rsi_period + 1):
            delta = closes[i] - closes[i - 1]
            if delta > 0:
                avg_gain += delta
            else:
                avg_loss -= delta
        avg_gain /= rsi_period
        avg_loss /= rsi_period

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[rsi_period] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[rsi_period] = 100.0

        # Wilder's smoothing
        for i in range(rsi_period + 1, n):
            delta = closes[i] - closes[i - 1]
            if delta > 0:
                avg_gain = (avg_gain * (rsi_period - 1) + delta) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1)) / rsi_period
            else:
                avg_gain = (avg_gain * (rsi_period - 1)) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1) - delta) / rsi_period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi[i] = 100.0

    # === Pre-compute ATR(14) ===
    atr_period = 14
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))

    if n > atr_period:
        # Initial ATR = SMA of first 14 TR values
        s = 0.0
        for i in range(atr_period):
            s += tr[i]
        atr[atr_period - 1] = s / atr_period

        # Wilder's smoothing
        for i in range(atr_period, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    # === Pre-compute SMA(ATR, 50) ===
    atr_sma_period = 50
    atr_sma = np.zeros(n, dtype=np.float64)
    if n > atr_sma_period:
        s = 0.0
        for i in range(atr_sma_period):
            s += atr[i]
        atr_sma[atr_sma_period - 1] = s / atr_sma_period
        for i in range(atr_sma_period, n):
            atr_sma[i] = atr_sma[i - 1] + (atr[i] - atr[i - atr_sma_period]) / atr_sma_period

    # === Pre-compute MA50 and MA200 ===
    ma50 = np.zeros(n, dtype=np.float64)
    ma200 = np.zeros(n, dtype=np.float64)

    if n >= 50:
        s = 0.0
        for i in range(50):
            s += closes[i]
        ma50[49] = s / 50
        for i in range(50, n):
            ma50[i] = ma50[i - 1] + (closes[i] - closes[i - 50]) / 50

    if n >= 200:
        s = 0.0
        for i in range(200):
            s += closes[i]
        ma200[199] = s / 200
        for i in range(200, n):
            ma200[i] = ma200[i - 1] + (closes[i] - closes[i - 200]) / 200

    # === Pre-compute 20-bar high/low ===
    range_period = 20

    # === Pre-compute SMA(high-low, 20) for range_contract ===
    bar_range = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bar_range[i] = highs[i] - lows[i]

    bar_range_sma = np.zeros(n, dtype=np.float64)
    if n >= range_period:
        s = 0.0
        for i in range(range_period):
            s += bar_range[i]
        bar_range_sma[range_period - 1] = s / range_period
        for i in range(range_period, n):
            bar_range_sma[i] = bar_range_sma[i - 1] + (bar_range[i] - bar_range[i - range_period]) / range_period

    # === Warm-up threshold: need at least 200 bars for MA200 ===
    warmup = 200

    # === Compute features per bar ===
    for i in range(warmup, n):
        cur_atr = atr[i]
        if cur_atr < 1e-10:
            cur_atr = 1e-10

        # Feature 0: rsi_momentum = (RSI[i] - RSI[i-5]) / 20, clipped to [-1, 1]
        if i >= 5:
            val = (rsi[i] - rsi[i - 5]) / 20.0
            if val > 1.0:
                val = 1.0
            elif val < -1.0:
                val = -1.0
            features[i, 0] = val

        # Feature 1: atr_ratio = ATR(14)/SMA(ATR,50) - 1.0, clipped to [-2, 2]
        if atr_sma[i] > 1e-10:
            val = cur_atr / atr_sma[i] - 1.0
            if val > 2.0:
                val = 2.0
            elif val < -2.0:
                val = -2.0
            features[i, 1] = val

        # Feature 2: range_position = (close - 20bar_low)/(20bar_high - 20bar_low), [0, 1]
        if i >= range_period:
            period_high = highs[i]
            period_low = lows[i]
            for j in range(1, range_period):
                if highs[i - j] > period_high:
                    period_high = highs[i - j]
                if lows[i - j] < period_low:
                    period_low = lows[i - j]
            rng = period_high - period_low
            if rng > 1e-10:
                val = (closes[i] - period_low) / rng
                if val > 1.0:
                    val = 1.0
                elif val < 0.0:
                    val = 0.0
                features[i, 2] = val
            else:
                features[i, 2] = 0.5

        # Feature 3: trend_alignment = sign(MA50-MA200) * min(1, |diff|/ATR), [-1, 1]
        if ma50[i] > 0 and ma200[i] > 0:
            diff = ma50[i] - ma200[i]
            if diff > 0:
                sign = 1.0
            elif diff < 0:
                sign = -1.0
            else:
                sign = 0.0
            magnitude = abs(diff) / cur_atr
            if magnitude > 1.0:
                magnitude = 1.0
            features[i, 3] = sign * magnitude

        # Feature 4: momentum_exh = (RSI - 50) / 50, [-1, 1]
        features[i, 4] = (rsi[i] - 50.0) / 50.0

        # Feature 5: bar_rejection = 1 - |close-open|/(high-low+1e-10), [0, 1]
        body = abs(closes[i] - opens[i])
        total_range = highs[i] - lows[i] + 1e-10
        val = 1.0 - body / total_range
        if val > 1.0:
            val = 1.0
        elif val < 0.0:
            val = 0.0
        features[i, 5] = val

        # Feature 6: cc_momentum = (close[i]-close[i-3])/ATR, clipped to [-3, 3]
        if i >= 3:
            val = (closes[i] - closes[i - 3]) / cur_atr
            if val > 3.0:
                val = 3.0
            elif val < -3.0:
                val = -3.0
            features[i, 6] = val

        # Feature 7: range_contract = (high-low)/SMA(high-low,20), clipped to [0.3, 3]
        if bar_range_sma[i] > 1e-10:
            val = bar_range[i] / bar_range_sma[i]
            if val > 3.0:
                val = 3.0
            elif val < 0.3:
                val = 0.3
            features[i, 7] = val
        else:
            features[i, 7] = 1.0

    return features


def compute_ml_scores(
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    """
    Compute direction-aware ML exit scores from features and weights.

    Direction-aware flip table (how each feature maps to exit urgency):

    | Feature         | Long flip | Short flip | Rationale                          |
    |-----------------|-----------|------------|------------------------------------|
    | rsi_momentum    | -1        | +1         | Negative momentum = exit long      |
    | atr_ratio       | +1        | +1         | High vol = exit either direction   |
    | range_position  | +1        | -1         | Near top = exit long               |
    | trend_alignment | -1        | +1         | Trend against position = exit      |
    | momentum_exh    | +1        | -1         | RSI overbought = exit long         |
    | bar_rejection   | +1        | +1         | Rejection candle = reversal signal |
    | cc_momentum     | -1        | +1         | Negative CC = exit long            |
    | range_contract  | -1        | -1         | Range contracting = move exhausting|

    long_scores[bar] = sum(weights * features[bar] * long_flips), clipped to [0, max]
    short_scores[bar] = sum(weights * features[bar] * short_flips), clipped to [0, max]

    Args:
        features: (n_bars, 8) float64 array from compute_ml_features()
        weights: (8,) float64 array of feature weights

    Returns:
        (long_scores, short_scores) each (n_bars,) float64
    """
    # Direction-aware flip tables
    long_flips = np.array([-1.0, +1.0, +1.0, -1.0, +1.0, +1.0, -1.0, -1.0], dtype=np.float64)
    short_flips = np.array([+1.0, +1.0, -1.0, +1.0, -1.0, +1.0, +1.0, -1.0], dtype=np.float64)

    # Vectorized: (n_bars, 8) * (8,) * (8,) -> (n_bars, 8) -> sum -> (n_bars,)
    long_scores = np.sum(features * weights[np.newaxis, :] * long_flips[np.newaxis, :], axis=1)
    short_scores = np.sum(features * weights[np.newaxis, :] * short_flips[np.newaxis, :], axis=1)

    # Clip to [0, max] - negative scores mean "don't exit"
    np.clip(long_scores, 0.0, None, out=long_scores)
    np.clip(short_scores, 0.0, None, out=short_scores)

    return long_scores, short_scores
