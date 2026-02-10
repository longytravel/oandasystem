"""Signal-level features for meta-labeling / entry filtering.

Instead of predicting per-bar exit timing (which failed on RSI divergence),
predict which signals will be profitable. This is the meta-labeling approach
from Lopez de Prado ("Advances in Financial Machine Learning").

One prediction per signal (not hundreds per trade) with a clean binary label.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

SIGNAL_FEATURE_NAMES = [
    'direction',          # +1 long, -1 short
    'atr_norm',           # ATR / close (normalized volatility)
    'vol_regime',         # ATR / 50-bar SMA of ATR (high/low vol)
    'trend_short',        # 5-bar price change / ATR
    'trend_long',         # 20-bar price change / ATR
    'rsi_at_signal',      # RSI-14 value at signal bar
    'momentum_rsi_3',     # 3-bar RSI change
    'momentum_rsi_10',    # 10-bar RSI change
    'hour_of_day',
    'day_of_week',
    'candle_body_ratio',  # abs(close-open) / (high-low)
    'aligned_trend',      # direction * trend_long (positive = supportive)
    # --- Enhanced features (v2) ---
    'price_vs_20bar_high',  # (close - 20bar high) / ATR  (0 to -X, how far from recent high)
    'price_vs_20bar_low',   # (close - 20bar low) / ATR   (0 to +X, how far from recent low)
    'atr_change_5',         # (ATR - ATR[-5]) / ATR[-5]   (vol acceleration)
    'close_vs_ema50',       # (close - EMA50) / ATR        (multi-TF trend)
    'close_vs_ema200',      # (close - EMA200) / ATR       (major trend)
    'bar_range_ratio',      # (high - low) / ATR           (bar size vs norm)
    'upper_shadow_pct',     # upper shadow / bar range     (rejection signal)
    'bars_since_prev',      # bars since previous signal   (signal clustering)
]

N_SIGNAL_FEATURES = len(SIGNAL_FEATURE_NAMES)


def compute_signal_features(
    signal_bars: np.ndarray,
    directions: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    hours: np.ndarray,
    days: np.ndarray,
    strategy_attrs: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Compute features at each signal's entry bar for meta-labeling.

    Args:
        signal_bars: Bar indices where signals fire, shape (n_signals,)
        directions: Signal directions (+1/-1), shape (n_signals,)
        closes/highs/lows/opens: Market data arrays, shape (n_bars,)
        hours: Hour of day, shape (n_bars,)
        days: Day of week, shape (n_bars,)
        strategy_attrs: Optional dict of strategy-specific signal attribute
            arrays. Each value shape (n_all_signals,) from the full signal set.
            A signal_indices array maps signal_bars to indices in these arrays.
            Keys like 'rsi_diffs', 'bars_between', 'price_slopes', etc.

    Returns:
        DataFrame with n_signals rows and feature columns (market + strategy).
    """
    n_signals = len(signal_bars)
    n_bars = len(closes)

    # Determine column names including any strategy-specific attributes
    strat_attr_names = []
    if strategy_attrs:
        strat_attr_names = sorted(strategy_attrs.keys())
    all_feature_names = SIGNAL_FEATURE_NAMES + strat_attr_names
    n_total_features = N_SIGNAL_FEATURES + len(strat_attr_names)

    if n_signals == 0:
        return pd.DataFrame(columns=all_feature_names)

    # Precompute indicators once
    atr_14 = _compute_atr(highs, lows, closes, period=14)
    rsi_14 = _compute_rsi(closes, period=14)
    atr_50_sma = _rolling_mean(atr_14, period=50)
    ema_50 = _compute_ema(closes, period=50)
    ema_200 = _compute_ema(closes, period=200)
    rolling_high_20 = _rolling_max(highs, period=20)
    rolling_low_20 = _rolling_min(lows, period=20)

    features_arr = np.zeros((n_signals, n_total_features), dtype=np.float64)

    for i in range(n_signals):
        bar = int(signal_bars[i])
        if bar < 0 or bar >= n_bars:
            continue

        direction = float(directions[i])
        close = closes[bar]
        atr = atr_14[bar]

        # 0: direction
        features_arr[i, 0] = direction

        # 1: atr_norm (ATR / close)
        features_arr[i, 1] = atr / close if close > 0 else 0.0

        # 2: vol_regime (ATR / 50-bar SMA of ATR)
        atr_50 = atr_50_sma[bar]
        features_arr[i, 2] = atr / atr_50 if atr_50 > 0 else 1.0

        # 3: trend_short (5-bar price change / ATR)
        if bar >= 5 and atr > 0:
            features_arr[i, 3] = (closes[bar] - closes[bar - 5]) / atr

        # 4: trend_long (20-bar price change / ATR)
        if bar >= 20 and atr > 0:
            features_arr[i, 4] = (closes[bar] - closes[bar - 20]) / atr

        # 5: rsi_at_signal
        features_arr[i, 5] = rsi_14[bar]

        # 6: momentum_rsi_3
        if bar >= 3:
            features_arr[i, 6] = rsi_14[bar] - rsi_14[bar - 3]

        # 7: momentum_rsi_10
        if bar >= 10:
            features_arr[i, 7] = rsi_14[bar] - rsi_14[bar - 10]

        # 8: hour_of_day
        features_arr[i, 8] = hours[bar] if bar < len(hours) else 0

        # 9: day_of_week
        features_arr[i, 9] = days[bar] if bar < len(days) else 0

        # 10: candle_body_ratio
        bar_range = highs[bar] - lows[bar]
        if bar_range > 0:
            features_arr[i, 10] = abs(closes[bar] - opens[bar]) / bar_range

        # 11: aligned_trend (direction * trend_long — positive = trend supports signal)
        features_arr[i, 11] = direction * features_arr[i, 4]

        # --- Enhanced features (v2) ---

        # 12: price_vs_20bar_high (0 = at high, negative = below high)
        if bar >= 20 and atr > 0:
            features_arr[i, 12] = (close - rolling_high_20[bar]) / atr

        # 13: price_vs_20bar_low (0 = at low, positive = above low)
        if bar >= 20 and atr > 0:
            features_arr[i, 13] = (close - rolling_low_20[bar]) / atr

        # 14: atr_change_5 (volatility acceleration)
        if bar >= 5 and atr_14[bar - 5] > 0:
            features_arr[i, 14] = (atr - atr_14[bar - 5]) / atr_14[bar - 5]

        # 15: close_vs_ema50 (position relative to EMA50)
        if bar >= 50 and atr > 0 and ema_50[bar] > 0:
            features_arr[i, 15] = (close - ema_50[bar]) / atr

        # 16: close_vs_ema200 (position relative to EMA200)
        if bar >= 200 and atr > 0 and ema_200[bar] > 0:
            features_arr[i, 16] = (close - ema_200[bar]) / atr

        # 17: bar_range_ratio (current bar size vs ATR)
        if atr > 0:
            features_arr[i, 17] = bar_range / atr if bar_range > 0 else 0.0

        # 18: upper_shadow_pct (rejection/indecision signal)
        if bar_range > 0:
            upper_shadow = highs[bar] - max(opens[bar], closes[bar])
            features_arr[i, 18] = upper_shadow / bar_range

        # 19: bars_since_prev (signal clustering - many signals close together = noisy)
        if i > 0:
            prev_bar = int(signal_bars[i - 1])
            features_arr[i, 19] = min(bar - prev_bar, 500)  # cap at 500
        else:
            features_arr[i, 19] = 500  # first signal = max distance

    # Append strategy-specific attributes as additional columns
    if strategy_attrs and strat_attr_names:
        # strategy_attrs values are indexed by signal_indices (provided by caller)
        # The caller should have already sliced these to match signal_bars
        for j, attr_name in enumerate(strat_attr_names):
            col_idx = N_SIGNAL_FEATURES + j
            attr_arr = strategy_attrs[attr_name]
            if len(attr_arr) == n_signals:
                features_arr[:, col_idx] = attr_arr
            else:
                # Mismatch — leave as zeros (logged by caller)
                pass

    return pd.DataFrame(features_arr, columns=all_feature_names)


def train_signal_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    n_optuna_trials: int = 20,
) -> Optional[Tuple[object, Dict[str, float]]]:
    """Train a binary classifier to predict signal profitability.

    Includes OOS skill check: splits training data temporally (80/20),
    checks validation AUC >= 0.55. If no skill detected, returns None.
    If skill detected, retrains on full training set.

    Args:
        features: Signal features DataFrame
        labels: Binary labels (1=profitable, 0=not)
        n_optuna_trials: Hyperparameter tuning trials

    Returns:
        Tuple of (trained classifier, metrics dict) or None on failure.
        Metrics include 'train_auc', 'val_auc', 'skill_detected'.
    """
    if len(features) < 10:
        logger.warning(f"  Signal filter: too few samples ({len(features)}), need >= 10")
        return None

    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    if n_positive < 3 or n_negative < 3:
        logger.warning(f"  Signal filter: imbalanced classes (pos={n_positive}, neg={n_negative})")
        return None

    n = len(features)
    n_samples = len(labels)

    # --- OOS skill check: temporal 80/20 split ---
    split_idx = int(n * 0.8)
    if split_idx < 8 or (n - split_idx) < 3:
        # Not enough data for meaningful split — train on all, skip skill check
        logger.info(f"  Signal filter: too few for skill check ({n} samples), training on all")
        clf, backend = _fit_classifier(features, labels)
        if clf is None:
            return None
        return clf, {'train_auc': -1.0, 'val_auc': -1.0, 'skill_detected': True, 'backend': backend}

    X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]

    # Check class balance in train split
    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    if n_pos_train < 2 or n_neg_train < 2:
        logger.info(f"  Signal filter: imbalanced train split (pos={n_pos_train}, neg={n_neg_train}), training on all")
        clf, backend = _fit_classifier(features, labels)
        if clf is None:
            return None
        return clf, {'train_auc': -1.0, 'val_auc': -1.0, 'skill_detected': True, 'backend': backend}

    # Train on 80%
    clf_val, backend = _fit_classifier(X_train, y_train)
    if clf_val is None:
        return None

    # Compute validation AUC
    try:
        val_probs = clf_val.predict_proba(X_val.values)[:, 1]
        from sklearn.metrics import roc_auc_score
        # Need at least 2 classes in validation
        if len(set(y_val)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(y_val, val_probs)
    except Exception as e:
        logger.warning(f"  Signal filter: AUC computation failed: {e}")
        val_auc = 0.5

    # Compute train AUC for reference
    try:
        train_probs = clf_val.predict_proba(X_train.values)[:, 1]
        if len(set(y_train)) < 2:
            train_auc = 0.5
        else:
            train_auc = roc_auc_score(y_train, train_probs)
    except Exception:
        train_auc = -1.0

    skill_detected = val_auc >= 0.55
    logger.info(
        f"  Signal filter skill check: train_AUC={train_auc:.3f}, val_AUC={val_auc:.3f} "
        f"-> {'SKILL DETECTED' if skill_detected else 'NO SKILL (AUC < 0.55)'}"
    )

    if not skill_detected:
        return None

    # Skill detected — retrain on full training set for final model
    clf_full, backend = _fit_classifier(features, labels)
    if clf_full is None:
        return None

    metrics = {
        'train_auc': round(train_auc, 4),
        'val_auc': round(val_auc, 4),
        'skill_detected': True,
        'backend': backend,
    }
    return clf_full, metrics


def _fit_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> Tuple[Optional[object], str]:
    """Fit a classifier with reduced complexity for small sample sizes.

    Returns:
        Tuple of (classifier, backend_name) or (None, '').
    """
    n = len(features)

    # Try CatBoost first, fall back to sklearn
    try:
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=100,      # was 200 — reduced for small samples
            depth=3,             # was 4 — shallower to prevent overfitting
            learning_rate=0.05,
            l2_leaf_reg=10.0,    # was 5.0 — more regularization
            verbose=0,
            random_seed=42,
            auto_class_weights='Balanced',
        )
        clf.fit(features.values, labels)
        n_positive = int(labels.sum())
        n_negative = len(labels) - n_positive
        logger.info(f"  Signal filter: trained CatBoost ({n} samples, pos={n_positive}, neg={n_negative})")
        return clf, 'catboost'
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        min_samples_leaf = max(5, n // 10)
        clf = GradientBoostingClassifier(
            n_estimators=50,         # was 100 — reduced
            max_depth=2,             # was 3 — shallower
            learning_rate=0.05,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        clf.fit(features.values, labels)
        n_positive = int(labels.sum())
        n_negative = len(labels) - n_positive
        logger.info(f"  Signal filter: trained sklearn GBM ({n} samples, pos={n_positive}, neg={n_negative}, "
                    f"min_leaf={min_samples_leaf})")
        return clf, 'sklearn'
    except ImportError:
        logger.warning("  Signal filter: no ML backend available")
        return None, ''


def calibrate_threshold(
    clf,
    features: pd.DataFrame,
    labels: np.ndarray,
    pnls: np.ndarray,
) -> float:
    """Calibrate probability threshold to maximize profit factor improvement.

    Tests thresholds from 0.3 to 0.7 in 0.05 steps on training data.
    Scores each threshold by: PF(kept_signals) / PF(all_signals).
    Requires keeping >= 50% of signals.

    Args:
        clf: Trained classifier with predict_proba()
        features: Training signal features
        labels: Binary labels (1=profitable, 0=not)
        pnls: Per-signal PnL array (same length as features)

    Returns:
        Optimal threshold (float), defaults to 0.5 on failure.
    """
    if clf is None or len(features) == 0 or len(pnls) == 0:
        return 0.5

    try:
        probs = clf.predict_proba(features.values)[:, 1]
    except Exception:
        return 0.5

    # Compute baseline profit factor
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses <= 0:
        return 0.5  # All winners — no filtering needed
    base_pf = wins / losses if losses > 0 else 999.0

    best_threshold = 0.5
    best_score = 0.0
    n_total = len(pnls)

    for thresh_10x in range(6, 15):  # 0.30 to 0.70 in 0.05 steps
        thresh = thresh_10x / 20.0
        keep = probs >= thresh
        n_kept = int(keep.sum())

        # Must keep at least 50% of signals
        if n_kept < n_total * 0.5 or n_kept < 3:
            continue

        kept_pnls = pnls[keep]
        kept_wins = kept_pnls[kept_pnls > 0].sum()
        kept_losses = abs(kept_pnls[kept_pnls < 0].sum())

        if kept_losses <= 0:
            kept_pf = 999.0
        else:
            kept_pf = kept_wins / kept_losses

        # Score = PF improvement ratio
        pf_improvement = kept_pf / base_pf if base_pf > 0 else 1.0

        if pf_improvement > best_score:
            best_score = pf_improvement
            best_threshold = thresh

    # Only use calibrated threshold if it actually improved PF
    if best_score <= 1.0:
        return 0.5

    return best_threshold


def predict_signal_filter(
    clf,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict which signals to keep.

    Args:
        clf: Trained classifier with predict_proba()
        features: Signal features for test window
        threshold: Probability threshold (keep signals with prob >= threshold)

    Returns:
        Boolean mask, shape (n_signals,). True = keep signal.
    """
    if clf is None or len(features) == 0:
        return np.ones(len(features), dtype=bool)

    try:
        probs = clf.predict_proba(features.values)[:, 1]
        keep_mask = probs >= threshold
        n_kept = int(keep_mask.sum())
        n_total = len(keep_mask)
        logger.info(f"  Signal filter: keeping {n_kept}/{n_total} signals "
                    f"(threshold={threshold:.2f}, mean_prob={probs.mean():.3f})")
        return keep_mask
    except Exception as e:
        logger.warning(f"  Signal filter prediction failed: {e}")
        return np.ones(len(features), dtype=bool)


def train_mae_regressor(
    features: pd.DataFrame,
    mae_values: np.ndarray,
) -> Optional[object]:
    """Train a regressor to predict MAE_r (max adverse excursion in R-multiples).

    Args:
        features: Signal features DataFrame
        mae_values: Continuous MAE_r values (0.0 to 1.0+)

    Returns:
        Trained regressor with predict() method, or None on failure.
    """
    if len(features) < 30:
        logger.warning(f"  MAE regressor: too few samples ({len(features)}), need >= 30")
        return None

    # Try CatBoost first, fall back to sklearn
    try:
        from catboost import CatBoostRegressor
        reg = CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            verbose=0,
            random_seed=42,
        )
        reg.fit(features.values, mae_values)
        # Quick R² on training data
        preds = reg.predict(features.values)
        ss_res = np.sum((mae_values - preds) ** 2)
        ss_tot = np.sum((mae_values - mae_values.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        logger.info(f"  MAE regressor: CatBoost R²={r2:.3f} ({len(features)} samples)")
        return reg
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=5,
            random_state=42,
        )
        reg.fit(features.values, mae_values)
        preds = reg.predict(features.values)
        ss_res = np.sum((mae_values - preds) ** 2)
        ss_tot = np.sum((mae_values - mae_values.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        logger.info(f"  MAE regressor: sklearn GBM R²={r2:.3f} ({len(features)} samples)")
        return reg
    except ImportError:
        logger.warning("  MAE regressor: no ML backend available")
        return None


# === Helper functions for indicator computation ===

def _compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range using EMA smoothing."""
    n = len(closes)
    tr = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    # EMA-style ATR
    atr[0] = tr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    return atr


def _compute_rsi(closes, period=14):
    """Compute RSI using Wilder's smoothing."""
    n = len(closes)
    rsi = np.full(n, 50.0, dtype=np.float64)

    if n < period + 1:
        return rsi

    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])

    if avg_loss > 0:
        rsi[period] = 100 - 100 / (1 + avg_gain / avg_loss)
    elif avg_gain > 0:
        rsi[period] = 100.0

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rsi[i] = 100 - 100 / (1 + avg_gain / avg_loss)
        elif avg_gain > 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 50.0

    return rsi


def _compute_ema(arr, period=50):
    """Compute Exponential Moving Average."""
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _rolling_max(arr, period=20):
    """Compute rolling maximum."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        result[i] = np.max(arr[start:i + 1])
    return result


def _rolling_min(arr, period=20):
    """Compute rolling minimum."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        result[i] = np.min(arr[start:i + 1])
    return result


def _rolling_mean(arr, period=50):
    """Compute rolling mean using cumulative sum approach."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    cum_sum = 0.0
    for i in range(n):
        cum_sum += arr[i]
        if i >= period:
            cum_sum -= arr[i - period]
            result[i] = cum_sum / period
        elif i > 0:
            result[i] = cum_sum / (i + 1)
        else:
            result[i] = arr[0]
    return result
