"""
ML Exit Dataset Builder - Orchestrates feature/label computation into a DataFrame.

Runs a backtest with telemetry, then iterates over each trade and each bar
within that trade to build a (trade_id, bar_idx) -> (features, labels) dataset.

Typical output: 3000-6000 rows for a 3yr backtest with 30-100 trades of ~30-60 bars each.
"""
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from loguru import logger

from optimization.numba_backtest import (
    full_backtest_with_telemetry,
    get_quote_conversion_rate,
)
from pipeline.ml_exit.features import (
    get_feature_names,
    precompute_market_features,
    compute_decision_features,
    N_FEATURES,
)
from pipeline.ml_exit.labeling import get_label_names, compute_labels


def build_dataset_hash(df: pd.DataFrame, params: Dict[str, Any]) -> str:
    """
    Compute a reproducibility hash from data range and params.

    Args:
        df: OHLC DataFrame
        params: Strategy parameter dict

    Returns:
        Hex digest string (first 12 chars of SHA256)
    """
    h = hashlib.sha256()
    h.update(str(df.index[0]).encode())
    h.update(str(df.index[-1]).encode())
    h.update(str(len(df)).encode())
    h.update(str(sorted(params.items())).encode())
    return h.hexdigest()[:12]


def build_exit_dataset(
    df: pd.DataFrame,
    params: Dict[str, Any],
    strategy,
    config,
    pair: Optional[str] = None,
    min_trade_bars: int = 3,
) -> pd.DataFrame:
    """
    Build the ML exit training dataset from a single backtest run.

    Steps:
    1. Run backtest with full_backtest_with_telemetry() to get per-trade telemetry
    2. Pre-compute market features once for the entire dataset
    3. For each trade, for each bar while the trade is active:
       - Compute 14 decision features (no look-ahead)
       - Compute 5 supervised labels (uses future data)
       - Append row

    Args:
        df: OHLC DataFrame with columns 'open', 'high', 'low', 'close'
            and a DatetimeIndex.
        params: Strategy parameter dict (same as used in pipeline optimization).
        strategy: Strategy instance (must have precompute_for_dataset, get_all_arrays).
        config: PipelineConfig (for initial_capital, risk_per_trade, pair, spread_pips).
        pair: Currency pair override (uses config.pair if None).
        min_trade_bars: Minimum bars a trade must last to be included (default 3).

    Returns:
        DataFrame with columns:
        - trade_id: int (0-indexed trade number)
        - bar_idx: int (bar index in the dataset)
        - <14 feature columns>
        - <5 label columns>
        - dataset_hash: str (for reproducibility)
    """
    pair = pair or config.pair
    pip_size = 0.01 if 'JPY' in pair else 0.0001

    # --- Prepare market data arrays ---
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64) if 'open' in df.columns else None
    days = df.index.dayofweek.values.astype(np.int64)
    hours = df.index.hour.values.astype(np.int64) if hasattr(df.index, 'hour') else np.zeros(len(df), dtype=np.int64)
    day_of_week = df.index.dayofweek.values.astype(np.int64)

    n_bars = len(highs)

    # --- Precompute strategy signals ---
    strategy.precompute_for_dataset(df)
    signal_arrays, mgmt_arrays = strategy.get_all_arrays(params, highs, lows, closes, days)

    n_signals = len(signal_arrays['entry_bars'])
    if n_signals < 3:
        logger.warning(f'Too few signals ({n_signals}) to build dataset')
        return pd.DataFrame()

    # --- Apply spread to entry prices (same as s5_montecarlo) ---
    entry_prices = np.where(
        signal_arrays['directions'] == 1,
        signal_arrays['entry_prices'] + config.spread_pips * pip_size,
        signal_arrays['entry_prices'] - config.spread_pips * pip_size,
    )

    # --- Prepare management arrays (same pattern as s5_montecarlo._get_trade_pnls) ---
    n = n_signals
    use_trailing = mgmt_arrays.get('use_trailing', np.zeros(n, dtype=np.bool_))
    trail_start = mgmt_arrays.get('trail_start_pips', np.zeros(n, dtype=np.float64))
    trail_step = mgmt_arrays.get('trail_step_pips', np.zeros(n, dtype=np.float64))
    use_be = mgmt_arrays.get('use_breakeven', np.zeros(n, dtype=np.bool_))
    be_trigger = mgmt_arrays.get('be_trigger_pips', np.zeros(n, dtype=np.float64))
    be_offset = mgmt_arrays.get('be_offset_pips', np.zeros(n, dtype=np.float64))
    use_partial = mgmt_arrays.get('use_partial', np.zeros(n, dtype=np.bool_))
    partial_pct = mgmt_arrays.get('partial_pct', np.zeros(n, dtype=np.float64))
    partial_target = mgmt_arrays.get('partial_target_pips', np.zeros(n, dtype=np.float64))
    max_bars_arr = mgmt_arrays.get('max_bars', np.zeros(n, dtype=np.int64))
    trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n, dtype=np.int64))
    chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n, 3.0, dtype=np.float64))
    atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n, 35.0, dtype=np.float64))
    stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n, dtype=np.int64))
    quality_mult = np.empty(0, dtype=np.float64)

    # V6: ML exit arrays (disabled for dataset building — we want raw trade behavior)
    use_ml = np.zeros(n, dtype=np.bool_)
    ml_min_hold_arr = np.zeros(n, dtype=np.int64)
    ml_threshold_arr = np.ones(n, dtype=np.float64)
    ml_long = np.zeros(n_bars, dtype=np.float64)
    ml_short = np.zeros(n_bars, dtype=np.float64)

    # --- Run backtest with telemetry ---
    result = full_backtest_with_telemetry(
        signal_arrays['entry_bars'],
        entry_prices,
        signal_arrays['directions'],
        signal_arrays['sl_prices'],
        signal_arrays['tp_prices'],
        use_trailing, trail_start, trail_step,
        use_be, be_trigger, be_offset,
        use_partial, partial_pct, partial_target,
        max_bars_arr,
        trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
        ml_long, ml_short, use_ml, ml_min_hold_arr, ml_threshold_arr,
        highs, lows, closes, days,
        config.initial_capital,
        config.risk_per_trade,
        pip_size,
        params.get('max_daily_trades', 0),
        params.get('max_daily_loss_pct', 0.0),
        quality_mult,
        get_quote_conversion_rate(pair, 'USD'),
    )

    # Unpack telemetry
    pnls = result[0]
    equity_curve = result[1]
    exit_reasons = result[2]
    bars_held = result[3]
    entry_bar_indices = result[4]
    exit_bar_indices = result[5]
    mfe_r_arr = result[6]
    mae_r_arr = result[7]
    signal_indices_arr = result[8]
    n_trades = result[9]

    if n_trades < 3:
        logger.warning(f'Too few trades ({n_trades}) to build dataset')
        return pd.DataFrame()

    logger.info(f'Building dataset from {n_trades} trades')

    # Validate signal alignment: every trade's entry_bar should match its signal's entry_bar
    n_mismatches = 0
    for ti in range(min(n_trades, len(signal_indices_arr))):
        si = int(signal_indices_arr[ti])
        if si < n_signals and entry_bar_indices[ti] != signal_arrays['entry_bars'][si]:
            n_mismatches += 1
            if n_mismatches <= 3:
                logger.warning(
                    f'Signal alignment mismatch: trade {ti} entry_bar={entry_bar_indices[ti]} '
                    f'!= signal {si} entry_bar={signal_arrays["entry_bars"][si]}'
                )
    if n_mismatches > 0:
        logger.warning(f'Total signal alignment mismatches: {n_mismatches}/{n_trades}')
    else:
        logger.info(f'Signal alignment verified: all {n_trades} trades match their signals')

    # --- Pre-compute market features once ---
    market_data = precompute_market_features(highs, lows, closes, opens)

    # --- Map trade index to signal index for SL/TP prices ---
    # signal_indices maps each trade to its original signal (trade_i may != signal_i
    # when signals are skipped during an open position).
    sig_entry_prices = entry_prices  # spread-adjusted
    sig_sl_prices = signal_arrays['sl_prices']
    sig_tp_prices = signal_arrays['tp_prices']
    sig_directions = signal_arrays['directions']

    # --- Build dataset rows ---
    feature_names = get_feature_names()
    label_names = get_label_names()

    rows = []
    dataset_hash = build_dataset_hash(df, params)

    for trade_i in range(n_trades):
        entry_bar = int(entry_bar_indices[trade_i])
        exit_bar = int(exit_bar_indices[trade_i])
        trade_duration = exit_bar - entry_bar

        if trade_duration < min_trade_bars:
            continue

        # Map trade to its original signal index (signals can be skipped when position is open)
        sig_idx = int(signal_indices_arr[trade_i])
        if sig_idx >= len(sig_entry_prices):
            continue

        direction = int(sig_directions[sig_idx])
        entry_price = float(sig_entry_prices[sig_idx])
        sl_price = float(sig_sl_prices[sig_idx])
        tp_price = float(sig_tp_prices[sig_idx])
        initial_sl_dist = abs(entry_price - sl_price)

        if initial_sl_dist < 1e-10:
            continue

        # Track running MFE/MAE bar-by-bar
        running_mfe_r = 0.0
        running_mae_r = 0.0

        for bar in range(entry_bar, exit_bar + 1):
            if bar >= n_bars:
                break

            # Update running MFE/MAE
            if direction == 1:
                unrealized = (closes[bar] - entry_price) / initial_sl_dist
                fav_excursion = (highs[bar] - entry_price) / initial_sl_dist
                adv_excursion = (entry_price - lows[bar]) / initial_sl_dist
            else:
                unrealized = (entry_price - closes[bar]) / initial_sl_dist
                fav_excursion = (entry_price - lows[bar]) / initial_sl_dist
                adv_excursion = (highs[bar] - entry_price) / initial_sl_dist

            if fav_excursion > running_mfe_r:
                running_mfe_r = fav_excursion
            if adv_excursion > running_mae_r:
                running_mae_r = adv_excursion

            # Build trade_state dict for feature computation
            trade_state = {
                'direction': direction,
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'initial_sl_dist': initial_sl_dist,
                'mfe_r': running_mfe_r,
                'mae_r': running_mae_r,
                'hour': int(hours[bar]),
                'day_of_week': int(day_of_week[bar]),
                'max_hold_bars': int(max_bars_arr[sig_idx]) if sig_idx < len(max_bars_arr) else 0,
            }

            # Compute features (no look-ahead)
            features = compute_decision_features(trade_state, market_data, bar)

            # Compute labels (uses future data)
            labels = compute_labels(
                highs, lows, closes,
                entry_price, sl_price, tp_price,
                direction, entry_bar, exit_bar, bar,
                initial_sl_dist,
            )

            # Build row
            row = {
                'trade_id': trade_i,
                'bar_idx': bar,
            }
            for j, fname in enumerate(feature_names):
                row[fname] = features[j]
            for lname in label_names:
                row[lname] = labels[lname]
            row['dataset_hash'] = dataset_hash

            rows.append(row)

    if not rows:
        logger.warning('No dataset rows generated')
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    logger.info(
        f'Dataset built: {len(result_df)} rows from {result_df["trade_id"].nunique()} trades, '
        f'hash={dataset_hash}'
    )
    return result_df
