"""
Full-Featured Numba Backtest Engine with R² and OnTester Score.

Supports ALL trade management features while maintaining fast execution:
- Basic SL/TP exits
- Trailing stops (start trigger, step size)
- Breakeven (trigger, lock-in offset)
- Partial closes (close X% at TP1, trail remainder)
- Time-based exits (max bars in trade)
- Daily trade limits
- Daily loss limits

Metrics include:
- R² (equity curve smoothness) - higher = smoother growth
- OnTester Score (MT5-style combined metric)

Design: All position state as primitives for Numba compatibility.
"""
import numpy as np
from numba import njit
from typing import Tuple, NamedTuple


class Metrics(NamedTuple):
    """Lightweight metrics container with R² and OnTester score."""
    trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    max_dd: float
    total_return: float
    r_squared: float       # Equity curve smoothness (0-1, higher=smoother)
    ontester_score: float  # MT5-style combined score


@njit(cache=True, fastmath=True)
def calculate_r_squared(equity_curve: np.ndarray) -> float:
    """
    Calculate R² of equity curve vs linear regression.

    R² = 1 - (SS_residual / SS_total)

    Higher R² = smoother, more consistent growth
    1.0 = perfectly linear growth
    0.0 = random walk

    Args:
        equity_curve: Array of cumulative equity values (one per trade)

    Returns:
        R² value between 0 and 1
    """
    n = len(equity_curve)
    if n < 3:
        return 0.0

    # Create x values (trade number)
    x_sum = 0.0
    y_sum = 0.0
    for i in range(n):
        x_sum += i
        y_sum += equity_curve[i]

    x_mean = x_sum / n
    y_mean = y_sum / n

    # Calculate slope and intercept
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        x_diff = i - x_mean
        numerator += x_diff * (equity_curve[i] - y_mean)
        denominator += x_diff * x_diff

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Calculate R²
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        predicted = slope * i + intercept
        ss_res += (equity_curve[i] - predicted) ** 2
        ss_tot += (equity_curve[i] - y_mean) ** 2

    if ss_tot == 0:
        return 0.0

    r2 = 1.0 - (ss_res / ss_tot)

    # Clamp to [0, 1]
    if r2 < 0:
        r2 = 0.0
    if r2 > 1:
        r2 = 1.0

    return r2


@njit(cache=True, fastmath=True)
def calculate_ontester_score(
    profit: float,
    r_squared: float,
    profit_factor: float,
    trades: int,
    max_dd_pct: float
) -> float:
    """
    Calculate MT5-style OnTester score.

    Score = Profit × R² × ProfitFactor × √Trades / (MaxDrawdown + 5)

    This formula rewards:
    - Higher profit (obvious)
    - Smoother equity curve (R²)
    - Higher profit factor (more profit per loss)
    - More trades (statistical significance)
    - Lower drawdown

    Args:
        profit: Total profit in account currency
        r_squared: Equity curve R² (0-1)
        profit_factor: Gross profit / gross loss
        trades: Number of trades
        max_dd_pct: Maximum drawdown as percentage (e.g., 15.0 for 15%)

    Returns:
        OnTester score (higher = better)
    """
    if trades < 1 or profit <= 0 or profit_factor <= 0:
        return 0.0

    # Cap profit factor to avoid extreme values
    pf = min(profit_factor, 10.0)

    # The formula
    score = (profit * r_squared * pf * np.sqrt(trades)) / (max_dd_pct + 5.0)

    return score


@njit(cache=True, fastmath=True)
def full_backtest_numba(
    # === Signal arrays (per signal) ===
    entry_bars: np.ndarray,         # Bar index to enter
    entry_prices: np.ndarray,       # Entry price
    directions: np.ndarray,         # 1=buy, -1=sell
    sl_prices: np.ndarray,          # Stop loss price
    tp_prices: np.ndarray,          # Take profit price

    # === Trade management arrays (per signal) ===
    use_trailing: np.ndarray,       # bool: enable trailing for this trade
    trail_start_pips: np.ndarray,   # Pips profit before trailing starts
    trail_step_pips: np.ndarray,    # Trail step size in pips
    use_breakeven: np.ndarray,      # bool: enable breakeven
    be_trigger_pips: np.ndarray,    # Pips profit to trigger BE
    be_offset_pips: np.ndarray,     # Pips to lock in at BE
    use_partial: np.ndarray,        # bool: enable partial close
    partial_pct: np.ndarray,        # Percent to close at TP1 (0-1)
    partial_target_pips: np.ndarray,  # Pips for partial TP
    max_bars: np.ndarray,           # Max bars in trade (0=unlimited)

    # === V5: New exit management arrays (per signal) ===
    trail_mode: np.ndarray,           # int: 0=fixed, 1=chandelier
    chandelier_atr_mult: np.ndarray,  # float: ATR multiplier for chandelier
    atr_pips: np.ndarray,             # float: ATR in pips per signal
    stale_exit_bars: np.ndarray,      # int: max bars without progress (0=disabled)

    # === V6: ML exit arrays ===
    ml_long_scores: np.ndarray,    # float64 (n_bars,) — pre-computed long exit scores
    ml_short_scores: np.ndarray,   # float64 (n_bars,) — pre-computed short exit scores
    use_ml_exit: np.ndarray,       # bool (n_signals,) — enable ML exit per signal
    ml_min_hold: np.ndarray,       # int64 (n_signals,) — min bars before ML can trigger
    ml_threshold: np.ndarray,      # float64 (n_signals,) — score threshold to exit

    # === Market data ===
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    days: np.ndarray,               # Day of week (0=Mon, 6=Sun)

    # === Account params ===
    initial_capital: float,
    risk_pct: float,
    pip_size: float,
    max_daily_trades: int,          # 0 = unlimited
    max_daily_loss_pct: float,      # 0 = unlimited

    # === V2: Quality-based sizing ===
    quality_mult: np.ndarray,  # Position size multiplier per signal (1.0 = normal, use empty array to disable)

    # === V3: Cross-currency pip value correction ===
    quote_conversion_rate: float = 1.0,  # Quote currency to account currency rate

    # === Fix Finding 11: configurable bars_per_year for Sharpe annualization ===
    bars_per_year: float = 5544.0,  # H1 default (252 days * ~22 hours)

    # === V6.2: ML exit cooldown ===
    ml_exit_cooldown_bars: int = 0,  # Bars to skip after ML exit (0=disabled)
) -> Tuple[int, float, float, float, float, float, float, float]:
    """
    Full-featured Numba-compiled backtest engine.

    Returns: (trades, win_rate, profit_factor, sharpe, max_dd, total_return, r_squared, ontester_score)

    V5: Replaced chain_be_to_trail with trail_mode, chandelier_atr_mult, atr_pips, stale_exit_bars.
    V6: Added ML exit arrays (ml_long_scores, ml_short_scores, use_ml_exit, ml_min_hold, ml_threshold).
    V6.2: Added ml_exit_cooldown_bars to prevent trade count inflation after ML exit.
    """
    n_signals = len(entry_bars)
    if n_signals == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n_bars = len(highs)
    # FIX V3: pip_value now includes quote currency conversion
    # For USD-quoted pairs with USD account: pip_value = 1/pip_size * 1.0 = 10000 (GBP/USD)
    # For JPY-quoted pairs with USD account: pip_value = 1/pip_size * (1/USD_JPY) = 100 * 0.0067 = 0.67 (USD/JPY @ 150)
    pip_value = (1.0 / pip_size) * quote_conversion_rate

    # Trade tracking
    max_trades = n_signals * 2  # Allow for partial closes
    pnls = np.zeros(max_trades, dtype=np.float64)
    equity_curve = np.zeros(max_trades, dtype=np.float64)
    n_trades = 0

    # Account tracking
    equity = initial_capital
    peak_equity = initial_capital
    max_dd = 0.0

    # Daily tracking
    current_day = -1
    daily_trades = 0
    daily_pnl = 0.0
    daily_start_equity = initial_capital

    # Position state (primitives for Numba)
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0
    pos_start_bar = 0
    pos_signal_idx = -1

    # Position management state
    pos_be_triggered = False
    pos_trail_active = False
    pos_trail_high = 0.0
    pos_partial_done = False
    pos_remaining_size = 0.0
    # Fix Finding 2: accumulate partial close PnL instead of recording separate trade
    pos_partial_pnl = 0.0

    # Signal index
    sig_idx = 0

    # V6.2: ML exit cooldown counter
    cooldown_remaining = 0

    for bar in range(n_bars):
        # === Daily reset ===
        if days[bar] != current_day:
            current_day = days[bar]
            daily_trades = 0
            daily_pnl = 0.0
            daily_start_equity = equity

        # === Check position exit ===
        if in_pos:
            exited = False
            exit_price = 0.0
            exit_size = pos_remaining_size
            partial_exit = False
            ml_triggered_exit = False

            # Get current high/low for this bar
            bar_high = highs[bar]
            bar_low = lows[bar]

            # === Time-based exit ===
            if max_bars[pos_signal_idx] > 0:
                bars_in_trade = bar - pos_start_bar
                if bars_in_trade >= max_bars[pos_signal_idx]:
                    exit_price = closes[bar]
                    exited = True

            # === V5: Stale trade exit ===
            if not exited and stale_exit_bars[pos_signal_idx] > 0:
                bars_since_entry = bar - pos_start_bar
                if bars_since_entry >= stale_exit_bars[pos_signal_idx]:
                    # Check if price hasn't moved significantly (< 0.5R from entry)
                    half_r = (atr_pips[pos_signal_idx] * pip_size) * 0.5
                    if pos_dir == 1:
                        move = closes[bar] - pos_entry
                    else:
                        move = pos_entry - closes[bar]
                    if move < half_r:
                        exit_price = closes[bar]
                        exited = True

            # === V6: ML-based exit ===
            if not exited and use_ml_exit[pos_signal_idx]:
                bars_held = bar - pos_start_bar
                if bars_held >= ml_min_hold[pos_signal_idx]:
                    if pos_dir == 1:
                        score = ml_long_scores[bar]
                    else:
                        score = ml_short_scores[bar]
                    if score >= ml_threshold[pos_signal_idx]:
                        exit_price = closes[bar]
                        exited = True
                        ml_triggered_exit = True

            if not exited:
                # Fix Finding 3: Check SL/TP BEFORE management adjustments
                # to avoid intrabar look-ahead bias. Management adjustments
                # apply to the NEXT bar only.
                # === SL/TP check (using pre-adjustment levels) ===
                if pos_dir == 1:  # Long
                    if bar_low <= pos_sl:
                        exit_price = pos_sl
                        exited = True
                    elif bar_high >= pos_tp:
                        exit_price = pos_tp
                        exited = True
                else:  # Short
                    if bar_high >= pos_sl:
                        exit_price = pos_sl
                        exited = True
                    elif bar_low <= pos_tp:
                        exit_price = pos_tp
                        exited = True

            # === Apply management adjustments for NEXT bar (only if not exited) ===
            if not exited and in_pos:
                # === Breakeven logic (V5: no chaining, just wider triggers) ===
                if use_breakeven[pos_signal_idx] and not pos_be_triggered:
                    be_trigger = be_trigger_pips[pos_signal_idx] * pip_size
                    be_offset = be_offset_pips[pos_signal_idx] * pip_size

                    if pos_dir == 1:  # Long
                        if bar_high - pos_entry >= be_trigger:
                            new_sl = pos_entry + be_offset
                            if new_sl > pos_sl:
                                pos_sl = new_sl
                            pos_be_triggered = True
                    else:  # Short
                        if pos_entry - bar_low >= be_trigger:
                            new_sl = pos_entry - be_offset
                            if new_sl < pos_sl:
                                pos_sl = new_sl
                            pos_be_triggered = True

                # === Trailing stop logic: branch on trail_mode ===
                if use_trailing[pos_signal_idx]:
                    if trail_mode[pos_signal_idx] == 0:
                        # === Fixed pip trailing (V4 behavior) ===
                        trail_start = trail_start_pips[pos_signal_idx] * pip_size
                        trail_step = trail_step_pips[pos_signal_idx] * pip_size

                        if pos_dir == 1:  # Long
                            current_profit = bar_high - pos_entry
                            if not pos_trail_active and current_profit >= trail_start:
                                pos_trail_active = True
                                pos_trail_high = bar_high
                            if pos_trail_active:
                                if bar_high > pos_trail_high:
                                    pos_trail_high = bar_high
                                new_sl = pos_trail_high - trail_step
                                if new_sl > pos_sl:
                                    pos_sl = new_sl
                        else:  # Short
                            current_profit = pos_entry - bar_low
                            if not pos_trail_active and current_profit >= trail_start:
                                pos_trail_active = True
                                pos_trail_high = bar_low
                            if pos_trail_active:
                                if pos_trail_high == 0.0 or bar_low < pos_trail_high:
                                    pos_trail_high = bar_low
                                new_sl = pos_trail_high + trail_step
                                if new_sl < pos_sl:
                                    pos_sl = new_sl

                    elif trail_mode[pos_signal_idx] == 1:
                        # === V5: Chandelier Exit (ATR-adaptive, active from bar 1) ===
                        ch_dist = chandelier_atr_mult[pos_signal_idx] * atr_pips[pos_signal_idx] * pip_size
                        if pos_dir == 1:  # Long
                            if bar_high > pos_trail_high or pos_trail_high == 0.0:
                                pos_trail_high = bar_high
                            new_sl = pos_trail_high - ch_dist
                            if new_sl > pos_sl:
                                pos_sl = new_sl
                                pos_trail_active = True
                        else:  # Short
                            if pos_trail_high == 0.0 or bar_low < pos_trail_high:
                                pos_trail_high = bar_low
                            new_sl = pos_trail_high + ch_dist
                            if new_sl < pos_sl:
                                pos_sl = new_sl
                                pos_trail_active = True

                # === Partial close logic ===
                # Fix Finding 2: accumulate partial PnL into parent trade
                if use_partial[pos_signal_idx] and not pos_partial_done:
                    partial_target = partial_target_pips[pos_signal_idx] * pip_size
                    pct = partial_pct[pos_signal_idx]

                    if pos_dir == 1:  # Long
                        if bar_high - pos_entry >= partial_target:
                            partial_close_size = pos_size * pct
                            partial_pnl = partial_target * partial_close_size * pip_value

                            # Accumulate partial PnL - will be added to final exit trade
                            pos_partial_pnl += partial_pnl
                            equity += partial_pnl
                            daily_pnl += partial_pnl

                            pos_remaining_size = pos_size * (1.0 - pct)
                            pos_partial_done = True

                            if equity > peak_equity:
                                peak_equity = equity
                            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                            if dd > max_dd:
                                max_dd = dd
                    else:  # Short
                        if pos_entry - bar_low >= partial_target:
                            partial_close_size = pos_size * pct
                            partial_pnl = partial_target * partial_close_size * pip_value

                            pos_partial_pnl += partial_pnl
                            equity += partial_pnl
                            daily_pnl += partial_pnl

                            pos_remaining_size = pos_size * (1.0 - pct)
                            pos_partial_done = True

                            if equity > peak_equity:
                                peak_equity = equity
                            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                            if dd > max_dd:
                                max_dd = dd

            # === Process exit ===
            if exited:
                if pos_dir == 1:
                    pnl = (exit_price - pos_entry) * pos_remaining_size * pip_value
                else:
                    pnl = (pos_entry - exit_price) * pos_remaining_size * pip_value

                # Fix Finding 2: record combined PnL (partial + remainder) as one trade
                # Note: equity already includes pos_partial_pnl from earlier, so only
                # add the remaining portion's PnL to equity here.
                pnls[n_trades] = pnl + pos_partial_pnl
                equity += pnl
                equity_curve[n_trades] = equity
                n_trades += 1
                daily_pnl += pnl

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                # Reset position
                in_pos = False
                pos_be_triggered = False
                pos_trail_active = False
                pos_trail_high = 0.0
                pos_partial_done = False
                pos_partial_pnl = 0.0

                # V6.2: Start cooldown if ML triggered the exit
                if ml_triggered_exit and ml_exit_cooldown_bars > 0:
                    cooldown_remaining = ml_exit_cooldown_bars

        # V6.2: Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # === Check daily limits ===
        skip_entry = False
        if cooldown_remaining > 0:
            skip_entry = True
        if max_daily_trades > 0 and daily_trades >= max_daily_trades:
            skip_entry = True
        if max_daily_loss_pct > 0:
            daily_loss = (daily_start_equity - equity) / daily_start_equity * 100
            if daily_loss >= max_daily_loss_pct:
                skip_entry = True

        # === Check for new entry ===
        if not in_pos and not skip_entry and sig_idx < n_signals:
            if entry_bars[sig_idx] == bar:
                # Calculate position size based on risk
                sl_dist = abs(entry_prices[sig_idx] - sl_prices[sig_idx])
                risk_amt = equity * risk_pct / 100.0
                pos_size = risk_amt / (sl_dist * pip_value) if sl_dist > 0 else 0.01

                # V2: Apply quality multiplier if provided
                if len(quality_mult) > 0 and sig_idx < len(quality_mult):
                    pos_size *= quality_mult[sig_idx]

                # Set position
                pos_entry = entry_prices[sig_idx]
                pos_sl = sl_prices[sig_idx]
                pos_tp = tp_prices[sig_idx]
                pos_dir = directions[sig_idx]
                pos_start_bar = bar
                pos_signal_idx = sig_idx
                pos_remaining_size = pos_size
                in_pos = True
                daily_trades += 1

                # Reset management state
                pos_be_triggered = False
                pos_trail_active = False
                pos_trail_high = 0.0
                pos_partial_done = False
                pos_partial_pnl = 0.0

                sig_idx += 1

        # Skip past signals if we couldn't enter
        while sig_idx < n_signals and entry_bars[sig_idx] < bar:
            sig_idx += 1

    # === Force close remaining position at end ===
    if in_pos and n_bars > 0:
        exit_price = closes[n_bars - 1]
        if pos_dir == 1:
            pnl = (exit_price - pos_entry) * pos_remaining_size * pip_value
        else:
            pnl = (pos_entry - exit_price) * pos_remaining_size * pip_value
        # Fix Finding 2: include accumulated partial PnL in single trade record
        pnls[n_trades] = pnl + pos_partial_pnl
        equity += pnl
        equity_curve[n_trades] = equity
        n_trades += 1

        # Track drawdown after force close
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # === Calculate metrics ===
    if n_trades == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls = pnls[:n_trades]
    equity_curve = equity_curve[:n_trades]

    # Win rate and profit factor
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n_trades):
        if pnls[i] > 0:
            wins += 1
            gross_profit += pnls[i]
        else:
            gross_loss -= pnls[i]

    win_rate = wins / n_trades
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe ratio - FIX V4: annualize by actual trade frequency, not sqrt(252)
    # sqrt(252) assumes daily returns, but we have per-trade returns.
    # Correct: sqrt(trades_per_year) where trades_per_year = n_trades / (n_bars / bars_per_year)
    # For H1 forex: ~5544 bars/year (252 days * ~22 trading hours)
    mean_pnl = 0.0
    for i in range(n_trades):
        mean_pnl += pnls[i]
    mean_pnl /= n_trades

    var = 0.0
    for i in range(n_trades):
        var += (pnls[i] - mean_pnl) ** 2
    std = np.sqrt(var / (n_trades - 1)) if n_trades > 1 else 1.0

    # Fix Finding 11: bars_per_year is now a parameter instead of hardcoded
    trades_per_year = n_trades * (bars_per_year / n_bars) if n_bars > 0 else n_trades
    sharpe = np.sqrt(trades_per_year) * mean_pnl / std if std > 0 else 0.0

    total_profit = equity - initial_capital
    total_ret = total_profit / initial_capital * 100

    # R² of equity curve
    r_squared = calculate_r_squared(equity_curve)

    # OnTester score
    ontester = calculate_ontester_score(total_profit, r_squared, pf, n_trades, max_dd * 100)

    return (n_trades, win_rate, pf, sharpe, max_dd * 100, total_ret, r_squared, ontester)


@njit(cache=True, fastmath=True)
def basic_backtest_numba(
    entry_bars: np.ndarray,
    entry_prices: np.ndarray,
    directions: np.ndarray,
    sl_prices: np.ndarray,
    tp_prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    initial_capital: float,
    risk_pct: float,
    pip_size: float = 0.0001,  # FIX: Added pip_size parameter for JPY pairs
    quote_conversion_rate: float = 1.0,  # FIX V3: Quote currency to account currency rate
    bars_per_year: float = 5544.0,  # Fix Finding 11: configurable for different timeframes
) -> Tuple[int, float, float, float, float, float, float, float]:
    """
    Basic Numba backtest - SL/TP only, maximum speed.

    Use this for quick optimization of signal parameters,
    then use full_backtest_numba for management parameter optimization.

    FIX V3: quote_conversion_rate corrects pip value for cross-currency pairs.
    - USD-quoted pairs (GBP/USD, EUR/USD) with USD account: rate = 1.0
    - JPY-quoted pairs (USD/JPY) with USD account: rate = 1/USD_JPY (approx 0.0067)
    - GBP-quoted pairs (EUR/GBP) with USD account: rate = GBP/USD (approx 1.27)

    Returns: (trades, win_rate, profit_factor, sharpe, max_dd, total_return, r_squared, ontester_score)
    """
    n_signals = len(entry_bars)
    if n_signals == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # FIX V3: pip_value now includes quote currency conversion
    pip_value = (1.0 / pip_size) * quote_conversion_rate

    pnls = np.zeros(n_signals, dtype=np.float64)
    equity_curve = np.zeros(n_signals, dtype=np.float64)
    n_trades = 0
    equity = initial_capital
    peak_equity = initial_capital
    max_dd = 0.0

    sig_idx = 0
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0

    n_bars = len(highs)

    for bar in range(n_bars):
        # Check exit
        if in_pos:
            exited = False
            exit_price = 0.0

            if pos_dir == 1:  # Long
                if lows[bar] <= pos_sl:
                    exit_price = pos_sl
                    exited = True
                elif highs[bar] >= pos_tp:
                    exit_price = pos_tp
                    exited = True
            else:  # Short
                if highs[bar] >= pos_sl:
                    exit_price = pos_sl
                    exited = True
                elif lows[bar] <= pos_tp:
                    exit_price = pos_tp
                    exited = True

            if exited:
                if pos_dir == 1:
                    pnl = (exit_price - pos_entry) * pos_size * pip_value
                else:
                    pnl = (pos_entry - exit_price) * pos_size * pip_value

                pnls[n_trades] = pnl
                equity += pnl
                equity_curve[n_trades] = equity
                n_trades += 1

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_dd:
                    max_dd = dd

                in_pos = False

        # Check entry
        if not in_pos and sig_idx < n_signals:
            if entry_bars[sig_idx] == bar:
                sl_dist = abs(entry_prices[sig_idx] - sl_prices[sig_idx])
                risk_amt = equity * risk_pct / 100.0
                pos_size = risk_amt / (sl_dist * pip_value) if sl_dist > 0 else 0.01

                pos_entry = entry_prices[sig_idx]
                pos_sl = sl_prices[sig_idx]
                pos_tp = tp_prices[sig_idx]
                pos_dir = directions[sig_idx]
                in_pos = True
                sig_idx += 1

        # Advance past expired signals (matching full_backtest_numba behavior)
        while sig_idx < n_signals and entry_bars[sig_idx] < bar:
            sig_idx += 1

    # Force close remaining position
    if in_pos and n_bars > 0:
        exit_price = closes[n_bars - 1]
        if pos_dir == 1:
            pnl = (exit_price - pos_entry) * pos_size * pip_value
        else:
            pnl = (pos_entry - exit_price) * pos_size * pip_value
        pnls[n_trades] = pnl
        equity += pnl
        equity_curve[n_trades] = equity
        n_trades += 1

        # Track drawdown after force close
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    if n_trades == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Calculate metrics
    pnls = pnls[:n_trades]
    equity_curve = equity_curve[:n_trades]

    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n_trades):
        if pnls[i] > 0:
            wins += 1
            gross_profit += pnls[i]
        else:
            gross_loss -= pnls[i]

    win_rate = wins / n_trades
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe ratio - FIX V4: annualize by actual trade frequency
    mean_pnl = 0.0
    for i in range(n_trades):
        mean_pnl += pnls[i]
    mean_pnl /= n_trades

    var = 0.0
    for i in range(n_trades):
        var += (pnls[i] - mean_pnl) ** 2
    std = np.sqrt(var / (n_trades - 1)) if n_trades > 1 else 1.0

    # Fix Finding 11: bars_per_year is now a parameter instead of hardcoded
    trades_per_year = n_trades * (bars_per_year / n_bars) if n_bars > 0 else n_trades
    sharpe = np.sqrt(trades_per_year) * mean_pnl / std if std > 0 else 0.0

    total_profit = equity - initial_capital
    total_ret = total_profit / initial_capital * 100

    # R² of equity curve
    r_squared = calculate_r_squared(equity_curve)

    # OnTester score
    ontester = calculate_ontester_score(total_profit, r_squared, pf, n_trades, max_dd * 100)

    return (n_trades, win_rate, pf, sharpe, max_dd * 100, total_ret, r_squared, ontester)


@njit(cache=True, fastmath=True)
def full_backtest_with_trades(
    # === Signal arrays (per signal) ===
    entry_bars: np.ndarray,         # Bar index to enter
    entry_prices: np.ndarray,       # Entry price
    directions: np.ndarray,         # 1=buy, -1=sell
    sl_prices: np.ndarray,          # Stop loss price
    tp_prices: np.ndarray,          # Take profit price

    # === Trade management arrays (per signal) ===
    use_trailing: np.ndarray,       # bool: enable trailing for this trade
    trail_start_pips: np.ndarray,   # Pips profit before trailing starts
    trail_step_pips: np.ndarray,    # Trail step size in pips
    use_breakeven: np.ndarray,      # bool: enable breakeven
    be_trigger_pips: np.ndarray,    # Pips profit to trigger BE
    be_offset_pips: np.ndarray,     # Pips to lock in at BE
    use_partial: np.ndarray,        # bool: enable partial close
    partial_pct: np.ndarray,        # Percent to close at TP1 (0-1)
    partial_target_pips: np.ndarray,  # Pips for partial TP
    max_bars: np.ndarray,           # Max bars in trade (0=unlimited)

    # === V5: New exit management arrays (per signal) ===
    trail_mode: np.ndarray,           # int: 0=fixed, 1=chandelier
    chandelier_atr_mult: np.ndarray,  # float: ATR multiplier for chandelier
    atr_pips: np.ndarray,             # float: ATR in pips per signal
    stale_exit_bars: np.ndarray,      # int: max bars without progress (0=disabled)

    # === V6: ML exit arrays ===
    ml_long_scores: np.ndarray,    # float64 (n_bars,) — pre-computed long exit scores
    ml_short_scores: np.ndarray,   # float64 (n_bars,) — pre-computed short exit scores
    use_ml_exit: np.ndarray,       # bool (n_signals,) — enable ML exit per signal
    ml_min_hold: np.ndarray,       # int64 (n_signals,) — min bars before ML can trigger
    ml_threshold: np.ndarray,      # float64 (n_signals,) — score threshold to exit

    # === Market data ===
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    days: np.ndarray,               # Day of week (0=Mon, 6=Sun)

    # === Account params ===
    initial_capital: float,
    risk_pct: float,
    pip_size: float,
    max_daily_trades: int,          # 0 = unlimited
    max_daily_loss_pct: float,      # 0 = unlimited

    # === V2: Quality-based sizing ===
    quality_mult: np.ndarray,  # Position size multiplier per signal

    # === V3: Cross-currency pip value correction ===
    quote_conversion_rate: float = 1.0,  # Quote currency to account currency rate

    # === Fix Finding 11: configurable bars_per_year for Sharpe annualization ===
    bars_per_year: float = 5544.0,  # H1 default (252 days * ~22 hours)
) -> Tuple[np.ndarray, np.ndarray, int, float, float, float, float, float, float, float]:
    """
    Full-featured backtest that RETURNS trade PnL array for Monte Carlo analysis.

    V5: Replaced chain_be_to_trail with trail_mode, chandelier_atr_mult, atr_pips, stale_exit_bars.
    V6: Added ML exit arrays (ml_long_scores, ml_short_scores, use_ml_exit, ml_min_hold, ml_threshold).

    Returns: (pnls, equity_curve, trades, win_rate, profit_factor, sharpe, max_dd, total_return, r_squared, ontester_score)
    """
    n_signals = len(entry_bars)
    if n_signals == 0:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64),
                0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n_bars = len(highs)
    pip_value = (1.0 / pip_size) * quote_conversion_rate

    # Trade tracking
    max_trades = n_signals * 2
    pnls = np.zeros(max_trades, dtype=np.float64)
    equity_curve = np.zeros(max_trades, dtype=np.float64)
    n_trades = 0

    # Account tracking
    equity = initial_capital
    peak_equity = initial_capital
    max_dd = 0.0

    # Daily tracking
    current_day = -1
    daily_trades = 0
    daily_pnl = 0.0
    daily_start_equity = initial_capital

    # Position state
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0
    pos_start_bar = 0
    pos_signal_idx = -1

    # Position management state
    pos_be_triggered = False
    pos_trail_active = False
    pos_trail_high = 0.0
    pos_partial_done = False
    pos_remaining_size = 0.0
    # Fix Finding 2: accumulate partial close PnL instead of recording separate trade
    pos_partial_pnl = 0.0

    sig_idx = 0

    for bar in range(n_bars):
        # Daily reset
        if days[bar] != current_day:
            current_day = days[bar]
            daily_trades = 0
            daily_pnl = 0.0
            daily_start_equity = equity

        # Check position exit
        if in_pos:
            exited = False
            exit_price = 0.0
            exit_size = pos_remaining_size

            bar_high = highs[bar]
            bar_low = lows[bar]

            # Time-based exit
            if max_bars[pos_signal_idx] > 0:
                bars_in_trade = bar - pos_start_bar
                if bars_in_trade >= max_bars[pos_signal_idx]:
                    exit_price = closes[bar]
                    exited = True

            # === V5: Stale trade exit ===
            if not exited and stale_exit_bars[pos_signal_idx] > 0:
                bars_since_entry = bar - pos_start_bar
                if bars_since_entry >= stale_exit_bars[pos_signal_idx]:
                    half_r = (atr_pips[pos_signal_idx] * pip_size) * 0.5
                    if pos_dir == 1:
                        move = closes[bar] - pos_entry
                    else:
                        move = pos_entry - closes[bar]
                    if move < half_r:
                        exit_price = closes[bar]
                        exited = True

            # === V6: ML-based exit ===
            if not exited and use_ml_exit[pos_signal_idx]:
                bars_held = bar - pos_start_bar
                if bars_held >= ml_min_hold[pos_signal_idx]:
                    if pos_dir == 1:
                        score = ml_long_scores[bar]
                    else:
                        score = ml_short_scores[bar]
                    if score >= ml_threshold[pos_signal_idx]:
                        exit_price = closes[bar]
                        exited = True

            if not exited:
                # Fix Finding 3: Check SL/TP BEFORE management adjustments
                # to avoid intrabar look-ahead bias.
                # === SL/TP check (using pre-adjustment levels) ===
                if pos_dir == 1:
                    if bar_low <= pos_sl:
                        exit_price = pos_sl
                        exited = True
                    elif bar_high >= pos_tp:
                        exit_price = pos_tp
                        exited = True
                else:
                    if bar_high >= pos_sl:
                        exit_price = pos_sl
                        exited = True
                    elif bar_low <= pos_tp:
                        exit_price = pos_tp
                        exited = True

            # === Apply management adjustments for NEXT bar (only if not exited) ===
            if not exited and in_pos:
                # === Breakeven logic (V5: no chaining, just wider triggers) ===
                if use_breakeven[pos_signal_idx] and not pos_be_triggered:
                    be_trigger = be_trigger_pips[pos_signal_idx] * pip_size
                    be_offset = be_offset_pips[pos_signal_idx] * pip_size

                    if pos_dir == 1:
                        if bar_high - pos_entry >= be_trigger:
                            new_sl = pos_entry + be_offset
                            if new_sl > pos_sl:
                                pos_sl = new_sl
                            pos_be_triggered = True
                    else:
                        if pos_entry - bar_low >= be_trigger:
                            new_sl = pos_entry - be_offset
                            if new_sl < pos_sl:
                                pos_sl = new_sl
                            pos_be_triggered = True

                # === Trailing stop logic: branch on trail_mode ===
                if use_trailing[pos_signal_idx]:
                    if trail_mode[pos_signal_idx] == 0:
                        # === Fixed pip trailing (V4 behavior) ===
                        trail_start = trail_start_pips[pos_signal_idx] * pip_size
                        trail_step = trail_step_pips[pos_signal_idx] * pip_size

                        if pos_dir == 1:
                            current_profit = bar_high - pos_entry
                            if not pos_trail_active and current_profit >= trail_start:
                                pos_trail_active = True
                                pos_trail_high = bar_high
                            if pos_trail_active:
                                if bar_high > pos_trail_high:
                                    pos_trail_high = bar_high
                                new_sl = pos_trail_high - trail_step
                                if new_sl > pos_sl:
                                    pos_sl = new_sl
                        else:
                            current_profit = pos_entry - bar_low
                            if not pos_trail_active and current_profit >= trail_start:
                                pos_trail_active = True
                                pos_trail_high = bar_low
                            if pos_trail_active:
                                if pos_trail_high == 0.0 or bar_low < pos_trail_high:
                                    pos_trail_high = bar_low
                                new_sl = pos_trail_high + trail_step
                                if new_sl < pos_sl:
                                    pos_sl = new_sl

                    elif trail_mode[pos_signal_idx] == 1:
                        # === V5: Chandelier Exit (ATR-adaptive, active from bar 1) ===
                        ch_dist = chandelier_atr_mult[pos_signal_idx] * atr_pips[pos_signal_idx] * pip_size
                        if pos_dir == 1:
                            if bar_high > pos_trail_high or pos_trail_high == 0.0:
                                pos_trail_high = bar_high
                            new_sl = pos_trail_high - ch_dist
                            if new_sl > pos_sl:
                                pos_sl = new_sl
                                pos_trail_active = True
                        else:
                            if pos_trail_high == 0.0 or bar_low < pos_trail_high:
                                pos_trail_high = bar_low
                            new_sl = pos_trail_high + ch_dist
                            if new_sl < pos_sl:
                                pos_sl = new_sl
                                pos_trail_active = True

                # Partial close logic
                # Fix Finding 2: accumulate partial PnL into parent trade
                if use_partial[pos_signal_idx] and not pos_partial_done:
                    partial_target = partial_target_pips[pos_signal_idx] * pip_size
                    pct = partial_pct[pos_signal_idx]

                    if pos_dir == 1:
                        if bar_high - pos_entry >= partial_target:
                            partial_close_size = pos_size * pct
                            partial_pnl = partial_target * partial_close_size * pip_value

                            pos_partial_pnl += partial_pnl
                            equity += partial_pnl
                            daily_pnl += partial_pnl

                            pos_remaining_size = pos_size * (1.0 - pct)
                            pos_partial_done = True

                            if equity > peak_equity:
                                peak_equity = equity
                            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                            if dd > max_dd:
                                max_dd = dd
                    else:
                        if pos_entry - bar_low >= partial_target:
                            partial_close_size = pos_size * pct
                            partial_pnl = partial_target * partial_close_size * pip_value

                            pos_partial_pnl += partial_pnl
                            equity += partial_pnl
                            daily_pnl += partial_pnl

                            pos_remaining_size = pos_size * (1.0 - pct)
                            pos_partial_done = True

                            if equity > peak_equity:
                                peak_equity = equity
                            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                            if dd > max_dd:
                                max_dd = dd

            # Process exit
            if exited:
                if pos_dir == 1:
                    pnl = (exit_price - pos_entry) * pos_remaining_size * pip_value
                else:
                    pnl = (pos_entry - exit_price) * pos_remaining_size * pip_value

                # Fix Finding 2: record combined PnL (partial + remainder) as one trade
                pnls[n_trades] = pnl + pos_partial_pnl
                equity += pnl
                equity_curve[n_trades] = equity
                n_trades += 1
                daily_pnl += pnl

                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                in_pos = False
                pos_be_triggered = False
                pos_trail_active = False
                pos_trail_high = 0.0
                pos_partial_done = False
                pos_partial_pnl = 0.0

        # Check daily limits
        skip_entry = False
        if max_daily_trades > 0 and daily_trades >= max_daily_trades:
            skip_entry = True
        if max_daily_loss_pct > 0:
            daily_loss = (daily_start_equity - equity) / daily_start_equity * 100
            if daily_loss >= max_daily_loss_pct:
                skip_entry = True

        # Check for new entry
        if not in_pos and not skip_entry and sig_idx < n_signals:
            if entry_bars[sig_idx] == bar:
                sl_dist = abs(entry_prices[sig_idx] - sl_prices[sig_idx])
                risk_amt = equity * risk_pct / 100.0
                pos_size = risk_amt / (sl_dist * pip_value) if sl_dist > 0 else 0.01

                if len(quality_mult) > 0 and sig_idx < len(quality_mult):
                    pos_size *= quality_mult[sig_idx]

                pos_entry = entry_prices[sig_idx]
                pos_sl = sl_prices[sig_idx]
                pos_tp = tp_prices[sig_idx]
                pos_dir = directions[sig_idx]
                pos_start_bar = bar
                pos_signal_idx = sig_idx
                pos_remaining_size = pos_size
                in_pos = True
                daily_trades += 1

                pos_be_triggered = False
                pos_trail_active = False
                pos_trail_high = 0.0
                pos_partial_done = False
                pos_partial_pnl = 0.0

                sig_idx += 1

        # Skip past signals
        while sig_idx < n_signals and entry_bars[sig_idx] < bar:
            sig_idx += 1

    # Force close remaining position
    if in_pos and n_bars > 0:
        exit_price = closes[n_bars - 1]
        if pos_dir == 1:
            pnl = (exit_price - pos_entry) * pos_remaining_size * pip_value
        else:
            pnl = (pos_entry - exit_price) * pos_remaining_size * pip_value
        # Fix Finding 2: include accumulated partial PnL in single trade record
        pnls[n_trades] = pnl + pos_partial_pnl
        equity += pnl
        equity_curve[n_trades] = equity
        n_trades += 1

        # Track drawdown after force close
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Calculate metrics
    if n_trades == 0:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64),
                0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls_out = pnls[:n_trades].copy()
    equity_out = equity_curve[:n_trades].copy()

    # Win rate and profit factor
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n_trades):
        if pnls_out[i] > 0:
            wins += 1
            gross_profit += pnls_out[i]
        else:
            gross_loss -= pnls_out[i]

    win_rate = wins / n_trades
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe ratio - FIX V4: annualize by actual trade frequency
    mean_pnl = 0.0
    for i in range(n_trades):
        mean_pnl += pnls_out[i]
    mean_pnl /= n_trades

    var = 0.0
    for i in range(n_trades):
        var += (pnls_out[i] - mean_pnl) ** 2
    std = np.sqrt(var / (n_trades - 1)) if n_trades > 1 else 1.0

    # Fix Finding 11: bars_per_year is now a parameter instead of hardcoded
    trades_per_year = n_trades * (bars_per_year / n_bars) if n_bars > 0 else n_trades
    sharpe = np.sqrt(trades_per_year) * mean_pnl / std if std > 0 else 0.0

    total_profit = equity - initial_capital
    total_ret = total_profit / initial_capital * 100

    r_squared = calculate_r_squared(equity_out)
    ontester = calculate_ontester_score(total_profit, r_squared, pf, n_trades, max_dd * 100)

    return (pnls_out, equity_out, n_trades, win_rate, pf, sharpe, max_dd * 100, total_ret, r_squared, ontester)


@njit(cache=True, fastmath=True)
def full_backtest_with_telemetry(
    # === Signal arrays (per signal) ===
    entry_bars: np.ndarray,         # Bar index to enter
    entry_prices: np.ndarray,       # Entry price
    directions: np.ndarray,         # 1=buy, -1=sell
    sl_prices: np.ndarray,          # Stop loss price
    tp_prices: np.ndarray,          # Take profit price

    # === Trade management arrays (per signal) ===
    use_trailing: np.ndarray,       # bool: enable trailing for this trade
    trail_start_pips: np.ndarray,   # Pips profit before trailing starts
    trail_step_pips: np.ndarray,    # Trail step size in pips
    use_breakeven: np.ndarray,      # bool: enable breakeven
    be_trigger_pips: np.ndarray,    # Pips profit to trigger BE
    be_offset_pips: np.ndarray,     # Pips to lock in at BE
    use_partial: np.ndarray,        # bool: enable partial close
    partial_pct: np.ndarray,        # Percent to close at TP1 (0-1)
    partial_target_pips: np.ndarray,  # Pips for partial TP
    max_bars: np.ndarray,           # Max bars in trade (0=unlimited)

    # === V5: New exit management arrays (per signal) ===
    trail_mode: np.ndarray,           # int: 0=fixed, 1=chandelier
    chandelier_atr_mult: np.ndarray,  # float: ATR multiplier for chandelier
    atr_pips: np.ndarray,             # float: ATR in pips per signal
    stale_exit_bars: np.ndarray,      # int: max bars without progress (0=disabled)

    # === V6: ML exit arrays ===
    ml_long_scores: np.ndarray,    # float64 (n_bars,) -- pre-computed long exit scores
    ml_short_scores: np.ndarray,   # float64 (n_bars,) -- pre-computed short exit scores
    use_ml_exit: np.ndarray,       # bool (n_signals,) -- enable ML exit per signal
    ml_min_hold: np.ndarray,       # int64 (n_signals,) -- min bars before ML can trigger
    ml_threshold: np.ndarray,      # float64 (n_signals,) -- score threshold to exit

    # === Market data ===
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    days: np.ndarray,               # Day of week (0=Mon, 6=Sun)

    # === Account params ===
    initial_capital: float,
    risk_pct: float,
    pip_size: float,
    max_daily_trades: int,          # 0 = unlimited
    max_daily_loss_pct: float,      # 0 = unlimited

    # === V2: Quality-based sizing ===
    quality_mult: np.ndarray,  # Position size multiplier per signal

    # === V3: Cross-currency pip value correction ===
    quote_conversion_rate: float = 1.0,

    # === Fix Finding 11: configurable bars_per_year for Sharpe annualization ===
    bars_per_year: float = 5544.0,

    # === V6.2: ML exit cooldown ===
    ml_exit_cooldown_bars: int = 0,  # Bars to skip after ML exit (0=disabled)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, int, float, float, float, float, float, float, float]:
    """
    Full-featured backtest with trade telemetry for ML dataset building and diagnostics.

    Same logic as full_backtest_with_trades but returns additional per-trade arrays:
    - exit_reasons: int64 (0=sl, 1=tp, 2=trailing, 3=time, 4=stale, 5=ml, 6=force_close)
    - bars_held: int64 (number of bars each trade was held)
    - entry_bar_indices: int64 (bar index of entry)
    - exit_bar_indices: int64 (bar index of exit)
    - mfe_r: float64 (max favorable excursion in R-multiples)
    - mae_r: float64 (max adverse excursion in R-multiples)
    - signal_indices: int64 (maps trade_i to the original signal index)

    Returns: (pnls, equity_curve, exit_reasons, bars_held, entry_bar_indices,
              exit_bar_indices, mfe_r, mae_r, signal_indices,
              n_trades, win_rate, pf, sharpe, max_dd, total_return, r_squared, ontester)
    """
    n_signals = len(entry_bars)
    if n_signals == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_i = np.zeros(0, dtype=np.int64)
        return (empty_f, empty_f, empty_i, empty_i, empty_i, empty_i, empty_f, empty_f,
                empty_i, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n_bars = len(highs)
    pip_value = (1.0 / pip_size) * quote_conversion_rate

    # Trade tracking
    max_trades = n_signals * 2
    pnls = np.zeros(max_trades, dtype=np.float64)
    equity_curve = np.zeros(max_trades, dtype=np.float64)
    # Telemetry arrays
    exit_reasons = np.zeros(max_trades, dtype=np.int64)
    bars_held_arr = np.zeros(max_trades, dtype=np.int64)
    entry_bar_indices = np.zeros(max_trades, dtype=np.int64)
    exit_bar_indices = np.zeros(max_trades, dtype=np.int64)
    mfe_r = np.zeros(max_trades, dtype=np.float64)
    mae_r = np.zeros(max_trades, dtype=np.float64)
    signal_indices = np.zeros(max_trades, dtype=np.int64)  # Maps trade_i -> original signal index
    n_trades = 0

    # Account tracking
    equity = initial_capital
    peak_equity = initial_capital
    max_dd = 0.0

    # Daily tracking
    current_day = -1
    daily_trades = 0
    daily_pnl = 0.0
    daily_start_equity = initial_capital

    # Position state
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0
    pos_start_bar = 0
    pos_signal_idx = -1

    # Position management state
    pos_be_triggered = False
    pos_trail_active = False
    pos_trail_high = 0.0
    pos_partial_done = False
    pos_remaining_size = 0.0
    pos_partial_pnl = 0.0

    # Telemetry: MFE/MAE tracking per position
    pos_best_price = 0.0
    pos_worst_price = 0.0
    pos_sl_dist = 0.0  # SL distance at entry (for R-multiple calc)
    pos_exit_reason = 0  # default to SL

    sig_idx = 0

    # V6.2: ML exit cooldown counter
    cooldown_remaining = 0

    for bar in range(n_bars):
        # Daily reset
        if days[bar] != current_day:
            current_day = days[bar]
            daily_trades = 0
            daily_pnl = 0.0
            daily_start_equity = equity

        # Check position exit
        if in_pos:
            exited = False
            exit_price = 0.0
            exit_size = pos_remaining_size
            pos_exit_reason = 0  # default SL
            ml_triggered_exit = False

            bar_high = highs[bar]
            bar_low = lows[bar]

            # Update MFE/MAE tracking
            if bar_high > pos_best_price:
                pos_best_price = bar_high
            if bar_low < pos_worst_price:
                pos_worst_price = bar_low

            # Time-based exit
            if max_bars[pos_signal_idx] > 0:
                bars_in_trade = bar - pos_start_bar
                if bars_in_trade >= max_bars[pos_signal_idx]:
                    exit_price = closes[bar]
                    pos_exit_reason = 3  # time
                    exited = True

            # V5: Stale trade exit
            if not exited and stale_exit_bars[pos_signal_idx] > 0:
                bars_since_entry = bar - pos_start_bar
                if bars_since_entry >= stale_exit_bars[pos_signal_idx]:
                    half_r = (atr_pips[pos_signal_idx] * pip_size) * 0.5
                    if pos_dir == 1:
                        move = closes[bar] - pos_entry
                    else:
                        move = pos_entry - closes[bar]
                    if move < half_r:
                        exit_price = closes[bar]
                        pos_exit_reason = 4  # stale
                        exited = True

            # V6: ML-based exit
            if not exited and use_ml_exit[pos_signal_idx]:
                bars_held_now = bar - pos_start_bar
                if bars_held_now >= ml_min_hold[pos_signal_idx]:
                    if pos_dir == 1:
                        score = ml_long_scores[bar]
                    else:
                        score = ml_short_scores[bar]
                    if score >= ml_threshold[pos_signal_idx]:
                        exit_price = closes[bar]
                        pos_exit_reason = 5  # ml
                        exited = True
                        ml_triggered_exit = True

            if not exited:
                # Fix Finding 3: Check SL/TP BEFORE management adjustments
                if pos_dir == 1:
                    if bar_low <= pos_sl:
                        exit_price = pos_sl
                        # Distinguish trailing SL from initial SL
                        if pos_trail_active:
                            pos_exit_reason = 2  # trailing
                        else:
                            pos_exit_reason = 0  # sl
                        exited = True
                    elif bar_high >= pos_tp:
                        exit_price = pos_tp
                        pos_exit_reason = 1  # tp
                        exited = True
                else:
                    if bar_high >= pos_sl:
                        exit_price = pos_sl
                        if pos_trail_active:
                            pos_exit_reason = 2  # trailing
                        else:
                            pos_exit_reason = 0  # sl
                        exited = True
                    elif bar_low <= pos_tp:
                        exit_price = pos_tp
                        pos_exit_reason = 1  # tp
                        exited = True

            # Apply management adjustments for NEXT bar (only if not exited)
            if not exited and in_pos:
                # Breakeven logic
                if use_breakeven[pos_signal_idx] and not pos_be_triggered:
                    be_trigger = be_trigger_pips[pos_signal_idx] * pip_size
                    be_offset = be_offset_pips[pos_signal_idx] * pip_size

                    if pos_dir == 1:
                        if bar_high - pos_entry >= be_trigger:
                            new_sl = pos_entry + be_offset
                            if new_sl > pos_sl:
                                pos_sl = new_sl
                            pos_be_triggered = True
                    else:
                        if pos_entry - bar_low >= be_trigger:
                            new_sl = pos_entry - be_offset
                            if new_sl < pos_sl:
                                pos_sl = new_sl
                            pos_be_triggered = True

                # Trailing stop logic
                if use_trailing[pos_signal_idx]:
                    if trail_mode[pos_signal_idx] == 0:
                        # Fixed pip trailing
                        trail_start = trail_start_pips[pos_signal_idx] * pip_size
                        trail_step = trail_step_pips[pos_signal_idx] * pip_size

                        if pos_dir == 1:
                            current_profit = bar_high - pos_entry
                            if not pos_trail_active and current_profit >= trail_start:
                                pos_trail_active = True
                                pos_trail_high = bar_high
                            if pos_trail_active:
                                if bar_high > pos_trail_high:
                                    pos_trail_high = bar_high
                                new_sl = pos_trail_high - trail_step
                                if new_sl > pos_sl:
                                    pos_sl = new_sl
                        else:
                            current_profit = pos_entry - bar_low
                            if not pos_trail_active and current_profit >= trail_start:
                                pos_trail_active = True
                                pos_trail_high = bar_low
                            if pos_trail_active:
                                if pos_trail_high == 0.0 or bar_low < pos_trail_high:
                                    pos_trail_high = bar_low
                                new_sl = pos_trail_high + trail_step
                                if new_sl < pos_sl:
                                    pos_sl = new_sl

                    elif trail_mode[pos_signal_idx] == 1:
                        # V5: Chandelier Exit
                        ch_dist = chandelier_atr_mult[pos_signal_idx] * atr_pips[pos_signal_idx] * pip_size
                        if pos_dir == 1:
                            if bar_high > pos_trail_high or pos_trail_high == 0.0:
                                pos_trail_high = bar_high
                            new_sl = pos_trail_high - ch_dist
                            if new_sl > pos_sl:
                                pos_sl = new_sl
                                pos_trail_active = True
                        else:
                            if pos_trail_high == 0.0 or bar_low < pos_trail_high:
                                pos_trail_high = bar_low
                            new_sl = pos_trail_high + ch_dist
                            if new_sl < pos_sl:
                                pos_sl = new_sl
                                pos_trail_active = True

                # Partial close logic
                if use_partial[pos_signal_idx] and not pos_partial_done:
                    partial_target = partial_target_pips[pos_signal_idx] * pip_size
                    pct = partial_pct[pos_signal_idx]

                    if pos_dir == 1:
                        if bar_high - pos_entry >= partial_target:
                            partial_close_size = pos_size * pct
                            partial_pnl = partial_target * partial_close_size * pip_value

                            pos_partial_pnl += partial_pnl
                            equity += partial_pnl
                            daily_pnl += partial_pnl

                            pos_remaining_size = pos_size * (1.0 - pct)
                            pos_partial_done = True

                            if equity > peak_equity:
                                peak_equity = equity
                            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                            if dd > max_dd:
                                max_dd = dd
                    else:
                        if pos_entry - bar_low >= partial_target:
                            partial_close_size = pos_size * pct
                            partial_pnl = partial_target * partial_close_size * pip_value

                            pos_partial_pnl += partial_pnl
                            equity += partial_pnl
                            daily_pnl += partial_pnl

                            pos_remaining_size = pos_size * (1.0 - pct)
                            pos_partial_done = True

                            if equity > peak_equity:
                                peak_equity = equity
                            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                            if dd > max_dd:
                                max_dd = dd

            # Process exit
            if exited:
                if pos_dir == 1:
                    pnl = (exit_price - pos_entry) * pos_remaining_size * pip_value
                else:
                    pnl = (pos_entry - exit_price) * pos_remaining_size * pip_value

                pnls[n_trades] = pnl + pos_partial_pnl
                equity += pnl
                equity_curve[n_trades] = equity
                daily_pnl += pnl

                # Record telemetry
                exit_reasons[n_trades] = pos_exit_reason
                bars_held_arr[n_trades] = bar - pos_start_bar
                entry_bar_indices[n_trades] = pos_start_bar
                exit_bar_indices[n_trades] = bar
                signal_indices[n_trades] = pos_signal_idx

                # MFE/MAE in R-multiples
                if pos_sl_dist > 0.0:
                    if pos_dir == 1:  # Long
                        mfe_r[n_trades] = (pos_best_price - pos_entry) / pos_sl_dist
                        mae_r[n_trades] = (pos_entry - pos_worst_price) / pos_sl_dist
                    else:  # Short
                        mfe_r[n_trades] = (pos_entry - pos_worst_price) / pos_sl_dist
                        mae_r[n_trades] = (pos_best_price - pos_entry) / pos_sl_dist

                n_trades += 1

                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                in_pos = False
                pos_be_triggered = False
                pos_trail_active = False
                pos_trail_high = 0.0
                pos_partial_done = False
                pos_partial_pnl = 0.0

                # V6.2: Start cooldown if ML triggered the exit
                if ml_triggered_exit and ml_exit_cooldown_bars > 0:
                    cooldown_remaining = ml_exit_cooldown_bars

        # V6.2: Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Check daily limits
        skip_entry = False
        if cooldown_remaining > 0:
            skip_entry = True
        if max_daily_trades > 0 and daily_trades >= max_daily_trades:
            skip_entry = True
        if max_daily_loss_pct > 0:
            daily_loss = (daily_start_equity - equity) / daily_start_equity * 100
            if daily_loss >= max_daily_loss_pct:
                skip_entry = True

        # Check for new entry
        if not in_pos and not skip_entry and sig_idx < n_signals:
            if entry_bars[sig_idx] == bar:
                sl_dist = abs(entry_prices[sig_idx] - sl_prices[sig_idx])
                risk_amt = equity * risk_pct / 100.0
                pos_size = risk_amt / (sl_dist * pip_value) if sl_dist > 0 else 0.01

                if len(quality_mult) > 0 and sig_idx < len(quality_mult):
                    pos_size *= quality_mult[sig_idx]

                pos_entry = entry_prices[sig_idx]
                pos_sl = sl_prices[sig_idx]
                pos_tp = tp_prices[sig_idx]
                pos_dir = directions[sig_idx]
                pos_start_bar = bar
                pos_signal_idx = sig_idx
                pos_remaining_size = pos_size
                in_pos = True
                daily_trades += 1

                pos_be_triggered = False
                pos_trail_active = False
                pos_trail_high = 0.0
                pos_partial_done = False
                pos_partial_pnl = 0.0

                # Initialize telemetry for this position
                pos_best_price = highs[bar]
                pos_worst_price = lows[bar]
                pos_sl_dist = sl_dist

                sig_idx += 1

        # Skip past expired signals
        while sig_idx < n_signals and entry_bars[sig_idx] < bar:
            sig_idx += 1

    # Force close remaining position
    if in_pos and n_bars > 0:
        exit_price = closes[n_bars - 1]
        if pos_dir == 1:
            pnl = (exit_price - pos_entry) * pos_remaining_size * pip_value
        else:
            pnl = (pos_entry - exit_price) * pos_remaining_size * pip_value
        pnls[n_trades] = pnl + pos_partial_pnl
        equity += pnl
        equity_curve[n_trades] = equity

        # Telemetry for force close
        exit_reasons[n_trades] = 6  # force_close
        bars_held_arr[n_trades] = (n_bars - 1) - pos_start_bar
        entry_bar_indices[n_trades] = pos_start_bar
        exit_bar_indices[n_trades] = n_bars - 1
        signal_indices[n_trades] = pos_signal_idx
        if pos_sl_dist > 0.0:
            if pos_dir == 1:
                mfe_r[n_trades] = (pos_best_price - pos_entry) / pos_sl_dist
                mae_r[n_trades] = (pos_entry - pos_worst_price) / pos_sl_dist
            else:
                mfe_r[n_trades] = (pos_entry - pos_worst_price) / pos_sl_dist
                mae_r[n_trades] = (pos_best_price - pos_entry) / pos_sl_dist

        n_trades += 1

        # Track drawdown after force close
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Calculate metrics
    if n_trades == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_i = np.zeros(0, dtype=np.int64)
        return (empty_f, empty_f, empty_i, empty_i, empty_i, empty_i, empty_f, empty_f,
                empty_i, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls_out = pnls[:n_trades].copy()
    equity_out = equity_curve[:n_trades].copy()
    exit_reasons_out = exit_reasons[:n_trades].copy()
    bars_held_out = bars_held_arr[:n_trades].copy()
    entry_bars_out = entry_bar_indices[:n_trades].copy()
    exit_bars_out = exit_bar_indices[:n_trades].copy()
    mfe_r_out = mfe_r[:n_trades].copy()
    mae_r_out = mae_r[:n_trades].copy()
    signal_indices_out = signal_indices[:n_trades].copy()

    # Win rate and profit factor
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(n_trades):
        if pnls_out[i] > 0:
            wins += 1
            gross_profit += pnls_out[i]
        else:
            gross_loss -= pnls_out[i]

    win_rate = wins / n_trades
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Sharpe ratio
    mean_pnl = 0.0
    for i in range(n_trades):
        mean_pnl += pnls_out[i]
    mean_pnl /= n_trades

    var = 0.0
    for i in range(n_trades):
        var += (pnls_out[i] - mean_pnl) ** 2
    std = np.sqrt(var / (n_trades - 1)) if n_trades > 1 else 1.0

    trades_per_year = n_trades * (bars_per_year / n_bars) if n_bars > 0 else n_trades
    sharpe = np.sqrt(trades_per_year) * mean_pnl / std if std > 0 else 0.0

    total_profit = equity - initial_capital
    total_ret = total_profit / initial_capital * 100

    r_squared = calculate_r_squared(equity_out)
    ontester = calculate_ontester_score(total_profit, r_squared, pf, n_trades, max_dd * 100)

    return (pnls_out, equity_out, exit_reasons_out, bars_held_out, entry_bars_out,
            exit_bars_out, mfe_r_out, mae_r_out, signal_indices_out,
            n_trades, win_rate, pf, sharpe, max_dd * 100, total_ret, r_squared, ontester)


def get_quote_conversion_rate(pair: str, account_currency: str = 'USD') -> float:
    """
    Get the conversion rate from quote currency to account currency.

    FIX V3: This corrects pip value calculations for cross-currency pairs.

    For proper calculation, you'd need live exchange rates. This function provides
    reasonable approximations for common pairs with USD accounts.

    Args:
        pair: Currency pair (e.g., 'GBP_USD', 'USD_JPY', 'EUR_GBP')
        account_currency: Account currency (default 'USD')

    Returns:
        Conversion rate (quote currency to account currency)

    Examples:
        - GBP_USD with USD account: quote=USD, account=USD → 1.0
        - USD_JPY with USD account: quote=JPY, account=USD → ~0.0067 (1/150)
        - EUR_GBP with USD account: quote=GBP, account=USD → ~1.27
        - EUR_JPY with USD account: quote=JPY, account=USD → ~0.0067 (1/150)
    """
    # Parse quote currency from pair (second currency)
    parts = pair.replace('_', '/').replace('-', '/').split('/')
    if len(parts) != 2:
        return 1.0  # Default if can't parse

    quote_currency = parts[1].upper()

    # If quote currency matches account currency, no conversion needed
    if quote_currency == account_currency:
        return 1.0

    # Approximate conversion rates for common cases (USD account)
    # These are rough estimates - for accurate backtesting, use historical rates
    if account_currency == 'USD':
        conversion_rates = {
            'JPY': 0.0067,   # ~1/150 (USD/JPY ≈ 150)
            'GBP': 1.27,     # GBP/USD ≈ 1.27
            'EUR': 1.08,     # EUR/USD ≈ 1.08
            'AUD': 0.65,     # AUD/USD ≈ 0.65
            'NZD': 0.60,     # NZD/USD ≈ 0.60
            'CAD': 0.74,     # 1/USD_CAD ≈ 1/1.35
            'CHF': 1.12,     # 1/USD_CHF ≈ 1/0.89
        }
        return conversion_rates.get(quote_currency, 1.0)

    # For non-USD accounts, return 1.0 (user should override)
    return 1.0


def create_management_arrays(n_signals: int, params: dict, pip_size: float) -> dict:
    """
    Create management arrays from parameters.

    Helper function to convert parameter dict into numpy arrays
    suitable for the Numba backtest engine.

    Args:
        n_signals: Number of signals
        params: Parameter dictionary
        pip_size: Pip size for the instrument

    Returns:
        Dictionary of numpy arrays ready for full_backtest_numba
    """
    # Trailing stop
    use_trailing = np.full(n_signals, params.get('use_trailing', False), dtype=np.bool_)
    trail_start = np.full(n_signals, params.get('trail_start_pips', 30), dtype=np.float64)
    trail_step = np.full(n_signals, params.get('trail_step_pips', 15), dtype=np.float64)

    # Breakeven
    use_be = np.full(n_signals, params.get('use_break_even', False), dtype=np.bool_)
    be_trigger = np.full(n_signals, params.get('be_trigger_pips', 20), dtype=np.float64)
    be_offset = np.full(n_signals, params.get('be_offset_pips', 2), dtype=np.float64)

    # Partial close
    use_partial = np.full(n_signals, params.get('use_partial_close', False), dtype=np.bool_)
    partial_pct = np.full(n_signals, params.get('partial_close_pct', 0.5), dtype=np.float64)
    partial_target = np.full(n_signals, params.get('partial_target_pips', 20), dtype=np.float64)

    # Time exit
    max_bars = np.full(n_signals, params.get('max_bars_in_trade', 0), dtype=np.int64)

    # V5: Trail mode, chandelier, ATR, stale exit
    trail_mode = np.full(n_signals, params.get('trail_mode', 0), dtype=np.int64)
    chandelier_atr_mult_arr = np.full(n_signals, params.get('chandelier_atr_mult', 3.0), dtype=np.float64)
    atr_pips_arr = np.full(n_signals, params.get('atr_pips', 35.0), dtype=np.float64)
    stale_exit = np.full(n_signals, params.get('stale_exit_bars', 0), dtype=np.int64)

    # V6: ML exit arrays (defaults: disabled)
    use_ml_exit = np.full(n_signals, params.get('use_ml_exit', False), dtype=np.bool_)
    ml_min_hold = np.full(n_signals, params.get('ml_min_hold_bars', 5), dtype=np.int64)
    ml_threshold = np.full(n_signals, params.get('ml_exit_threshold', 0.5), dtype=np.float64)

    return {
        'use_trailing': use_trailing,
        'trail_start_pips': trail_start,
        'trail_step_pips': trail_step,
        'use_breakeven': use_be,
        'be_trigger_pips': be_trigger,
        'be_offset_pips': be_offset,
        'use_partial': use_partial,
        'partial_pct': partial_pct,
        'partial_target_pips': partial_target,
        'max_bars': max_bars,
        'trail_mode': trail_mode,
        'chandelier_atr_mult': chandelier_atr_mult_arr,
        'atr_pips': atr_pips_arr,
        'stale_exit_bars': stale_exit,
        'use_ml_exit': use_ml_exit,
        'ml_min_hold': ml_min_hold,
        'ml_threshold': ml_threshold,
    }
