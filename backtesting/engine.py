"""
Backtesting engine with realistic trade simulation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from strategies.base import Strategy, Signal, SignalType, Trade
from config.settings import settings


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade: float
    expectancy: float
    params: Dict[str, Any]

    def summary(self) -> str:
        """Return formatted summary string."""
        return f"""
========================================
BACKTEST RESULTS
========================================
Total Trades:    {self.total_trades}
Win Rate:        {self.win_rate:.1%}
Profit Factor:   {self.profit_factor:.2f}
Sharpe Ratio:    {self.sharpe_ratio:.2f}
----------------------------------------
Total Return:    {self.total_return:.2f} ({self.total_return_pct:.1%})
Max Drawdown:    {self.max_drawdown:.2f} ({self.max_drawdown_pct:.1%})
----------------------------------------
Avg Win:         {self.avg_win:.2f}
Avg Loss:        {self.avg_loss:.2f}
Avg Trade:       {self.avg_trade:.2f}
Expectancy:      {self.expectancy:.4f}
========================================
"""


@dataclass
class OpenPosition:
    """Tracks an open position during backtest."""
    entry_time: pd.Timestamp
    direction: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    signal: Signal


class BacktestEngine:
    """
    Backtesting engine that simulates strategy execution.

    Features:
    - Realistic spread simulation
    - Slippage modeling
    - Trade management (SL, TP)
    - Equity curve tracking
    - Performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        pip_value: float = 10.0,  # Value per pip for 1 lot
        leverage: float = 100.0,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting account balance
            spread_pips: Spread to apply on entries
            slippage_pips: Slippage to apply on entries
            pip_value: Dollar value per pip for 1 standard lot
            leverage: Account leverage
        """
        self.initial_capital = initial_capital
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.pip_value = pip_value
        self.leverage = leverage

    def run(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        risk_per_trade: float = 1.0,  # Percent of equity
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Strategy instance to test
            df: DataFrame with OHLCV data
            risk_per_trade: Risk per trade as percent of equity

        Returns:
            BacktestResult with performance metrics
        """
        # Determine pip size based on pair
        pip_size = 0.0001  # Default for most pairs
        # Could adjust for JPY pairs etc.

        # Generate signals
        signals = strategy.generate_signals(df)

        if not signals:
            logger.warning("No signals generated")
            return self._empty_result(strategy.params)

        # Initialize tracking
        equity = self.initial_capital
        equity_history = [equity]
        trades: List[Trade] = []
        position: Optional[OpenPosition] = None

        # Create signal lookup by timestamp
        signal_dict = {s.timestamp: s for s in signals}

        # Iterate through each bar
        for i in range(len(df)):
            bar_time = df.index[i]
            bar = df.iloc[i]

            # Check if we have an open position
            if position is not None:
                # Check for SL/TP hit using high/low of bar
                trade, closed = self._check_exit(
                    position, bar, bar_time, pip_size
                )

                if closed:
                    trades.append(trade)
                    equity += trade.pnl
                    equity_history.append(equity)
                    position = None

            # Check for new signal (only if flat)
            if position is None and bar_time in signal_dict:
                signal = signal_dict[bar_time]

                # Calculate position size based on risk
                risk_amount = equity * (risk_per_trade / 100.0)
                sl_pips = abs(signal.price - signal.stop_loss) / pip_size
                if sl_pips > 0:
                    size = risk_amount / (sl_pips * self.pip_value / 10)  # Convert to mini lots
                else:
                    size = 0.1  # Default small size

                # Apply spread and slippage to entry
                total_cost_pips = self.spread_pips + self.slippage_pips
                if signal.type == SignalType.BUY:
                    adjusted_entry = signal.price + (total_cost_pips * pip_size)
                else:
                    adjusted_entry = signal.price - (total_cost_pips * pip_size)

                position = OpenPosition(
                    entry_time=bar_time,
                    direction=signal.type,
                    entry_price=adjusted_entry,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    size=size,
                    signal=signal
                )

        # Close any remaining position at last bar
        if position is not None:
            last_bar = df.iloc[-1]
            trade = self._force_close(position, last_bar, df.index[-1], pip_size)
            trades.append(trade)
            equity += trade.pnl
            equity_history.append(equity)

        # Calculate metrics
        return self._calculate_results(
            trades, equity_history, strategy.params
        )

    def _check_exit(
        self,
        position: OpenPosition,
        bar: pd.Series,
        bar_time: pd.Timestamp,
        pip_size: float
    ) -> tuple[Optional[Trade], bool]:
        """Check if position should be closed this bar."""

        if position.direction == SignalType.BUY:
            # Long position - check if low hit SL or high hit TP
            if bar['low'] <= position.stop_loss:
                return self._create_trade(
                    position, position.stop_loss, bar_time, "SL", pip_size
                ), True
            elif bar['high'] >= position.take_profit:
                return self._create_trade(
                    position, position.take_profit, bar_time, "TP", pip_size
                ), True
        else:
            # Short position - check if high hit SL or low hit TP
            if bar['high'] >= position.stop_loss:
                return self._create_trade(
                    position, position.stop_loss, bar_time, "SL", pip_size
                ), True
            elif bar['low'] <= position.take_profit:
                return self._create_trade(
                    position, position.take_profit, bar_time, "TP", pip_size
                ), True

        return None, False

    def _force_close(
        self,
        position: OpenPosition,
        bar: pd.Series,
        bar_time: pd.Timestamp,
        pip_size: float
    ) -> Trade:
        """Force close position at bar close."""
        return self._create_trade(
            position, bar['close'], bar_time, "CLOSE", pip_size
        )

    def _create_trade(
        self,
        position: OpenPosition,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
        pip_size: float
    ) -> Trade:
        """Create a trade record."""
        if position.direction == SignalType.BUY:
            pnl_pips = (exit_price - position.entry_price) / pip_size
        else:
            pnl_pips = (position.entry_price - exit_price) / pip_size

        # PnL in currency (simplified - assumes pip value)
        pnl = pnl_pips * self.pip_value * position.size / 10  # Adjust for size

        return Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            size=position.size,
            pnl=pnl,
            pnl_pips=pnl_pips,
            exit_reason=exit_reason,
            metadata=position.signal.metadata
        )

    def _calculate_results(
        self,
        trades: List[Trade],
        equity_history: List[float],
        params: Dict[str, Any]
    ) -> BacktestResult:
        """Calculate performance metrics from trades."""

        if not trades:
            return self._empty_result(params)

        # Basic stats
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl <= 0)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL stats
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean([t.pnl for t in trades])

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Equity curve analysis
        equity_series = pd.Series(equity_history)
        total_return = equity_series.iloc[-1] - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        # Drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = equity_series - rolling_max
        max_drawdown = abs(drawdown.min())
        max_drawdown_pct = max_drawdown / rolling_max[drawdown.idxmin()] if len(drawdown) > 0 else 0

        # Sharpe ratio (simplified - assumes risk-free = 0)
        if len(trades) > 1:
            returns = pd.Series([t.pnl for t in trades])
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            expectancy=expectancy,
            params=params,
        )

    def _empty_result(self, params: Dict[str, Any]) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series([self.initial_capital]),
            total_return=0,
            total_return_pct=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            profit_factor=0,
            win_rate=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            expectancy=0,
            params=params,
        )


# Quick test
if __name__ == "__main__":
    print("Backtest engine loaded successfully")
