"""
Risk manager for live trading.

Performs pre-trade risk checks to ensure compliance with
risk limits before executing trades.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger

from config.settings import settings
from live.position_manager import PositionManager


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    passed: bool
    reason: str = ""
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class RiskManager:
    """
    Pre-trade risk manager.

    Checks before allowing a trade:
    - Max daily trades
    - Max daily loss
    - Max drawdown (circuit breaker)
    - Max open positions
    - Max spread
    """

    def __init__(
        self,
        position_manager: PositionManager,
        max_daily_trades: int = None,
        max_daily_loss_pct: float = None,
        max_drawdown_pct: float = None,
        pause_drawdown_pct: float = None,
        max_open_positions: int = None,
        max_spread_pips: float = None,
    ):
        """
        Initialize risk manager.

        Args:
            position_manager: Position manager instance
            max_daily_trades: Maximum trades per day (default from settings)
            max_daily_loss_pct: Maximum daily loss as % of equity
            max_drawdown_pct: Maximum drawdown % (circuit breaker)
            pause_drawdown_pct: Drawdown % to pause trading
            max_open_positions: Maximum concurrent positions
            max_spread_pips: Maximum allowed spread in pips
        """
        self.position_manager = position_manager

        # Load from settings or use provided values
        self.max_daily_trades = max_daily_trades or settings.MAX_DAILY_TRADES
        self.max_daily_loss_pct = max_daily_loss_pct or settings.MAX_DAILY_LOSS_PCT
        self.max_drawdown_pct = max_drawdown_pct or settings.MAX_DRAWDOWN_PCT
        self.pause_drawdown_pct = pause_drawdown_pct or settings.PAUSE_DRAWDOWN_PCT
        self.max_open_positions = max_open_positions or settings.MAX_OPEN_POSITIONS
        self.max_spread_pips = max_spread_pips or settings.MAX_SPREAD_PIPS

        # Track circuit breaker state
        self.circuit_breaker_tripped = False
        self.trading_paused = False

        logger.info(f"Risk manager initialized:")
        logger.info(f"  Max daily trades: {self.max_daily_trades}")
        logger.info(f"  Max daily loss: {self.max_daily_loss_pct}%")
        logger.info(f"  Max drawdown: {self.max_drawdown_pct}%")
        logger.info(f"  Pause drawdown: {self.pause_drawdown_pct}%")
        logger.info(f"  Max open positions: {self.max_open_positions}")
        logger.info(f"  Max spread: {self.max_spread_pips} pips")

    def check_daily_trades(self) -> RiskCheckResult:
        """Check if daily trade limit has been reached."""
        daily_trades = self.position_manager.daily_trade_count

        if daily_trades >= self.max_daily_trades:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily trade limit reached ({daily_trades}/{self.max_daily_trades})",
                details={'daily_trades': daily_trades, 'max': self.max_daily_trades}
            )

        return RiskCheckResult(
            passed=True,
            details={'daily_trades': daily_trades, 'max': self.max_daily_trades}
        )

    def check_daily_loss(self, current_balance: float) -> RiskCheckResult:
        """Check if daily loss limit has been reached."""
        stats = self.position_manager.daily_stats
        if not stats:
            return RiskCheckResult(passed=True, reason="No daily stats yet")

        daily_loss_pct = abs(min(0, stats.daily_return_pct))

        if daily_loss_pct >= self.max_daily_loss_pct:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily loss limit reached ({daily_loss_pct:.2f}% >= {self.max_daily_loss_pct}%)",
                details={
                    'daily_loss_pct': daily_loss_pct,
                    'max': self.max_daily_loss_pct,
                    'starting_balance': stats.starting_balance,
                    'current_balance': current_balance
                }
            )

        return RiskCheckResult(
            passed=True,
            details={'daily_loss_pct': daily_loss_pct, 'max': self.max_daily_loss_pct}
        )

    def check_drawdown(self, current_balance: float, peak_balance: float) -> RiskCheckResult:
        """
        Check drawdown levels.

        Returns:
            RiskCheckResult with passed=False if circuit breaker triggered
        """
        if peak_balance <= 0:
            return RiskCheckResult(passed=True)

        drawdown_pct = (peak_balance - current_balance) / peak_balance * 100

        # Circuit breaker - stop all trading
        if drawdown_pct >= self.max_drawdown_pct:
            self.circuit_breaker_tripped = True
            return RiskCheckResult(
                passed=False,
                reason=f"CIRCUIT BREAKER: Drawdown {drawdown_pct:.2f}% >= {self.max_drawdown_pct}%",
                details={
                    'drawdown_pct': drawdown_pct,
                    'max': self.max_drawdown_pct,
                    'circuit_breaker': True
                }
            )

        # Pause level - warning but allow trades
        if drawdown_pct >= self.pause_drawdown_pct:
            self.trading_paused = True
            logger.warning(f"Drawdown warning: {drawdown_pct:.2f}% >= {self.pause_drawdown_pct}%")
            # Still allow trading but log warning
            return RiskCheckResult(
                passed=True,
                reason=f"Drawdown warning: {drawdown_pct:.2f}%",
                details={
                    'drawdown_pct': drawdown_pct,
                    'pause_level': self.pause_drawdown_pct,
                    'warning': True
                }
            )

        # Reset pause flag if recovered
        if drawdown_pct < self.pause_drawdown_pct * 0.8:  # 80% of pause level
            self.trading_paused = False

        return RiskCheckResult(
            passed=True,
            details={'drawdown_pct': drawdown_pct, 'max': self.max_drawdown_pct}
        )

    def check_open_positions(self, instrument: Optional[str] = None) -> RiskCheckResult:
        """
        Check if position limit has been reached.

        Args:
            instrument: Optional instrument to check for existing position
        """
        open_count = self.position_manager.open_position_count

        if open_count >= self.max_open_positions:
            return RiskCheckResult(
                passed=False,
                reason=f"Position limit reached ({open_count}/{self.max_open_positions})",
                details={'open_positions': open_count, 'max': self.max_open_positions}
            )

        # Check if already have position in this instrument
        if instrument:
            existing = self.position_manager.get_positions_for_instrument(instrument)
            if existing:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Already have position in {instrument}",
                    details={'existing_positions': len(existing), 'instrument': instrument}
                )

        return RiskCheckResult(
            passed=True,
            details={'open_positions': open_count, 'max': self.max_open_positions}
        )

    def check_spread(self, current_spread_pips: float) -> RiskCheckResult:
        """Check if spread is within acceptable limits."""
        if current_spread_pips > self.max_spread_pips:
            return RiskCheckResult(
                passed=False,
                reason=f"Spread too wide ({current_spread_pips:.1f} > {self.max_spread_pips} pips)",
                details={'spread_pips': current_spread_pips, 'max': self.max_spread_pips}
            )

        return RiskCheckResult(
            passed=True,
            details={'spread_pips': current_spread_pips, 'max': self.max_spread_pips}
        )

    def can_trade(
        self,
        instrument: str,
        current_balance: float,
        peak_balance: float,
        current_spread_pips: float
    ) -> Tuple[bool, str, dict]:
        """
        Perform all risk checks before allowing a trade.

        Args:
            instrument: Currency pair
            current_balance: Current account balance
            peak_balance: Peak account balance
            current_spread_pips: Current spread in pips

        Returns:
            Tuple of (can_trade, reason, details)
        """
        all_checks = {}

        # Circuit breaker check first
        if self.circuit_breaker_tripped:
            return False, "Circuit breaker is active - trading halted", {'circuit_breaker': True}

        # Run all checks
        checks = [
            ('daily_trades', self.check_daily_trades()),
            ('daily_loss', self.check_daily_loss(current_balance)),
            ('drawdown', self.check_drawdown(current_balance, peak_balance)),
            ('positions', self.check_open_positions(instrument)),
            ('spread', self.check_spread(current_spread_pips)),
        ]

        for name, result in checks:
            all_checks[name] = result.details

            if not result.passed:
                logger.warning(f"Risk check failed: {result.reason}")
                return False, result.reason, all_checks

        return True, "All risk checks passed", all_checks

    def calculate_position_size(
        self,
        balance: float,
        risk_pct: float,
        stop_loss_pips: float,
        pip_value: float = 10.0
    ) -> int:
        """
        Calculate position size based on risk.

        Args:
            balance: Account balance
            risk_pct: Risk as percentage of balance (e.g., 1.0 = 1%)
            stop_loss_pips: Stop loss distance in pips
            pip_value: Value per pip for 1 standard lot (default 10 for most pairs)

        Returns:
            Position size in units
        """
        if stop_loss_pips <= 0:
            logger.warning("Invalid stop loss distance")
            return 0

        # Risk amount in account currency
        risk_amount = balance * (risk_pct / 100.0)

        # Position size calculation
        # risk_amount = stop_loss_pips * pip_value * lots
        # lots = risk_amount / (stop_loss_pips * pip_value)

        lots = risk_amount / (stop_loss_pips * pip_value)

        # Convert to units (1 lot = 100,000 units)
        units = int(lots * 100000)

        # Minimum 1 unit
        units = max(1, units)

        logger.debug(f"Position size: {units} units (risk: {risk_amount:.2f}, SL: {stop_loss_pips} pips)")

        return units

    def reset_circuit_breaker(self):
        """
        Manually reset the circuit breaker.

        Use with caution - should only be done after reviewing
        the situation that triggered it.
        """
        if self.circuit_breaker_tripped:
            logger.warning("Circuit breaker manually reset")
            self.circuit_breaker_tripped = False
            self.trading_paused = False

    def get_status(self) -> dict:
        """Get current risk manager status."""
        stats = self.position_manager.daily_stats

        return {
            'circuit_breaker_tripped': self.circuit_breaker_tripped,
            'trading_paused': self.trading_paused,
            'daily_trades': self.position_manager.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'open_positions': self.position_manager.open_position_count,
            'max_open_positions': self.max_open_positions,
            'daily_pnl_pct': stats.daily_return_pct if stats else 0,
            'limits': {
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_drawdown_pct': self.max_drawdown_pct,
                'pause_drawdown_pct': self.pause_drawdown_pct,
                'max_spread_pips': self.max_spread_pips,
            }
        }
