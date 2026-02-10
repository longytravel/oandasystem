"""
Telegram alerts for live trading notifications.

Sends alerts for:
- Trade opened
- Trade closed (with P&L)
- Daily summary
- Risk warnings
"""
import requests
from datetime import datetime
from typing import Optional
from loguru import logger

from config.settings import settings


class TelegramAlerts:
    """
    Telegram notification system for trading alerts.

    Setup:
    1. Create a bot with @BotFather on Telegram
    2. Get the bot token
    3. Start a chat with your bot and get your chat ID
    4. Add to .env:
       TELEGRAM_BOT_TOKEN=your_bot_token
       TELEGRAM_CHAT_ID=your_chat_id
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
        instance_id: Optional[str] = None,
    ):
        """
        Initialize Telegram alerts.

        Args:
            bot_token: Telegram bot token (default from settings)
            chat_id: Telegram chat ID (default from settings)
            enabled: Whether to send alerts (can disable for testing)
            instance_id: Instance identifier to prefix messages (for multi-strategy)
        """
        self.bot_token = bot_token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self.instance_id = instance_id

        if self.enabled:
            logger.info("Telegram alerts enabled")
        else:
            if enabled:
                logger.warning("Telegram alerts disabled - missing token or chat_id")

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram.

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Parse mode (HTML or Markdown)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Alert (disabled): {text[:50]}...")
            return False

        # Prefix with instance ID for multi-strategy deployments
        if self.instance_id:
            text = f"[{self.instance_id}]\n{text}"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        try:
            response = requests.post(url, data={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }, timeout=10)

            if response.status_code == 200:
                logger.debug("Telegram message sent")
                return True
            else:
                logger.warning(f"Telegram send failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def trade_opened(
        self,
        instrument: str,
        direction: str,
        units: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        trade_id: str = ""
    ):
        """
        Send trade opened alert.

        Args:
            instrument: Currency pair
            direction: BUY or SELL
            units: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            trade_id: Optional trade ID
        """
        emoji = "🟢" if direction == "BUY" else "🔴"

        message = f"""
{emoji} <b>TRADE OPENED</b>

<b>Pair:</b> {instrument}
<b>Direction:</b> {direction}
<b>Size:</b> {units:,} units
<b>Entry:</b> {entry_price:.5f}
<b>Stop Loss:</b> {stop_loss:.5f}
<b>Take Profit:</b> {take_profit:.5f}
{f'<b>Trade ID:</b> {trade_id}' if trade_id else ''}
<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def trade_closed(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str,
        trade_id: str = ""
    ):
        """
        Send trade closed alert.

        Args:
            instrument: Currency pair
            direction: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            realized_pnl: Realized P&L
            exit_reason: Reason for exit (TP, SL, etc.)
            trade_id: Optional trade ID
        """
        if realized_pnl > 0:
            emoji = "✅"
            result = "WIN"
        else:
            emoji = "❌"
            result = "LOSS"

        message = f"""
{emoji} <b>TRADE CLOSED - {result}</b>

<b>Pair:</b> {instrument}
<b>Direction:</b> {direction}
<b>Entry:</b> {entry_price:.5f}
<b>Exit:</b> {exit_price:.5f}
<b>P&L:</b> {realized_pnl:+.2f}
<b>Reason:</b> {exit_reason}
{f'<b>Trade ID:</b> {trade_id}' if trade_id else ''}
<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def daily_summary(
        self,
        date: str,
        starting_balance: float,
        ending_balance: float,
        trades_count: int,
        wins: int,
        losses: int,
        gross_profit: float,
        gross_loss: float
    ):
        """
        Send daily summary alert.

        Args:
            date: Date string
            starting_balance: Starting balance
            ending_balance: Ending balance
            trades_count: Total trades
            wins: Winning trades
            losses: Losing trades
            gross_profit: Gross profit
            gross_loss: Gross loss
        """
        net_pnl = gross_profit + gross_loss
        daily_return = (ending_balance - starting_balance) / starting_balance * 100 if starting_balance > 0 else 0
        win_rate = wins / trades_count * 100 if trades_count > 0 else 0

        emoji = "📈" if net_pnl >= 0 else "📉"

        message = f"""
{emoji} <b>DAILY SUMMARY - {date}</b>

<b>Starting:</b> {starting_balance:,.2f}
<b>Ending:</b> {ending_balance:,.2f}
<b>Net P&L:</b> {net_pnl:+,.2f} ({daily_return:+.2f}%)

<b>Trades:</b> {trades_count}
<b>Wins/Losses:</b> {wins}/{losses} ({win_rate:.1f}%)
<b>Gross Profit:</b> {gross_profit:+,.2f}
<b>Gross Loss:</b> {gross_loss:+,.2f}

<i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def risk_warning(
        self,
        warning_type: str,
        details: str,
        severity: str = "WARNING"
    ):
        """
        Send risk warning alert.

        Args:
            warning_type: Type of warning (DRAWDOWN, DAILY_LOSS, etc.)
            details: Warning details
            severity: WARNING or CRITICAL
        """
        if severity == "CRITICAL":
            emoji = "🚨"
        else:
            emoji = "⚠️"

        message = f"""
{emoji} <b>RISK {severity}</b>

<b>Type:</b> {warning_type}
<b>Details:</b> {details}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def circuit_breaker_triggered(self, drawdown_pct: float):
        """Send circuit breaker alert."""
        message = f"""
🛑 <b>CIRCUIT BREAKER TRIGGERED</b>

Trading has been automatically stopped.

<b>Drawdown:</b> {drawdown_pct:.2f}%
<b>Limit:</b> {settings.MAX_DRAWDOWN_PCT}%

Manual intervention required to resume trading.

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def startup_notification(
        self,
        instrument: str,
        timeframe: str,
        strategy: str,
        dry_run: bool = False
    ):
        """Send startup notification."""
        mode = "DRY RUN" if dry_run else "LIVE"
        emoji = "🔔" if dry_run else "🚀"

        message = f"""
{emoji} <b>TRADING SYSTEM STARTED</b>

<b>Mode:</b> {mode}
<b>Pair:</b> {instrument}
<b>Timeframe:</b> {timeframe}
<b>Strategy:</b> {strategy}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def shutdown_notification(self, reason: str = "Manual shutdown"):
        """Send shutdown notification."""
        message = f"""
🔴 <b>TRADING SYSTEM STOPPED</b>

<b>Reason:</b> {reason}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(message.strip())

    def custom_alert(self, title: str, message: str, emoji: str = "📌"):
        """
        Send a custom alert.

        Args:
            title: Alert title
            message: Alert message
            emoji: Emoji to use
        """
        full_message = f"""
{emoji} <b>{title}</b>

{message}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_message(full_message.strip())

    def test_connection(self) -> bool:
        """Test the Telegram connection."""
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False

        return self._send_message("🔔 Test message from OANDA Trading System")


# Singleton instance for easy import
alerts = TelegramAlerts()


if __name__ == "__main__":
    # Test alerts
    logger.info("Testing Telegram alerts...")

    if not alerts.enabled:
        logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    else:
        if alerts.test_connection():
            logger.info("Test message sent successfully!")
        else:
            logger.error("Failed to send test message")
