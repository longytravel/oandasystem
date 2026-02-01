"""
Global configuration loaded from environment variables.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings from .env file."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # OANDA API
    OANDA_API_KEY: str = ""
    OANDA_ACCOUNT_TYPE: str = "practice"  # "practice" or "live"
    OANDA_ACCOUNT_ID: str = ""

    @property
    def OANDA_API_URL(self) -> str:
        """Get the correct API URL based on account type."""
        if self.OANDA_ACCOUNT_TYPE == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    @property
    def OANDA_STREAM_URL(self) -> str:
        """Get the correct streaming URL based on account type."""
        if self.OANDA_ACCOUNT_TYPE == "live":
            return "https://stream-fxtrade.oanda.com"
        return "https://stream-fxpractice.oanda.com"

    # Trading defaults
    DEFAULT_PAIR: str = "GBP_USD"
    DEFAULT_TIMEFRAME: str = "H1"

    # Risk management
    MAX_RISK_PER_TRADE: float = 1.0      # Percent of equity
    MAX_DAILY_LOSS_PCT: float = 3.0      # Percent of equity
    MAX_DRAWDOWN_PCT: float = 25.0       # Hard stop
    PAUSE_DRAWDOWN_PCT: float = 15.0     # Alert and pause
    MAX_DAILY_TRADES: int = 5
    MAX_OPEN_POSITIONS: int = 3

    # Spread assumptions (pips)
    DEFAULT_SPREAD_PIPS: float = 1.5
    MAX_SPREAD_PIPS: float = 3.0
    SLIPPAGE_PIPS: float = 0.5

    # Telegram (optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def print_settings():
    """Print current settings (for debugging)."""
    print("\n" + "=" * 50)
    print("FOREX PYTHON TRADING SYSTEM - SETTINGS")
    print("=" * 50)
    print(f"OANDA Account Type: {settings.OANDA_ACCOUNT_TYPE}")
    print(f"OANDA API URL: {settings.OANDA_API_URL}")
    print(f"OANDA Account ID: {settings.OANDA_ACCOUNT_ID or 'NOT SET'}")
    print(f"API Key: {'*' * 20}...{settings.OANDA_API_KEY[-8:] if settings.OANDA_API_KEY else 'NOT SET'}")
    print(f"Default Pair: {settings.DEFAULT_PAIR}")
    print(f"Default Timeframe: {settings.DEFAULT_TIMEFRAME}")
    print(f"Max Risk Per Trade: {settings.MAX_RISK_PER_TRADE}%")
    print(f"Max Drawdown: {settings.MAX_DRAWDOWN_PCT}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print_settings()
