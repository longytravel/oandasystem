"""
Fast Strategy Registry.

Provides strategy lookup by name for the optimization pipeline.
"""
from typing import List

from optimization.fast_strategy import FastStrategy

# Registry of available fast strategies
FAST_STRATEGIES = {}

# Import V3 strategy (default, stability-hardened)
try:
    from strategies.rsi_full_v3 import RSIDivergenceFullFastV3
    FAST_STRATEGIES['rsi_v3'] = RSIDivergenceFullFastV3
except ImportError:
    pass

# Import V4 strategy (trade management optimization)
try:
    from strategies.rsi_full_v4 import RSIDivergenceFullFastV4
    FAST_STRATEGIES['rsi_v4'] = RSIDivergenceFullFastV4
except ImportError:
    pass

# Import V5 strategy (chandelier exit + stale exit)
try:
    from strategies.rsi_full_v5 import RSIDivergenceFullFastV5
    FAST_STRATEGIES['rsi_v5'] = RSIDivergenceFullFastV5
except ImportError:
    pass

# Import EMA Cross strategy
try:
    from strategies.ema_cross_ml import EMACrossMLStrategy
    FAST_STRATEGIES['ema_cross_ml'] = EMACrossMLStrategy
except ImportError:
    pass

# Import Fair Price MA strategy
try:
    from strategies.fair_price_ma import FairPriceMAStrategy
    FAST_STRATEGIES['fair_price_ma'] = FairPriceMAStrategy
except ImportError:
    pass


def get_strategy(name: str) -> FastStrategy:
    """Get a fast strategy by name."""
    if name not in FAST_STRATEGIES:
        available = ', '.join(FAST_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return FAST_STRATEGIES[name]()


def list_strategies() -> List[str]:
    """List all available fast strategies."""
    return list(FAST_STRATEGIES.keys())
