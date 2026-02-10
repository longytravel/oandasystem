"""Optimization module.

Lazy imports to avoid requiring numba/llvmlite on machines
that only run live trading (not optimization).
"""


def __getattr__(name):
    if name == "UltraFastOptimizer" or name == "Metrics":
        from optimization.ultra_fast import UltraFastOptimizer, Metrics
        return UltraFastOptimizer if name == "UltraFastOptimizer" else Metrics
    if name == "FastStrategy":
        from optimization.fast_strategy import FastStrategy
        return FastStrategy
    if name == "FastSignal":
        from optimization.fast_strategy import FastSignal
        return FastSignal
    raise AttributeError(f"module 'optimization' has no attribute {name}")


__all__ = ["UltraFastOptimizer", "Metrics", "FastStrategy", "FastSignal"]
