# RSI Divergence Pro V2 - Implementation Summary

## Overview

V2 addresses three root causes of the EUR_USD forward failure:
1. **Market Regime Change** - Strategy trades blindly regardless of conditions
2. **Staged Optimization Lock-In** - Early stages lock bad decisions
3. **No "Don't Trade" Signal** - Always trades if divergence appears

## Files Modified/Created

### New Files
- `strategies/rsi_divergence_v2.py` - V2 strategy with regime detection and quality scoring
- `scripts/compare_v1_v2.py` - Side-by-side comparison script

### Modified Files
- `optimization/unified_optimizer.py` - Added forward threshold and weighted ranking
- `optimization/numba_backtest.py` - Added quality-based position sizing support
- `strategies/rsi_fast.py` - Added V2 to strategy registry

## V2 New Features

### 1. Market Regime Detection (6 parameters)

#### ADX Trend Strength Filter
- `use_adx_filter`: Enable/disable ADX filtering
- `min_adx`: Minimum ADX value (15-30, default 20)

Only trades when market is trending (ADX > threshold). Avoids choppy ranging markets.

#### Volatility Regime Filter
- `use_volatility_filter`: Enable/disable
- `volatility_regime`: 'low', 'normal', 'high', 'any'

Classifies ATR percentile vs recent history:
- Low: ATR < 30th percentile
- Normal: ATR 30-70th percentile
- High: ATR > 70th percentile

#### Range Detection (Bollinger Band Squeeze)
- `avoid_ranging_market`: Enable/disable
- `range_bb_percentile`: Threshold (10-30, default 20)

Detects consolidation via BB width percentile. Avoids signals during squeeze.

### 2. Signal Quality Scoring (6 parameters)

#### Divergence Strength
- `min_divergence_strength`: 'weak', 'moderate', 'strong'
- `min_divergence_score`: 0.3-0.7

Scores based on:
- Price divergence magnitude (normalized by ATR) - 40%
- RSI divergence magnitude - 40%
- Time span (sweet spot: 15-50 bars) - 20%

#### Swing Quality
- `min_swing_quality`: 0.3-0.7

Measures swing depth and symmetry (left vs right sides).

#### Confluence Scoring
- `use_confluence_filter`: Enable/disable
- `min_confluence_score`: 40-70

Multi-factor confirmation (0-100 points):
- Divergence strength: 30 pts
- Trend alignment (price vs MA): 25 pts
- Session quality (London/NY): 20 pts
- Volatility regime (moderate best): 15 pts
- ADX trend strength: 10 pts

### 3. Optimization System Improvements

#### Forward Performance Threshold
```python
UnifiedOptimizer(
    min_forward_ratio=0.05,  # Reject if forward < 5% of back
)
```

Filters out overfit results where forward performance is dramatically worse than back.

#### Weighted Combined Ranking
```python
UnifiedOptimizer(
    forward_rank_weight=2.0,  # Forward matters more
)
```

Formula: `Combined_Rank = Back_Rank + Forward_Rank * weight`

With weight=2.0, forward stability is prioritized over backtest performance.

#### Quality-Based Position Sizing
Signals now carry a `quality_mult` field (0.5-1.5x) based on signal quality.
Higher quality signals get larger positions.

## Parameter Summary

### V1 Parameters: 36 total
- Signal: 8
- Filters: 7
- Risk: 7
- Management: 8
- Time: 6

### V2 Parameters: 48 total (+12 new)
- Signal: 8
- Filters: 7
- **Regime: 6 (NEW)**
- **Quality: 6 (NEW)**
- Risk: 7
- Management: 8
- Time: 6

## Usage

### Run Comparison
```bash
python scripts/compare_v1_v2.py --pair EUR_USD --timeframe H1 --trials 3000
```

### Use V2 Strategy
```python
from strategies.rsi_divergence_v2 import RSIDivergenceV2
from optimization.unified_optimizer import UnifiedOptimizer

strategy = RSIDivergenceV2()
strategy.set_pip_size('EUR_USD')

optimizer = UnifiedOptimizer(
    strategy,
    min_forward_ratio=0.05,
    forward_rank_weight=2.0,
)

results = optimizer.run(df_back, df_forward, mode='staged')
```

## Expected Improvements

1. **EUR_USD**: Should either:
   - Show positive forward Sharpe (regime filters work)
   - Generate fewer/no signals in bad regime (correct "don't trade" behavior)

2. **GBP_USD**: Should maintain good performance with potentially:
   - Higher forward/back ratio
   - Lower drawdown (quality filters)

3. **All pairs**: More robust forward performance due to:
   - Forward threshold rejecting overfit results
   - Weighted ranking prioritizing stability

## Success Criteria

| Metric | V1 EUR_USD | V2 Target |
|--------|------------|-----------|
| Forward Sharpe | -2.06 | > 0 or no trades |
| Forward/Back Ratio | 0% | > 5% |
| Forward R² | 0.101 | > 0.5 |

## Key Insight

V2 isn't about forcing EUR_USD to be profitable - it's about:
1. **Detecting** when conditions are unfavorable
2. **Not trading** during those conditions
3. **Preventing selection** of overfit results
