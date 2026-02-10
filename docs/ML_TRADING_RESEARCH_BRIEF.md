# Research Brief: ML-Based Forex Trading System

## Objective

Design a profitable algorithmic forex trading system that uses machine learning as its **primary decision engine**, not as an add-on filter to a rule-based strategy.

## Context: What We've Tried (and Why It Failed)

We built a 7-stage validation pipeline (optimization, walk-forward, stability, Monte Carlo, confidence scoring) and tested ML in two roles:

1. **ML exit timing** — CatBoost/sklearn trained per walk-forward window to predict when to close trades. Result: neutral across 5 A/B tests. Too few trades (~70-100/window on H1), no learnable exit structure.

2. **ML entry filter** — CatBoost classifier filtering EMA crossover signals. Result: detected in-sample skill (AUC 0.61-0.76) but zero skill on out-of-sample window. ML couldn't improve a weak underlying signal.

**Core problem**: We used ML as a **secondary filter** on a rule-based strategy. The rule-based strategy already captured most of the edge, leaving ML with noise.

## What We Need Researched

### 1. Model Architecture

Which ML model is best suited for forex price prediction at the 15-minute to 4-hour horizon?

Candidates to evaluate:
- **Gradient boosting** (XGBoost/LightGBM/CatBoost) — tabular features, fast training
- **LSTM/GRU** — sequential price patterns, regime memory
- **Temporal Fusion Transformer (TFT)** — multi-horizon forecasting with attention
- **Temporal Convolutional Networks (TCN)** — dilated convolutions over price series
- **Ensemble** — combine multiple model types

Key question: **What has published evidence of out-of-sample profitability on forex?**

### 2. Target Variable

What should the model predict?

Options:
- **Next-N-bar direction** (binary: up/down) — simplest, but ignores magnitude
- **Next-N-bar return** (regression) — harder, but enables position sizing
- **Regime classification** (trending/ranging/volatile) — trade differently per regime
- **Optimal holding period** — predict how long to hold for max risk-adjusted return
- **Probability of hitting X pips before Y pips** — directly models risk/reward

### 3. Feature Engineering

What input features have proven predictive for forex?

Categories to research:
- **Price-derived**: returns, volatility, momentum, mean-reversion indicators
- **Microstructure**: spread, volume patterns, order flow proxies
- **Cross-pair**: correlations, currency strength indices, carry differentials
- **Macro**: interest rate differentials, economic calendar proximity
- **Technical**: support/resistance levels, chart pattern embeddings
- **Time**: session (London/NY/Tokyo), day of week, month seasonality

Key question: **What is the minimum viable feature set that captures real signal?** (More features != better — overfitting is the #1 risk)

### 4. Timeframe & Pair Selection

- **Timeframe**: M15 or H1? M15 gives more data but more noise. H1 is cleaner but fewer samples. What does the research say about optimal prediction horizons for ML?
- **Pairs**: Major pairs (EUR/USD, GBP/USD) have tightest spreads but most efficient. Are exotic or cross pairs (GBP/JPY, AUD/NZD) less efficient and more exploitable by ML?
- **Multi-pair**: Should one model trade multiple pairs, or should each pair have a dedicated model?

### 5. Training Methodology

Critical questions:
- **Walk-forward vs expanding window** — which prevents overfitting better?
- **Purged cross-validation** — how to properly handle time-series data leakage?
- **Sample size** — minimum trades/observations needed for reliable ML training?
- **Label construction** — fixed horizon vs triple barrier method vs event-driven?
- **Feature selection** — how to identify genuinely predictive features vs noise?

### 6. Execution & Risk Framework

- **Position sizing**: Kelly criterion, fixed fractional, or ML-predicted confidence-based?
- **Transaction costs**: How to incorporate spread + slippage into the training objective?
- **Drawdown control**: Maximum drawdown limits, regime-based position scaling?

## Deliverable Expected

A recommendation document covering:

1. **Recommended model** with justification (published results, sample sizes needed)
2. **Feature set** — specific features ranked by expected predictive power
3. **Training pipeline** — exact methodology to avoid overfitting (validation scheme, sample sizes)
4. **Pair + timeframe** — which pair(s) and timeframe(s) to target
5. **Realistic expectations** — what Sharpe ratio / win rate / drawdown is achievable?
6. **Key references** — 3-5 papers or proven open-source implementations to build on

## Constraints

- We trade forex via OANDA (spot FX, no futures/options)
- We have a working Python pipeline with numba backtesting, Optuna optimization, and walk-forward validation
- We need **out-of-sample proof** — in-sample performance means nothing
- Minimum 200+ trades in the test period for statistical significance
- Target: Sharpe > 1.5, max drawdown < 20%, profitable after spreads

## Key Reference

Marcos Lopez de Prado, *Advances in Financial Machine Learning* (2018) — considered the gold standard for ML in finance. Chapters on triple barrier labeling, purged cross-validation, and feature importance are directly relevant.
