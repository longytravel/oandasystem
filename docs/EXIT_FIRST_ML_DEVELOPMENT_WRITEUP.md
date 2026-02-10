# Exit-First ML Program Write-Up

## Document Control

- Project: OANDA System
- Audience: Engineering, Quant, QA, and MLOps teams
- Date: 2026-02-07
- Owner: Trading Systems Team
- Status: Draft for implementation

## 1. Executive Summary

This write-up defines a detailed technical approach for shifting strategy complexity from entries to exits.

Core direction:

1. Keep entries simple and deterministic.
2. Build a constrained machine-learning exit layer.
3. Preserve hard deterministic risk controls at all times.
4. Promote ML only when out-of-sample performance and risk gates are satisfied.

The program is explicitly staged to avoid "black-box replacement risk." The deterministic exit remains a fallback path in both backtest and live operation.

## 2. Problem Statement

Recent strategy iterations indicate that adding more rule-based exit management has not consistently improved overall quality, and in some configurations the optimizer disables most advanced management features.

Observed symptoms in current workflow:

1. Exit management complexity has produced unstable outcomes across runs.
2. Some candidate configurations revert to "minimal management" to perform better.
3. Historical review identified simulator and metrics integrity issues that can distort exit research if not fixed first.

Conclusion:

- Entry edge may be adequate.
- Exit policy quality and robustness are now the primary bottleneck.

## 3. Objectives And Non-Objectives

### 3.1 Objectives

1. Build an exit policy that improves forward expectancy and profit factor while controlling drawdown.
2. Ensure reproducible, leakage-safe model training and validation.
3. Integrate the exit policy into existing pipeline and live stack with strict fallbacks.
4. Ship with robust observability and rollback controls.

### 3.2 Non-Objectives

1. No expansion to multi-position portfolio optimization in this phase.
2. No replacement of core entry signal engine in this phase.
3. No full autonomy for ML over hard risk rules.

## 4. Guiding Principles

1. Simulator correctness before model sophistication.
2. Out-of-sample evidence over in-sample optimization.
3. Conservative assumptions for OHLC event ordering.
4. Progressive deployment: offline replay -> shadow -> paper -> limited live.
5. Deterministic fallback must always be available.

## 5. Technical Preconditions (Must Pass Before ML Development)

These are release blockers for the ML workstream.

### 5.1 Backtest Event Ordering

File: `optimization/numba_backtest.py`

Issue:

- Intrabar management updates and exit checks can be ordered in a way that creates optimistic bias under OHLC constraints.

Required change:

1. Define explicit event order policy for each bar.
2. Apply conservative sequencing for SL/TP versus management updates.
3. Add unit/regression tests covering ambiguous high/low order cases.

Acceptance criteria:

1. Event ordering behavior documented and test-backed.
2. No optimistic look-ahead behavior in regression scenarios.

### 5.2 Partial-Close Accounting Integrity

File: `optimization/numba_backtest.py`

Issue:

- Partial closes can distort trade-level metrics if treated as separate independent trades in certain analytics paths.

Required change:

1. Use parent-trade accounting model.
2. Preserve leg-level detail, but aggregate parent-level PnL and counts for strategy metrics.

Acceptance criteria:

1. Trade count is semantically correct.
2. Metrics are invariant under equivalent execution representation.

### 5.3 Monte Carlo Return Distribution Integrity

File: `pipeline/stages/s5_montecarlo.py`

Issue:

- Shuffle-only return computations can be degenerate for total-return distribution.

Required change:

1. Use bootstrap resampling for return distribution.
2. Keep path-based drawdown distribution and report separately.

Acceptance criteria:

1. Return distribution has meaningful variation.
2. Report clearly separates order-risk and sampling-risk effects.

### 5.4 Confidence Scoring Schema Consistency

File: `pipeline/stages/s6_confidence.py`

Issue:

- Candidate field/key mismatches can silently skew component scores.

Required change:

1. Enforce candidate schema validation before scoring.
2. Fail fast on missing/incorrect keys.

Acceptance criteria:

1. Score inputs validated by tests.
2. No silent default fallbacks for required fields.

### 5.5 Exit Telemetry Completeness

Files: `pipeline/report/*`, replay/backtest outputs

Required telemetry fields:

1. `trade_id`, `entry_bar`, `exit_bar`, `bars_held`
2. `exit_reason`, `exit_action`, `model_confidence`
3. `mfe_r`, `mae_r`, `giveback_r`
4. `spread_at_entry`, `spread_at_exit`, estimated slippage

Acceptance criteria:

1. Fields available in both back and forward trade details.
2. Report surfaces exit diagnostics in dedicated section.

## 6. Target System Architecture

### 6.1 Layered Exit Stack

1. Layer 1: Hard Risk Envelope (always on)
2. Layer 2: Deterministic Exit Policy (fallback/default)
3. Layer 3: ML Exit Policy (conditional override inside constraints)

### 6.2 Control Flow Per Decision Step

1. Read current position and market state.
2. Evaluate hard risk rules first.
3. If hard rule triggers, execute deterministic action immediately.
4. Else request ML action if model is healthy and confidence gate passes.
5. Validate ML action against constraints.
6. Execute action or fallback deterministic action.
7. Log full decision trace.

### 6.3 Fail-Safe Behavior

Fallback to deterministic exits if any of:

1. Model timeout.
2. Missing features.
3. Stale model version.
4. Confidence below threshold.
5. Action violates constraints.

## 7. Exit Decision Formulation

### 7.1 Decision Frequency

Start with bar-close decisions only for consistency with current backtest framework.

Future option:

- Move to sub-bar decisions only after bar-close policy is proven stable.

### 7.2 Action Space (Discrete, Constrained)

1. `HOLD`
2. `TIGHTEN_STOP_STEP`
3. `MOVE_TO_BE_OFFSET`
4. `PARTIAL_CLOSE_X`
5. `FULL_EXIT`

Constraints:

1. Never widen stop.
2. Never increase absolute risk.
3. Respect broker minimum stop distance.
4. One action per bar maximum.

### 7.3 Reward Framing

Primary reward unit: R-multiples net of costs.

Components:

1. Incremental return from action path.
2. Penalty for drawdown contribution.
3. Penalty for excessive giveback after high MFE.
4. Penalty for overstaying with decaying expectancy.

## 8. Data Engineering Design

### 8.1 Dataset Type

Open-trade decision dataset:

- One row per `(trade_id, decision_bar)` while trade is active.

### 8.2 Feature Groups

1. Trade-state features
2. Market regime features
3. Entry-context features
4. Execution/microstructure proxies
5. Portfolio context features

### 8.3 Minimum Feature Schema (Initial)

Trade state:

1. `direction`
2. `age_bars`
3. `unrealized_r`
4. `distance_to_sl_r`
5. `distance_to_tp_r`
6. `max_unrealized_r` (running MFE)
7. `max_adverse_r` (running MAE)

Market:

1. `atr_norm`
2. `realized_vol_lookback`
3. `trend_slope_short`
4. `trend_slope_long`
5. `momentum_short`
6. `momentum_long`

Entry context:

1. `entry_signal_strength`
2. `entry_filters_passed_count`
3. `entry_regime_tag`

Execution/session:

1. `spread_norm`
2. `hour_of_day`
3. `day_of_week`
4. `session_tag`

Portfolio:

1. `equity_drawdown_pct`
2. `recent_loss_streak`
3. `risk_budget_utilization`

### 8.4 Labeling Strategy

Primary supervised targets:

1. `delta_value_hold_h`: expected incremental value of holding for horizon `h`
2. `p_adverse_before_favorable`: risk probability target

Secondary action-value labels:

1. Replay-estimated `Q(s, a)` under conservative simulator rules.

### 8.5 Leakage Controls

1. Features must be available at decision timestamp.
2. No future high/low derived features without lag.
3. Strict time-based splits only.
4. Automated leakage tests in CI.

## 9. Modeling Approach

### 9.1 Model Track Strategy

Use challenger architecture:

1. Baseline deterministic exit policy.
2. Supervised ML baseline challenger.
3. Sequence model challenger.
4. Offline RL challenger (only after stable baselines).

### 9.2 Stage A: Supervised Baselines

Preferred models:

1. Gradient-boosted trees (CatBoost/LightGBM style)
2. Calibrated probability heads

Outputs:

1. Expected hold value.
2. Adverse-risk probability.
3. Confidence/uncertainty estimate.

Policy mapping example:

1. Exit if hold value < threshold and adverse risk > threshold.
2. Tighten stop if hold value slightly positive but risk rising.
3. Hold if hold value positive and risk acceptable.

### 9.3 Stage B: Sequence Challenger

Candidate classes:

1. Temporal tabular models with lag stacks.
2. Lightweight sequence encoder over recent bar window.

Promotion rule:

- Must beat Stage A across multiple OOS windows, not single-window uplift.

### 9.4 Stage C: Offline RL Challenger

Use only conservative offline methods.

Requirements before starting:

1. High-quality behavior dataset.
2. Stable OPE pipeline.
3. Reliable action constraints and fallback in runtime.

Promotion rule:

- No production promotion without paper-trade superiority and risk compliance.

## 10. Evaluation And Validation Framework

### 10.1 Core Validation Structure

1. Anchored/rolling walk-forward evaluation.
2. Purged boundaries to reduce temporal contamination.
3. Separate back and forward metrics for each candidate.

### 10.2 Primary Metrics

1. Forward expectancy/trade
2. Forward profit factor
3. Forward max drawdown
4. Recovery factor
5. MFE capture and giveback stats

### 10.3 Statistical Confidence

1. Bootstrap confidence intervals on metric deltas.
2. Regime-stratified performance checks.
3. Action-level attribution (which actions drive uplift).

### 10.4 Acceptance Gates

A model is promotable only if all pass:

1. Forward expectancy >= baseline + target uplift.
2. Forward PF >= baseline + target uplift.
3. Drawdown within allowed degradation band.
4. No major instability across OOS windows.
5. Replay determinism and audit checks pass.

## 11. Implementation Plan (12 Weeks)

Start date: 2026-02-09

### Sprint 1 (Weeks 1-2): Integrity Foundation

Scope:

1. Fix sequencing and accounting correctness.
2. Fix Monte Carlo return distribution design.
3. Enforce confidence schema checks.
4. Add exit telemetry primitives.

Deliverables:

1. Passing regression suite for simulator correctness.
2. Updated report fields and sample artifact.

Exit criteria:

1. All preconditions in Section 5 pass.

### Sprint 2 (Weeks 3-4): Exit Dataset Pipeline

Scope:

1. Build reproducible dataset extraction job.
2. Add schema versioning and data hash tracking.
3. Add leakage tests.

Deliverables:

1. Versioned dataset + data dictionary.
2. Rebuild reproducibility report.

Exit criteria:

1. Deterministic dataset rebuild from same inputs.

### Sprint 3 (Weeks 5-6): Supervised Baseline Models

Scope:

1. Train first supervised exit challengers.
2. Add calibration and threshold policy translator.
3. Run walk-forward replay.

Deliverables:

1. Baseline model card.
2. Metric comparison report vs deterministic baseline.

Exit criteria:

1. Demonstrated OOS uplift in at least one forward segment without risk breach.

### Sprint 4 (Weeks 7-8): Pipeline Integration

Scope:

1. Integrate inference into evaluation pipeline.
2. Add uncertainty gating and fallback mechanics.
3. Add action diagnostics in report tabs.

Deliverables:

1. End-to-end pipeline run with ML-exit candidate.
2. Debug and attribution dashboards.

Exit criteria:

1. Full run succeeds with deterministic fallback tests.

### Sprint 5 (Weeks 9-10): Advanced Challenger Models

Scope:

1. Add sequence challenger.
2. Optional offline RL sandbox (not production path yet).
3. Run stress scenarios.

Deliverables:

1. Challenger-vs-baseline benchmark pack.

Exit criteria:

1. Any challenger promotion candidate must pass all acceptance gates.

### Sprint 6 (Weeks 11-12): Shadow And Paper Trading

Scope:

1. Shadow inference in live loop.
2. Paper-trading activation with guardrails.
3. Rollback drills and monitoring hardening.

Deliverables:

1. Paper-trade validation report.
2. Runbook and rollback SOP.

Exit criteria:

1. Stable operations + risk-compliant paper results.

## 12. Repo-Level Integration Map

### 12.1 Files Likely To Change

1. `optimization/numba_backtest.py`
2. `pipeline/stages/s3_walkforward.py`
3. `pipeline/stages/s5_montecarlo.py`
4. `pipeline/stages/s6_confidence.py`
5. `pipeline/report/data_collector.py`
6. `pipeline/report/html_builder.py`
7. `live/position_manager.py`
8. `live/risk_manager.py`

### 12.2 New Suggested Modules

1. `pipeline/ml_exit/dataset_builder.py`
2. `pipeline/ml_exit/features.py`
3. `pipeline/ml_exit/labeling.py`
4. `pipeline/ml_exit/train.py`
5. `pipeline/ml_exit/inference.py`
6. `pipeline/ml_exit/policy.py`
7. `pipeline/ml_exit/model_registry.py`

## 13. Testing Strategy

### 13.1 Unit Tests

1. Event ordering semantics.
2. Partial-close accounting invariants.
3. Feature generation deterministic behavior.
4. Policy action validity checks.

### 13.2 Integration Tests

1. End-to-end pipeline with deterministic exits.
2. End-to-end pipeline with ML exits enabled.
3. Fallback path on model failure.

### 13.3 Regression Tests

1. Fixed-seed replay equality.
2. Metric drift alarms against baseline snapshots.

### 13.4 Live-Safety Tests

1. Inference timeout handling.
2. Missing feature handling.
3. Circuit-breaker trigger behavior.

## 14. MLOps And Governance

### 14.1 Model Registry Requirements

Each model artifact must include:

1. Model version and git commit.
2. Training window date range.
3. Dataset hash and schema version.
4. Feature list hash.
5. OOS metric summary and risk diagnostics.

### 14.2 Runtime SLOs

1. Inference latency budget per decision cycle.
2. Timeout threshold and fallback activation.
3. Uptime target for inference service/process.

### 14.3 Monitoring

1. Action distribution drift.
2. Feature drift.
3. Confidence drift.
4. Live slippage/cost drift vs backtest assumptions.

### 14.4 Rollback Policy

Immediate rollback conditions:

1. Breach of drawdown emergency threshold.
2. Repeated inference failures above threshold.
3. Action anomaly detection trigger.

Rollback action:

1. Disable ML policy flag.
2. Revert to deterministic exit policy without restart if possible.

## 15. Risk Register

1. Risk: model learns simulator artifacts.
- Mitigation: precondition gate + conservative replay + stress tests.

2. Risk: too few trades for complex models.
- Mitigation: start with simpler supervised models and strong regularization.

3. Risk: operational instability in live environment.
- Mitigation: timeout/fallback/circuit-breakers + shadow phase.

4. Risk: policy opacity reduces trust.
- Mitigation: per-action attribution and decision logging in reports.

5. Risk: complexity creep delays delivery.
- Mitigation: strict phased scope and gate-based promotion.

## 16. Team Ownership Matrix

1. Quant Lead
- Reward design, metric gates, promotion decisions.

2. Backtest Engineer
- Simulator correctness, replay engine, telemetry fields.

3. Data/ML Engineer
- Dataset pipeline, features, model training, calibration.

4. Platform/MLOps Engineer
- Inference integration, model registry, monitoring, rollback.

5. QA Engineer
- Test plans, leakage validation, regression guardrails.

## 17. Definition Of Done (Program Level)

Program is done when all are true:

1. ML exit policy outperforms deterministic baseline on forward KPIs.
2. Risk constraints and drawdown criteria are met.
3. Deterministic fallback validated under forced-failure tests.
4. Reproducibility and governance artifacts are complete.
5. Paper-trade phase passes for agreed observation period.

## 18. Immediate Next Actions (Week 1 Checklist)

1. Convert Section 5 preconditions into Jira blockers.
2. Implement simulator and telemetry fixes first.
3. Freeze baseline configs for A/B reference.
4. Approve initial feature schema and dataset contract.
5. Schedule sprint review with quant and trading stakeholders.

