# Sprint 5 Handover: RSI V3 ML Exit Execution Plan

Last updated: 2026-02-08  
Audience: Strategy research, ML engineering, platform engineering, QA  
Sprint length: 10 working days

## Executive Summary

Yes, this can still work for the RSI system, but only if Sprint 5 is run as a strict research sprint with hard pass/fail gates.

What is true today:
- The ML exit integration is technically stable and mostly correct.
- On RSI V3 (GBP_USD H1), ML is currently neutral: same top score as no-ML, with materially higher runtime.
- The biggest blocker is target quality: `hold_value` regression has weak predictive power and is currently choking a dual-model AND policy.
- Current A/B evidence is not fully conclusive because recent runs showed `oos_n_windows = 0` (no true walk-forward OOS windows).

Sprint 5 goal:
- Prove measurable out-of-sample value from ML exits on RSI V3, or explicitly terminate ML-on-RSI and keep deterministic exits.

## Current Baseline (Reference Snapshot)

Use these run directories as Sprint 5 baseline artifacts:
- No ML: `results/pipelines/GBP_USD_H1_20260208_122103`
- ML (CatBoost): `results/pipelines/GBP_USD_H1_20260208_122113`

Observed:
- Score: 86.3 GREEN in both runs
- Candidates: 5/5 GREEN in both runs
- Runtime: ~26 min (no-ML) vs ~78.5 min (ML)
- Best candidate/trial unchanged
- ML diagnostics: weak hold-value `val_r2`, strong risk AUC, low to moderate exit firing

Methodology caveat:
- `oos_n_windows = 0` in best-candidate WF stats. This must be fixed before accepting any "ML works/does not work" claim.

## Sprint 5 Success Criteria (All Required)

1. Evaluation integrity:
- Minimum 2 true OOS windows for every promoted result.
- Paired A/B protocol (same data, same candidate set, same seeds/config except ML layer).

2. Trading uplift:
- OOS net return improvement >= +5% relative to deterministic baseline, or
- OOS max drawdown reduction >= 10% with no drop in OOS return.

3. Statistical confidence:
- Paired bootstrap delta (ML - noML) for OOS return has 95% CI lower bound > 0, and
- Existing permutation significance remains valid (`p < 0.05`) for promoted candidate.

4. Practicality:
- Runtime multiplier <= 2.0x deterministic baseline for standard test profile (`--top-n 5`).

5. Safety:
- No widening of risk envelope and no regression in hard risk constraints.

If any criterion fails, Sprint 5 outcome is "ML exit not production-ready for RSI V3".

## Scope and Non-Goals

In scope:
- Exit model/policy redesign only (entries remain RSI V3 deterministic).
- Label redesign, feature expansion, regime-aware logic, evaluation hardening.
- Reproducible experiment harness and developer-facing reporting.

Out of scope:
- New entry strategy invention.
- Live trading rollout (this remains Sprint 6).
- Deep RL / sequence models in this sprint.

## Team Structure and Ownership

Research Lead (Quant):
- Experiment design, pass/fail decisions, metric governance.

ML Engineer:
- Labels, models, calibration, feature work, regime models.

Platform Engineer:
- Pipeline integration, runtime optimization, report plumbing, CLI tooling.

QA/Validation Engineer:
- Reproducibility checks, A/B parity checks, regression tests, audit trails.

## Workstream Plan

### WS1: Evaluation Integrity and A/B Harness (Days 1-2)

Objective:
- Remove ambiguity in results and enforce OOS-first validation.

Implementation tasks:
- Add walk-forward guard: fail promotion when OOS windows < configured minimum.
- Add explicit paired A/B runner for ML vs no-ML with frozen candidate set.
- Persist per-window OOS delta metrics in artifacts.

Target files:
- `pipeline/config.py`
- `pipeline/stages/s3_walkforward.py`
- `pipeline/report/data_collector.py`
- `pipeline/report/chart_generators.py`
- `pipeline/report/html_builder.py`
- `scripts/` (new A/B orchestration script)

Definition of done:
- Any run with `oos_n_windows < 2` is flagged invalid for model claims.
- A/B report shows per-window OOS deltas and aggregate uplift diagnostics.

### WS2: Policy Simplification - Risk-Only Baseline (Days 2-4)

Objective:
- Remove the low-signal hold-value bottleneck and test whether risk model alone can add value.

Implementation tasks:
- Add policy mode switch: `dual_model` vs `risk_only`.
- In `risk_only`, exit condition uses calibrated risk probability and confidence gate only.
- Keep deterministic fallback if confidence below threshold.

Target files:
- `pipeline/config.py`
- `pipeline/ml_exit/policy.py`
- `pipeline/stages/s3_walkforward.py`

Key sweeps:
- Risk threshold grid: 0.45, 0.50, 0.55, 0.60, 0.65
- Confidence threshold grid: 0.25, 0.30, 0.35, 0.40

Definition of done:
- Risk-only policy can be turned on/off by config and appears in report metadata.
- At least one threshold pair beats deterministic baseline on OOS return or OOS DD tradeoff.

### WS3: Label Redesign (Days 4-6)

Objective:
- Replace hard regression target with more learnable binary outcomes.

Implementation tasks:
- Add binary labels that map to exit utility:
  - `will_finish_loser` (trade final pnl < 0)
  - `will_hit_sl_before_positive_1r` (or equivalent adverse outcome)
  - Optional horizon label: `negative_next_n_bars` for n in {3, 5, 8}
- Support class weights and minimum positive-rate safeguards.

Target files:
- `pipeline/ml_exit/labeling.py`
- `pipeline/ml_exit/dataset_builder.py`
- `pipeline/ml_exit/train.py`
- `pipeline/ml_exit/inference.py`

Definition of done:
- New label modes selectable via config.
- Training metrics include class balance and calibration diagnostics.

### WS4: Feature Expansion Focused on Exit Timing (Days 5-7)

Objective:
- Add features likely to matter for exit timing, not entry discovery.

Implementation tasks:
- Add volatility regime features:
  - ATR percentile regime
  - rolling realized volatility
  - volatility expansion/compression flags
- Add structure and context features:
  - distance to recent swing high/low in R units
  - candle body/wick imbalance
  - spread-normalized move magnitude
- Add multi-timeframe context from H1-derived higher frame snapshots (H4-like aggregates).

Target files:
- `pipeline/ml_exit/features.py`
- `pipeline/ml_exit/dataset_builder.py`
- `pipeline/ml_exit/train.py`

Definition of done:
- Feature set versioned and logged in run metadata.
- Feature importance and ablation table generated in report.

### WS5: Regime-Aware Model Routing (Days 7-8)

Objective:
- Avoid one-model-fits-all behavior across trending and ranging conditions.

Implementation tasks:
- Define deterministic regime classifier (trend/range/high-vol squeeze).
- Train separate risk models per regime (or threshold map per regime).
- Route inference by regime with fallback to global model when sample size is low.

Target files:
- `pipeline/ml_exit/features.py`
- `pipeline/ml_exit/train.py`
- `pipeline/ml_exit/inference.py`
- `pipeline/ml_exit/policy.py`

Definition of done:
- Regime routing logged per decision row.
- No window fails due to missing regime model (fallback path verified).

### WS6: Runtime and Reliability Hardening (Days 8-9)

Objective:
- Reduce ML overhead and eliminate low-sample instability.

Implementation tasks:
- Resolve split edge cases around low sample counts (50-100 rows).
- Add adaptive Optuna budget by sample size.
- Cache model training inputs per window/candidate hash where valid.

Target files:
- `pipeline/ml_exit/train.py`
- `pipeline/stages/s3_walkforward.py`

Definition of done:
- No low-sample crashes.
- Runtime <= 2x deterministic baseline for `--top-n 5`.

### WS7: Final Evaluation and Decision (Day 10)

Objective:
- Produce clear go/no-go outcome for RSI ML exits.

Implementation tasks:
- Run final paired A/B on approved config set.
- Produce promotion memo with hard criteria outcome table.
- Tag winning config or close Sprint 5 as no-go for RSI.

Deliverables:
- Final result dirs (A/B)
- Summary markdown in `docs/ML_EXIT_PROGRAM.md`
- Decision section appended to this handover

## Experiment Matrix

E0. Baseline Reproduction
- Reproduce deterministic and current ML baseline with pinned settings.
- Output: baseline parity report.

E1. Risk-Only Policy Sweep
- Compare threshold grids under same candidate set.
- Promotion from E1 requires OOS uplift and safety compliance.

E2. Risk-Only + Calibration
- Add probability calibration (if required by reliability plots).
- Compare calibration on/off.

E3. Label Redesign
- Compare original labels vs binary utility labels.
- Keep only labels with stable OOS gains across windows.

E4. Feature Expansion and Ablation
- Add feature groups incrementally, run ablations.
- Remove any group that does not improve OOS metrics.

E5. Regime Routing
- Global model vs regime-aware model.
- Keep regime routing only if uplift survives OOS paired tests.

E6. Final Locked A/B
- Freeze winning config and run final confirmatory A/B.
- This run decides Sprint 5 outcome.

## Standard Run Protocol

Use explicit settings in all Sprint 5 comparisons:
- `--pair GBP_USD --timeframe H1 --strategy rsi_v3`
- `--test-months 6` (or chosen value, fixed across all experiments)
- fixed `--top-n` and trial budgets for fair comparisons
- fixed random seeds where supported

Run profile tiers:
- Dev profile: fast smoke checks (`--fast --top-n 3`)
- Research profile: intermediate (`--top-n 5`)
- Promotion profile: full (`--top-n 20` as final confirmation)

Recommendation:
- Increase `--years` enough to guarantee at least 2 OOS WF windows under chosen train/test settings.

## Reporting Requirements

Each experiment report must include:
- OOS window count and OOS pass rate
- OOS delta table (ML minus no-ML) for return, DD, Sharpe, PF
- Exit firing rate and confidence distribution
- Calibration/reliability summary for risk probabilities
- Runtime and failure logs

## Risks and Mitigations

Risk: ML still neutral on RSI despite redesign.
- Mitigation: hard stop criteria and deterministic fallback.

Risk: Small sample windows cause noisy conclusions.
- Mitigation: OOS minimum window guard, paired bootstrap CIs, no promotion without CI support.

Risk: Runtime explosion from per-window tuning.
- Mitigation: adaptive Optuna budgets, caching, strict promotion runtime cap.

Risk: Hidden overfitting via repeated threshold tuning.
- Mitigation: frozen validation protocol and final locked confirmatory run.

## Sprint 5 Go/No-Go Decision Logic

GO:
- All success criteria satisfied, including OOS uplift and runtime cap.

NO-GO (for RSI V3):
- Criteria not met after E6 final locked A/B.
- Keep deterministic RSI V3 exits for production and stop ML-on-RSI work.

Conditional GO:
- Only risk reduction improves materially while return stays flat.
- Acceptable only if risk objective was pre-declared as primary.

## Immediate Next Actions (First 48 Hours)

1. Implement WS1 guardrails and paired A/B harness.
2. Run E0 baseline reproduction with OOS validity check.
3. Run E1 risk-only threshold sweep and publish first OOS delta table.

