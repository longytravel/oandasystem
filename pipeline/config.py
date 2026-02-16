"""
Pipeline Configuration - Centralized settings for all stages.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Stage 1: Data configuration."""
    years: float = 4.0
    force_download: bool = False
    min_candles: int = 5000  # Minimum candles required
    max_gap_hours: int = 24  # Maximum allowed gap in data
    validate_quality: bool = True
    holdout_months: int = 0  # 0 = default 80/20; >0 = reserve last N months for OOS


@dataclass
class OptimizationConfig:
    """Stage 2: Optimization configuration."""
    mode: str = 'staged'  # 'quick', 'full', 'staged', 'fullspace'
    trials_per_stage: int = 5000
    final_trials: int = 10000
    min_trades: int = 20
    min_back_sharpe: float = 1.0
    top_n_candidates: int = 50
    min_forward_ratio: float = 0.40  # FIX V4: 0.15 too permissive, allows heavily overfitted candidates
    forward_rank_weight: float = 2.0
    n_jobs: int = 1  # Parallel workers for Optuna (-1 = all cores)
    # Hard pre-filters: reject garbage before scoring to focus optimizer
    max_dd_hard_limit: float = 30.0   # MaxDD > 30% → instant reject
    min_r2_hard: float = 0.5          # R² < 0.5 → instant reject (noisy equity curve)
    # Diversity settings - ensure candidates have variety in key params
    # Default 1 = maximum diversity (1 candidate per unique signature first)
    # Higher values allow more variations of same strategy after diversity pass
    max_same_signature: int = 1


@dataclass
class WalkForwardConfig:
    """Stage 3: Walk-forward configuration."""
    train_months: int = 6
    test_months: int = 6
    roll_step_months: int = 3
    min_windows: int = 1
    min_window_pass_rate: float = 0.75  # 75% of windows must pass
    min_mean_sharpe: float = 0.5
    min_trades_per_window: int = 5  # Minimum trades per window (low-freq strategies need lower threshold)
    reoptimize_per_window: bool = False  # If True, re-optimize for each window


@dataclass
class StabilityConfig:
    """Stage 4: Parameter stability configuration."""
    min_stability: float = 0.6  # Mean stability ratio threshold
    min_single_stability: float = 0.3  # Worst single param threshold
    test_forward: bool = True


@dataclass
class MonteCarloConfig:
    """Stage 5: Monte Carlo configuration."""
    iterations: int = 1000  # FIX V4: 500 was below minimum for reliable 5th percentile
    confidence_level: float = 0.95  # 95% confidence
    min_trades_for_mc: int = 30
    bootstrap_iterations: int = 1000  # V4: Bootstrap resampling iterations for CI estimation


@dataclass
class ConfidenceConfig:
    """Stage 6: Confidence scoring weights.

    Quality Score = Sortino × R² × min(PF,5) × √min(Trades,200) × (1 + min(Return%,200)/100) / (Ulcer + MaxDD%/2 + 5)
    Hard pre-filters: MaxDD > 30% or R² < 0.5 → instant reject before scoring.
    Used as the universal metric throughout the pipeline.
    """
    backtest_quality_weight: float = 0.15
    forward_back_weight: float = 0.15
    walkforward_weight: float = 0.25
    stability_weight: float = 0.15
    montecarlo_weight: float = 0.15
    quality_score_weight: float = 0.15

    # Hard caps for Ulcer Index (chronic underwater equity)
    max_ulcer_yellow_cap: float = 10.0  # Ulcer above this → cap at YELLOW (70)
    max_ulcer_red_cap: float = 20.0     # Ulcer above this → cap at RED (40)

    # Thresholds
    red_threshold: float = 40.0
    yellow_threshold: float = 70.0
    green_threshold: float = 70.0


@dataclass
class MLExitConfig:
    """ML Exit model configuration."""
    enabled: bool = False
    # Training
    n_optuna_trials: int = 30
    cv_folds: int = 5
    early_stopping_rounds: int = 20
    # Inference thresholds
    min_hold_value: float = 0.0    # Below this = exit signal (hold_value < 0 means negative expected R)
    max_adverse_risk: float = 0.5  # Above this = exit signal (>50% SL risk)
    min_confidence: float = 0.3    # Below this = fall back to deterministic
    # Integration with numba engine
    ml_min_hold_bars: int = 3      # Min bars before ML can trigger exit
    ml_exit_threshold: float = 0.5 # Score threshold for exit (binary policy outputs 1.0/0.0)
    retrain_per_window: bool = True # Retrain model for each WF window
    policy_mode: str = 'dual_model'  # 'dual_model', 'risk_only', 'hold_only'
    ml_exit_cooldown_bars: int = 10  # Bars to skip new entries after ML exit (prevents trade inflation)
    ml_mode: str = 'exit'  # 'exit' = per-bar exit timing, 'entry_filter' = meta-labeling signal filter
    signal_filter_threshold: float = 0.5  # Probability threshold for keeping signals in entry_filter mode


@dataclass
class ReportConfig:
    """Stage 7: Report configuration."""
    include_leaderboard: bool = True
    include_equity_charts: bool = True
    include_walkforward_heatmap: bool = True
    include_montecarlo_charts: bool = True
    include_stability_charts: bool = True
    include_trade_analysis: bool = True
    leaderboard_top_n: int = 20


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # Basic settings
    pair: str = "GBP_USD"
    timeframe: str = "H1"
    strategy_name: str = "RSI_Divergence_v3"
    description: str = ''

    # Account settings
    initial_capital: float = 3000.0
    risk_per_trade: float = 1.0
    spread_pips: float = 1.5
    slippage_pips: float = 0.5

    # Output settings
    output_dir: Optional[Path] = None
    save_intermediate: bool = True

    # Stage configurations
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    montecarlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    ml_exit: MLExitConfig = field(default_factory=MLExitConfig)

    def __post_init__(self):
        """Set up output directory if not specified."""
        if self.output_dir is None:
            from config.settings import settings
            self.output_dir = settings.RESULTS_DIR / "pipelines"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pair': self.pair,
            'timeframe': self.timeframe,
            'strategy_name': self.strategy_name,
            'description': self.description,
            'initial_capital': self.initial_capital,
            'risk_per_trade': self.risk_per_trade,
            'spread_pips': self.spread_pips,
            'slippage_pips': self.slippage_pips,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'save_intermediate': self.save_intermediate,
            'data': {
                'years': self.data.years,
                'force_download': self.data.force_download,
                'min_candles': self.data.min_candles,
                'max_gap_hours': self.data.max_gap_hours,
                'validate_quality': self.data.validate_quality,
                'holdout_months': self.data.holdout_months,
            },
            'optimization': {
                'mode': self.optimization.mode,
                'trials_per_stage': self.optimization.trials_per_stage,
                'final_trials': self.optimization.final_trials,
                'min_trades': self.optimization.min_trades,
                'min_back_sharpe': self.optimization.min_back_sharpe,
                'top_n_candidates': self.optimization.top_n_candidates,
                'min_forward_ratio': self.optimization.min_forward_ratio,
                'forward_rank_weight': self.optimization.forward_rank_weight,
                'max_dd_hard_limit': self.optimization.max_dd_hard_limit,
                'min_r2_hard': self.optimization.min_r2_hard,
                'max_same_signature': self.optimization.max_same_signature,
            },
            'walkforward': {
                'train_months': self.walkforward.train_months,
                'test_months': self.walkforward.test_months,
                'roll_step_months': self.walkforward.roll_step_months,
                'min_windows': self.walkforward.min_windows,
                'min_window_pass_rate': self.walkforward.min_window_pass_rate,
                'min_mean_sharpe': self.walkforward.min_mean_sharpe,
                'min_trades_per_window': self.walkforward.min_trades_per_window,
                'reoptimize_per_window': self.walkforward.reoptimize_per_window,
            },
            'stability': {
                'min_stability': self.stability.min_stability,
                'min_single_stability': self.stability.min_single_stability,
                'test_forward': self.stability.test_forward,
            },
            'montecarlo': {
                'iterations': self.montecarlo.iterations,
                'confidence_level': self.montecarlo.confidence_level,
                'min_trades_for_mc': self.montecarlo.min_trades_for_mc,
                'bootstrap_iterations': self.montecarlo.bootstrap_iterations,
            },
            'confidence': {
                'backtest_quality_weight': self.confidence.backtest_quality_weight,
                'forward_back_weight': self.confidence.forward_back_weight,
                'walkforward_weight': self.confidence.walkforward_weight,
                'stability_weight': self.confidence.stability_weight,
                'montecarlo_weight': self.confidence.montecarlo_weight,
                'quality_score_weight': self.confidence.quality_score_weight,
                'max_ulcer_yellow_cap': self.confidence.max_ulcer_yellow_cap,
                'max_ulcer_red_cap': self.confidence.max_ulcer_red_cap,
                'red_threshold': self.confidence.red_threshold,
                'yellow_threshold': self.confidence.yellow_threshold,
                'green_threshold': self.confidence.green_threshold,
            },
            'report': {
                'include_leaderboard': self.report.include_leaderboard,
                'include_equity_charts': self.report.include_equity_charts,
                'include_walkforward_heatmap': self.report.include_walkforward_heatmap,
                'include_montecarlo_charts': self.report.include_montecarlo_charts,
                'include_stability_charts': self.report.include_stability_charts,
                'include_trade_analysis': self.report.include_trade_analysis,
                'leaderboard_top_n': self.report.leaderboard_top_n,
            },
            'ml_exit': {
                'enabled': self.ml_exit.enabled,
                'n_optuna_trials': self.ml_exit.n_optuna_trials,
                'cv_folds': self.ml_exit.cv_folds,
                'early_stopping_rounds': self.ml_exit.early_stopping_rounds,
                'min_hold_value': self.ml_exit.min_hold_value,
                'max_adverse_risk': self.ml_exit.max_adverse_risk,
                'min_confidence': self.ml_exit.min_confidence,
                'ml_min_hold_bars': self.ml_exit.ml_min_hold_bars,
                'ml_exit_threshold': self.ml_exit.ml_exit_threshold,
                'retrain_per_window': self.ml_exit.retrain_per_window,
                'policy_mode': self.ml_exit.policy_mode,
                'ml_exit_cooldown_bars': self.ml_exit.ml_exit_cooldown_bars,
                'ml_mode': self.ml_exit.ml_mode,
                'signal_filter_threshold': self.ml_exit.signal_filter_threshold,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        config = cls(
            pair=data.get('pair', 'GBP_USD'),
            timeframe=data.get('timeframe', 'H1'),
            strategy_name=data.get('strategy_name', 'RSI_Divergence_v3'),
            description=data.get('description', ''),
            initial_capital=data.get('initial_capital', 3000.0),
            risk_per_trade=data.get('risk_per_trade', 1.0),
            spread_pips=data.get('spread_pips', 1.5),
            slippage_pips=data.get('slippage_pips', 0.5),
            save_intermediate=data.get('save_intermediate', True),
        )

        if data.get('output_dir'):
            config.output_dir = Path(data['output_dir'])

        # Load nested configs
        if 'data' in data:
            for k, v in data['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        if 'optimization' in data:
            for k, v in data['optimization'].items():
                if hasattr(config.optimization, k):
                    setattr(config.optimization, k, v)

        if 'walkforward' in data:
            for k, v in data['walkforward'].items():
                if hasattr(config.walkforward, k):
                    setattr(config.walkforward, k, v)

        if 'stability' in data:
            for k, v in data['stability'].items():
                if hasattr(config.stability, k):
                    setattr(config.stability, k, v)

        if 'montecarlo' in data:
            for k, v in data['montecarlo'].items():
                if hasattr(config.montecarlo, k):
                    setattr(config.montecarlo, k, v)

        if 'confidence' in data:
            for k, v in data['confidence'].items():
                if hasattr(config.confidence, k):
                    setattr(config.confidence, k, v)

        if 'report' in data:
            for k, v in data['report'].items():
                if hasattr(config.report, k):
                    setattr(config.report, k, v)

        if 'ml_exit' in data:
            for k, v in data['ml_exit'].items():
                if hasattr(config.ml_exit, k):
                    setattr(config.ml_exit, k, v)

        return config
