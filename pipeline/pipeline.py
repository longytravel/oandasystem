"""
Pipeline Orchestrator - Main entry point for the 7-stage pipeline.

Coordinates execution of all stages with:
- Resumability (can restart from any stage)
- State persistence
- Progress tracking
- Error handling
"""
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from loguru import logger

from pipeline.config import PipelineConfig
from pipeline.state import PipelineState, StageStatus, STAGE_ORDER
from pipeline.stages.s1_data import DataStage
from pipeline.stages.s2_optimization import OptimizationStage
from pipeline.stages.s3_walkforward import WalkForwardStage
from pipeline.stages.s4_stability import StabilityStage
from pipeline.stages.s5_montecarlo import MonteCarloStage
from pipeline.stages.s6_confidence import ConfidenceStage
from pipeline.stages.s7_report import ReportStage


class Pipeline:
    """
    7-Stage E2E Pipeline for strategy validation.

    Stages:
    1. Data Download & Validation
    2. Initial Optimization
    3. Walk-Forward Validation
    4. Parameter Stability
    5. Monte Carlo Simulation
    6. Confidence Scoring
    7. Report Generation

    Usage:
        config = PipelineConfig(pair='GBP_USD', timeframe='H1')
        pipeline = Pipeline(config)
        result = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state: Optional[PipelineState] = None

        # Stage instances
        self.stages = {
            'data': DataStage(config),
            'optimization': OptimizationStage(config),
            'walkforward': WalkForwardStage(config),
            'stability': StabilityStage(config),
            'montecarlo': MonteCarloStage(config),
            'confidence': ConfidenceStage(config),
            'report': ReportStage(config),
        }

        # Stage results cache
        self.results: Dict[str, Any] = {}

    def run(
        self,
        resume_from: Optional[str] = None,
        stop_after: Optional[str] = None,
        run_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            resume_from: Stage name to resume from (loads previous state)
            stop_after: Stage name to stop after
            run_dir: Existing run directory (for resuming)

        Returns:
            Dict with final results and state
        """
        start_time = time.time()

        # Initialize or load state
        if run_dir and (run_dir / "state.json").exists():
            logger.info(f"Resuming from existing run: {run_dir}")
            self.state = PipelineState.load(run_dir / "state.json")
        else:
            # Auto-generate description if not provided
            description = self.config.description
            if not description:
                ml_part = f"ML {self.config.ml_exit.policy_mode}" if self.config.ml_exit.enabled else "no ML"
                holdout_part = f" hold={self.config.data.holdout_months}m" if self.config.data.holdout_months > 0 else ""
                description = (f"{self.config.strategy_name} | "
                              f"{self.config.data.years:.0f}yr "
                              f"{self.config.walkforward.train_months}m/{self.config.walkforward.test_months}m | "
                              f"{ml_part}{holdout_part}")

            self.state = PipelineState.create_new(
                pair=self.config.pair,
                timeframe=self.config.timeframe,
                strategy_name=self.config.strategy_name,
                output_dir=self.config.output_dir,
                config=self.config.to_dict(),
                description=description,
            )

        # Print banner
        self._print_banner()

        # Determine starting stage
        if resume_from:
            start_stage = resume_from
            logger.info(f"Resuming from stage: {start_stage}")
        else:
            start_stage = self.state.get_next_stage() or STAGE_ORDER[0]

        # When resuming from a later stage, always reload data first
        # (it's fast - just loads from cache)
        if resume_from and resume_from != 'data':
            logger.info("Reloading data for resume...")
            data_result = self.stages['data'].run(self.state)
            self.results['data'] = data_result

        # Run stages
        try:
            for stage_name in STAGE_ORDER:
                # Skip stages before start point
                if STAGE_ORDER.index(stage_name) < STAGE_ORDER.index(start_stage):
                    logger.info(f"Skipping {stage_name} (before resume point)")
                    continue

                # Run stage
                self._run_stage(stage_name)

                # Check stop condition
                if stop_after and stage_name == stop_after:
                    logger.info(f"Stopping after {stage_name} (as requested)")
                    break

                # Check if pipeline should abort
                if self._should_abort():
                    logger.warning("Pipeline aborted due to critical failure")
                    break

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if self.state.current_stage:
                self.state.mark_stage_failed(self.state.current_stage, str(e))
            raise

        # Calculate timing
        elapsed = time.time() - start_time

        # Print final status
        self._print_summary(elapsed)

        # Auto-regenerate cross-run leaderboard
        self._regenerate_index()

        return {
            'state': self.state,
            'results': self.results,
            'elapsed_seconds': elapsed,
            'report_path': self.results.get('report', {}).get('report_path'),
        }

    def _run_stage(self, stage_name: str):
        """Run a single stage."""
        stage = self.stages.get(stage_name)
        if not stage:
            raise ValueError(f"Unknown stage: {stage_name}")

        # Mark started
        self.state.mark_stage_started(stage_name)
        logger.info(f"\n{'='*70}")
        logger.info(f"RUNNING STAGE: {stage_name.upper()}")
        logger.info(f"{'='*70}")

        try:
            # Execute stage with appropriate inputs
            result = self._execute_stage(stage_name, stage)

            # Store result
            self.results[stage_name] = result

            # Mark completed
            summary = result.get('summary', {})
            self.state.mark_stage_completed(stage_name, summary=summary)

            # Free memory from completed stages
            self._cleanup_after_stage(stage_name)

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.state.mark_stage_failed(stage_name, str(e))
            raise

    def _execute_stage(self, stage_name: str, stage) -> Dict[str, Any]:
        """Execute stage with proper inputs."""

        if stage_name == 'data':
            return stage.run(self.state)

        elif stage_name == 'optimization':
            data_result = self.results.get('data', {})
            return stage.run(
                self.state,
                data_result['df_back'],
                data_result['df_forward'],
            )

        elif stage_name == 'walkforward':
            data_result = self.results.get('data', {})
            opt_result = self.results.get('optimization', {})
            candidates = opt_result.get('candidates', self.state.candidates)
            return stage.run(
                self.state,
                data_result['df'],
                candidates,
                back_end_date=data_result['df_back'].index[-1],
            )

        elif stage_name == 'stability':
            data_result = self.results.get('data', {})
            wf_result = self.results.get('walkforward', {})
            candidates = wf_result.get('candidates', self.state.candidates)
            return stage.run(
                self.state,
                data_result['df_back'],
                data_result['df_forward'],
                candidates,
            )

        elif stage_name == 'montecarlo':
            data_result = self.results.get('data', {})
            stab_result = self.results.get('stability', {})
            candidates = stab_result.get('candidates', self.state.candidates)
            return stage.run(
                self.state,
                data_result['df_back'],
                data_result['df_forward'],
                candidates,
            )

        elif stage_name == 'confidence':
            mc_result = self.results.get('montecarlo', {})
            candidates = mc_result.get('candidates', self.state.candidates)
            return stage.run(
                self.state,
                candidates,
            )

        elif stage_name == 'report':
            conf_result = self.results.get('confidence', {})
            candidates = conf_result.get('candidates', self.state.candidates)
            return stage.run(
                self.state,
                candidates,
                self.results.get('data', {}),
                self.results.get('optimization', {}),
                self.results.get('walkforward', {}),
                self.results.get('stability', {}),
                self.results.get('montecarlo', {}),
                self.results.get('confidence', {}),
            )

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def _cleanup_after_stage(self, completed_stage: str):
        """Free memory from stages that are no longer needed.

        The pipeline accumulates all stage results in self.results,
        but most data is only needed by the next 1-2 stages. After that,
        we can release large objects (DataFrames, optimizer instances,
        trade arrays) to prevent OOM on constrained systems.
        """
        # After walkforward: release full dataset (only WF uses it) and trim optimization
        if completed_stage == 'walkforward':
            data_result = self.results.get('data', {})
            if 'df' in data_result:
                del data_result['df']
                logger.debug("Released full dataset DataFrame")
            if 'optimization' in self.results:
                self.results['optimization'] = {
                    'candidates': self.results['optimization'].get('candidates', []),
                    'summary': self.results['optimization'].get('summary', {}),
                }
                logger.debug("Trimmed optimization results")

        # Keep df_back/df_forward after MC — the report stage needs them
        # as fallback when the confidence-best candidate differs from the
        # combined-rank-best candidate (which is the one MC puts trade_details on).

        gc.collect()

    def _should_abort(self) -> bool:
        """Check if pipeline should abort."""
        # Fix Finding 7: abort if data validation failed
        data_result = self.results.get('data', {})
        if (self.state.stages['data'].status == StageStatus.COMPLETED and
            data_result.get('validation_passed') is False):
            logger.warning("Data validation failed - aborting pipeline")
            return True

        # Abort if no candidates after optimization
        # (Later stages can proceed with empty candidates to generate RED report)
        # Check both results cache and state for candidates (state persists across resumes)
        opt_candidates = self.results.get('optimization', {}).get('candidates', None)
        if opt_candidates is None:
            # When resuming, optimization results may not be in cache - check state
            opt_candidates = self.state.candidates

        if (self.state.stages['optimization'].status == StageStatus.COMPLETED and
            len(opt_candidates) == 0):
            logger.warning("No valid candidates from optimization - aborting pipeline")
            return True

        return False

    def _print_banner(self):
        """Print pipeline startup banner."""
        print("\n")
        print("+" + "=" * 68 + "+")
        print("|" + " " * 20 + "OANDA TRADING SYSTEM" + " " * 28 + "|")
        print("|" + " " * 20 + "E2E Validation Pipeline" + " " * 25 + "|")
        print("+" + "=" * 68 + "+")
        print(f"|  Pair:      {self.config.pair:<54} |")
        print(f"|  Timeframe: {self.config.timeframe:<54} |")
        print(f"|  Strategy:  {self.config.strategy_name:<54} |")
        if self.state.description:
            desc_display = self.state.description[:54]
            print(f"|  Desc:      {desc_display:<54} |")
        print(f"|  Run ID:    {self.state.run_id:<54} |")
        print("+" + "=" * 68 + "+")
        print()

    def _print_summary(self, elapsed: float):
        """Print final pipeline summary."""
        print("\n")
        print("+" + "=" * 74 + "+")
        print("|" + " " * 25 + "PIPELINE COMPLETE" + " " * 32 + "|")
        print("+" + "=" * 74 + "+")

        # Stage status
        for stage_name in STAGE_ORDER:
            stage = self.state.stages[stage_name]
            status_icon = {
                StageStatus.COMPLETED: "[OK]",
                StageStatus.FAILED: "[X]",
                StageStatus.SKIPPED: "[--]",
                StageStatus.PENDING: "[ ]",
                StageStatus.RUNNING: "[..]",
            }.get(stage.status, "[?]")

            duration = f"{stage.duration_seconds:.1f}s" if stage.duration_seconds > 0 else "-"
            print(f"|  {status_icon} {stage_name:<18} {stage.status.value:<12} {duration:<10}       |")

        print("+" + "=" * 74 + "+")

        # Final decision
        if self.state.final_score > 0:
            rating_display = {
                'GREEN': '[GREEN]',
                'YELLOW': '[YELLOW]',
                'RED': '[RED]',
            }.get(self.state.final_rating, self.state.final_rating)

            print(f"|  CONFIDENCE SCORE: {self.state.final_score:>5.1f}/100" + " " * 44 + "|")
            print(f"|  RATING: {rating_display:<62} |")

        print("+" + "=" * 74 + "+")
        print(f"|  Total time: {elapsed/60:.1f} minutes" + " " * 50 + "|")
        print(f"|  Run directory: {str(self.state.run_dir):<55} |")

        # Report path
        report_path = self.results.get('report', {}).get('report_path')
        if report_path:
            print(f"|  Report: file://{str(report_path):<54} |")

        print("+" + "=" * 74 + "+")
        print()

    def _regenerate_index(self):
        """Regenerate the cross-run leaderboard index.html."""
        try:
            from scripts.generate_index import collect_runs, build_index_html
            pipelines_dir = str(self.config.output_dir)
            runs = collect_runs(pipelines_dir)
            html = build_index_html(runs)
            index_path = Path(pipelines_dir) / 'index.html'
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Leaderboard updated: {index_path}")
        except Exception as e:
            logger.warning(f"Could not update leaderboard: {e}")
