"""
Pipeline State Management - Resumable state for pipeline execution.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

import numpy as np
from loguru import logger


def _json_serializer(obj):
    """
    Custom JSON serializer that properly handles numpy types.

    FIX: The old `default=str` approach silently converted numpy booleans
    to strings ("True"/"False"), which when reloaded are truthy in Python.
    This properly converts numpy types to native Python types.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, Path)):
        return str(obj)
    elif hasattr(obj, 'tolist'):  # Other array-like objects
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        # Last resort - but log a warning so we can fix it
        logger.warning(f"JSON serializer fell back to str() for type {type(obj)}: {obj}")
        return str(obj)


class StageStatus(str, Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


STAGE_ORDER = [
    'data',
    'optimization',
    'walkforward',
    'stability',
    'montecarlo',
    'confidence',
    'report',
]


@dataclass
class StageState:
    """State for a single stage."""
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    output_file: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'duration_seconds': self.duration_seconds,
            'error_message': self.error_message,
            'output_file': self.output_file,
            'summary': self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageState':
        return cls(
            name=data['name'],
            status=StageStatus(data['status']),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            duration_seconds=data.get('duration_seconds', 0.0),
            error_message=data.get('error_message'),
            output_file=data.get('output_file'),
            summary=data.get('summary', {}),
        )


@dataclass
class PipelineState:
    """
    Complete pipeline state for resumability.

    Saves after each stage to allow resuming interrupted pipelines.
    """
    run_id: str
    run_dir: Path
    pair: str
    timeframe: str
    strategy_name: str
    description: str = ''
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_stage: str = 'data'
    stages: Dict[str, StageState] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # Shared data between stages
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    best_candidate: Optional[Dict[str, Any]] = None
    final_score: float = 0.0
    final_rating: str = "RED"

    def __post_init__(self):
        """Initialize stage states if not present."""
        for stage_name in STAGE_ORDER:
            if stage_name not in self.stages:
                self.stages[stage_name] = StageState(name=stage_name)

    @property
    def state_file(self) -> Path:
        """Path to state file."""
        return self.run_dir / "state.json"

    def mark_stage_started(self, stage_name: str):
        """Mark a stage as started."""
        if stage_name in self.stages:
            self.stages[stage_name].status = StageStatus.RUNNING
            self.stages[stage_name].started_at = datetime.now().isoformat()
            self.current_stage = stage_name
            self.updated_at = datetime.now().isoformat()
            self.save()

    def mark_stage_completed(
        self,
        stage_name: str,
        summary: Dict[str, Any] = None,
        output_file: str = None
    ):
        """Mark a stage as completed."""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.status = StageStatus.COMPLETED
            stage.completed_at = datetime.now().isoformat()
            if stage.started_at:
                start = datetime.fromisoformat(stage.started_at)
                end = datetime.fromisoformat(stage.completed_at)
                stage.duration_seconds = (end - start).total_seconds()
            if summary:
                stage.summary = summary
            if output_file:
                stage.output_file = output_file
            self.updated_at = datetime.now().isoformat()
            self.save()

    def mark_stage_failed(self, stage_name: str, error: str):
        """Mark a stage as failed."""
        if stage_name in self.stages:
            self.stages[stage_name].status = StageStatus.FAILED
            self.stages[stage_name].completed_at = datetime.now().isoformat()
            self.stages[stage_name].error_message = error
            self.updated_at = datetime.now().isoformat()
            self.save()

    def mark_stage_skipped(self, stage_name: str, reason: str = None):
        """Mark a stage as skipped."""
        if stage_name in self.stages:
            self.stages[stage_name].status = StageStatus.SKIPPED
            if reason:
                self.stages[stage_name].summary['skip_reason'] = reason
            self.updated_at = datetime.now().isoformat()
            self.save()

    def get_next_stage(self) -> Optional[str]:
        """Get the next pending stage to run."""
        for stage_name in STAGE_ORDER:
            if self.stages[stage_name].status == StageStatus.PENDING:
                return stage_name
        return None

    def get_last_completed_stage(self) -> Optional[str]:
        """Get the last completed stage."""
        last = None
        for stage_name in STAGE_ORDER:
            if self.stages[stage_name].status == StageStatus.COMPLETED:
                last = stage_name
        return last

    def is_completed(self) -> bool:
        """Check if pipeline is fully completed."""
        return all(
            s.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
            for s in self.stages.values()
        )

    def has_failed(self) -> bool:
        """Check if any stage has failed."""
        return any(s.status == StageStatus.FAILED for s in self.stages.values())

    def get_stage_output_path(self, stage_name: str) -> Path:
        """Get output file path for a stage."""
        return self.run_dir / f"stage_{stage_name}.json"

    def save_stage_output(self, stage_name: str, data: Dict[str, Any]):
        """Save stage output to file."""
        output_path = self.get_stage_output_path(stage_name)
        with open(output_path, 'w') as f:
            # FIX: Use custom serializer for proper numpy type handling
            json.dump(data, f, indent=2, default=_json_serializer)
        self.stages[stage_name].output_file = str(output_path)
        logger.info(f"Saved {stage_name} output to {output_path}")

    def load_stage_output(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Load stage output from file."""
        output_path = self.get_stage_output_path(stage_name)
        if output_path.exists():
            with open(output_path, 'r') as f:
                return json.load(f)
        return None

    def save(self):
        """Save state to file."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            # FIX: Use custom serializer instead of default=str
            # The old approach silently converted numpy bools to strings,
            # which broke resumability ("False" is truthy in Python)
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'run_dir': str(self.run_dir),
            'pair': self.pair,
            'timeframe': self.timeframe,
            'strategy_name': self.strategy_name,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'current_stage': self.current_stage,
            'stages': {name: s.to_dict() for name, s in self.stages.items()},
            'config': self.config,
            'candidates': self.candidates,
            'best_candidate': self.best_candidate,
            'final_score': self.final_score,
            'final_rating': self.final_rating,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create from dictionary."""
        state = cls(
            run_id=data['run_id'],
            run_dir=Path(data['run_dir']),
            pair=data['pair'],
            timeframe=data['timeframe'],
            strategy_name=data['strategy_name'],
            description=data.get('description', ''),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            current_stage=data.get('current_stage', 'data'),
            config=data.get('config', {}),
            candidates=data.get('candidates', []),
            best_candidate=data.get('best_candidate'),
            final_score=data.get('final_score', 0.0),
            final_rating=data.get('final_rating', 'RED'),
        )

        # Load stage states
        if 'stages' in data:
            for name, stage_data in data['stages'].items():
                state.stages[name] = StageState.from_dict(stage_data)

        return state

    @classmethod
    def load(cls, state_file: Path) -> 'PipelineState':
        """Load state from file."""
        with open(state_file, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_new(
        cls,
        pair: str,
        timeframe: str,
        strategy_name: str,
        output_dir: Path,
        config: Dict[str, Any] = None,
        description: str = '',
    ) -> 'PipelineState':
        """Create a new pipeline state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{pair}_{timeframe}_{timestamp}"
        run_dir = output_dir / run_id

        state = cls(
            run_id=run_id,
            run_dir=run_dir,
            pair=pair,
            timeframe=timeframe,
            strategy_name=strategy_name,
            description=description,
            config=config or {},
        )

        state.save()
        logger.info(f"Created new pipeline run: {run_id}")
        logger.info(f"Run directory: {run_dir}")

        return state

    def print_status(self):
        """Print pipeline status summary."""
        print("\n" + "=" * 70)
        print(f"PIPELINE STATUS: {self.run_id}")
        print("=" * 70)
        print(f"  Pair:     {self.pair}")
        print(f"  Timeframe: {self.timeframe}")
        print(f"  Strategy:  {self.strategy_name}")
        if self.description:
            print(f"  Desc:      {self.description}")
        print(f"  Created:   {self.created_at[:19]}")
        print(f"  Updated:   {self.updated_at[:19]}")
        print()
        print(f"{'Stage':<20} {'Status':<12} {'Duration':<12} {'Summary'}")
        print("-" * 70)

        for stage_name in STAGE_ORDER:
            stage = self.stages[stage_name]
            status_icon = {
                StageStatus.PENDING: "[ ]",
                StageStatus.RUNNING: "[..]",
                StageStatus.COMPLETED: "[OK]",
                StageStatus.FAILED: "[X]",
                StageStatus.SKIPPED: "[--]",
            }.get(stage.status, "[?]")

            duration = f"{stage.duration_seconds:.1f}s" if stage.duration_seconds > 0 else "-"

            summary_str = ""
            if stage.summary:
                key_items = list(stage.summary.items())[:2]
                summary_str = ", ".join(f"{k}={v}" for k, v in key_items)
            elif stage.error_message:
                summary_str = f"Error: {stage.error_message[:30]}..."

            print(f"  {status_icon} {stage_name:<14} {stage.status.value:<12} {duration:<12} {summary_str}")

        print("=" * 70)

        if self.final_score > 0:
            print(f"\n  Final Score: {self.final_score:.1f}/100")
            print(f"  Rating:      {self.final_rating}")
        print()
