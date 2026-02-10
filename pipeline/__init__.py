"""
OANDA Trading System - E2E Pipeline

7-Stage pipeline for systematic strategy validation:
1. Data Download & Validation
2. Initial Optimization
3. Walk-Forward Validation
4. Parameter Stability
5. Monte Carlo Simulation
6. Confidence Scoring
7. Report Generation
"""
from pipeline.pipeline import Pipeline
from pipeline.state import PipelineState
from pipeline.config import PipelineConfig

__all__ = ['Pipeline', 'PipelineState', 'PipelineConfig']
