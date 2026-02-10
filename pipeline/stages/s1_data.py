"""
Stage 1: Data Download & Validation

Downloads OANDA data and validates quality:
- Gap detection
- Bad candle detection
- Minimum data requirements

Data Source Strategy:
====================
Prefers M1 data when available - builds higher TFs (M5, M15, M30, H1, H4)
from M1 for more accurate OHLC values (captures true highs/lows).

OANDA Limitations:
- Candle data: Available M1 to Monthly, ~5+ years history
- Tick data: NOT truly available (max 24 ticks/min, aggregated)
- For true tick data, use Dukascopy, TrueFX, or HistData.com
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
from pathlib import Path

from loguru import logger

from data.download import download_data, load_data, get_data_info, get_data_filepath
from pipeline.config import PipelineConfig
from pipeline.state import PipelineState


class DataStage:
    """Stage 1: Data download and quality validation."""

    name = "data"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: PipelineState) -> Dict[str, Any]:
        """
        Execute data stage.

        Returns:
            Dict with keys:
            - df: Full DataFrame
            - df_back: Back-test portion
            - df_forward: Forward test portion
            - quality_report: Data quality metrics
            - validation_passed: bool
        """
        logger.info("=" * 70)
        logger.info("STAGE 1: DATA DOWNLOAD & VALIDATION")
        logger.info("=" * 70)

        # Download/load data
        # Prefers M1 source when available for more accurate OHLC
        logger.info(f"Loading {self.config.pair} {self.config.timeframe}...")
        df = load_data(
            instrument=self.config.pair,
            timeframe=self.config.timeframe,
            auto_download=True,
            years=self.config.data.years,
            prefer_m1=True  # Build from M1 for accurate highs/lows
        )

        logger.info(f"Loaded {len(df):,} candles")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

        # Validate data quality
        quality_report = self._validate_quality(df)

        # Check minimum requirements
        validation_passed = self._check_requirements(df, quality_report)

        # Split into back/forward periods
        if self.config.data.holdout_months > 0:
            total_months = self.config.data.years * 12
            back_ratio = 1.0 - (self.config.data.holdout_months / total_months)
            back_ratio = max(0.5, min(0.9, back_ratio))  # Safety clamp
            logger.info(f"  Holdout: {self.config.data.holdout_months} months, back_ratio={back_ratio:.2f}")
        else:
            back_ratio = 0.8
        df_back, df_forward = self._split_data(df, back_ratio=back_ratio)

        logger.info(f"\nData split:")
        logger.info(f"  Back:    {len(df_back):,} candles ({df_back.index[0].date()} to {df_back.index[-1].date()})")
        logger.info(f"  Forward: {len(df_forward):,} candles ({df_forward.index[0].date()} to {df_forward.index[-1].date()})")

        # Check data source (M1 or direct)
        m1_filepath = get_data_filepath(self.config.pair, "M1")
        data_source = "M1 (built)" if m1_filepath.exists() else f"{self.config.timeframe} (direct)"

        # Build summary
        summary = {
            'total_candles': len(df),
            'back_candles': len(df_back),
            'forward_candles': len(df_forward),
            'date_start': str(df.index[0]),
            'date_end': str(df.index[-1]),
            'quality_score': quality_report['quality_score'],
            'validation_passed': validation_passed,
            'data_source': data_source,
        }

        # Save stage output
        output_data = {
            'summary': summary,
            'quality_report': quality_report,
            'back_period': {
                'start': str(df_back.index[0]),
                'end': str(df_back.index[-1]),
                'candles': len(df_back),
            },
            'forward_period': {
                'start': str(df_forward.index[0]),
                'end': str(df_forward.index[-1]),
                'candles': len(df_forward),
            },
        }
        state.save_stage_output('data', output_data)

        return {
            'df': df,
            'df_back': df_back,
            'df_forward': df_forward,
            'quality_report': quality_report,
            'validation_passed': validation_passed,
            'summary': summary,
        }

    def _validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.

        Checks:
        - Gaps in timestamps
        - Zero/missing values
        - Anomalous candles (huge wicks, zero range)
        - Weekend gaps (expected)
        """
        logger.info("\nValidating data quality...")

        issues = []
        warnings = []

        # 1. Check for gaps
        gaps = self._detect_gaps(df)
        if gaps['n_gaps'] > 0:
            # Only count gaps longer than expected
            significant_gaps = [g for g in gaps['gaps'] if g['hours'] > self.config.data.max_gap_hours]
            if significant_gaps:
                issues.append(f"{len(significant_gaps)} significant gaps (>{self.config.data.max_gap_hours}h)")
            if gaps['n_gaps'] - len(significant_gaps) > 0:
                warnings.append(f"{gaps['n_gaps'] - len(significant_gaps)} minor gaps (weekends/holidays)")

        # 2. Check for zero/missing values
        zero_check = self._check_zero_values(df)
        if zero_check['n_zero_rows'] > 0:
            issues.append(f"{zero_check['n_zero_rows']} rows with zero values")

        # 3. Check for anomalous candles
        anomaly_check = self._detect_anomalies(df)
        if anomaly_check['n_anomalies'] > 0:
            warnings.append(f"{anomaly_check['n_anomalies']} anomalous candles")

        # Calculate quality score (0-100)
        quality_score = 100.0
        quality_score -= len(issues) * 20  # Major issues
        quality_score -= len(warnings) * 5  # Minor warnings
        quality_score = max(0, quality_score)

        quality_report = {
            'quality_score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'gaps': gaps,
            'zero_values': zero_check,
            'anomalies': anomaly_check,
        }

        # Print quality report
        logger.info(f"  Quality Score: {quality_score:.0f}/100")
        if issues:
            for issue in issues:
                logger.warning(f"  ISSUE: {issue}")
        if warnings:
            for warning in warnings:
                logger.info(f"  Warning: {warning}")
        if not issues and not warnings:
            logger.info("  No issues detected")

        return quality_report

    def _detect_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect gaps in timestamp sequence."""
        # Expected gap based on timeframe
        tf_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D": 1440
        }
        expected_gap = timedelta(minutes=tf_minutes.get(self.config.timeframe, 60))

        gaps = []
        time_diffs = df.index.to_series().diff()

        for i, diff in enumerate(time_diffs):
            if pd.isna(diff):
                continue

            # Allow up to 3x expected gap for market closures
            if diff > expected_gap * 3:
                gap_hours = diff.total_seconds() / 3600
                # Skip weekend gaps (expected)
                start_day = df.index[i-1].dayofweek if i > 0 else 0
                end_day = df.index[i].dayofweek

                is_weekend = (start_day == 4 and end_day == 6) or (start_day == 4 and gap_hours > 48)

                gaps.append({
                    'index': i,
                    'from': str(df.index[i-1]) if i > 0 else None,
                    'to': str(df.index[i]),
                    'hours': round(gap_hours, 1),
                    'is_weekend': is_weekend,
                })

        return {
            'n_gaps': len(gaps),
            'gaps': gaps[:10],  # Only keep first 10 for report
            'max_gap_hours': max([g['hours'] for g in gaps]) if gaps else 0,
        }

    def _check_zero_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for zero or missing values."""
        zero_rows = (
            (df['open'] == 0) |
            (df['high'] == 0) |
            (df['low'] == 0) |
            (df['close'] == 0)
        )

        null_rows = df[['open', 'high', 'low', 'close']].isnull().any(axis=1)

        return {
            'n_zero_rows': zero_rows.sum(),
            'n_null_rows': null_rows.sum(),
            'zero_pct': round(zero_rows.mean() * 100, 4),
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalous candles."""
        anomalies = []

        # Calculate typical ranges
        typical_range = (df['high'] - df['low']).median()
        typical_body = abs(df['close'] - df['open']).median()

        for i in range(len(df)):
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            candle_body = abs(df['close'].iloc[i] - df['open'].iloc[i])

            # Huge candle (10x typical range)
            if candle_range > typical_range * 10:
                anomalies.append({
                    'index': i,
                    'time': str(df.index[i]),
                    'type': 'huge_range',
                    'value': round(candle_range / typical_range, 1),
                })

            # Zero range candle
            elif candle_range == 0:
                anomalies.append({
                    'index': i,
                    'time': str(df.index[i]),
                    'type': 'zero_range',
                })

            # High/Low violation
            elif df['high'].iloc[i] < df['low'].iloc[i]:
                anomalies.append({
                    'index': i,
                    'time': str(df.index[i]),
                    'type': 'high_low_violation',
                })

        return {
            'n_anomalies': len(anomalies),
            'anomalies': anomalies[:10],  # Only keep first 10
            'typical_range_pips': round(typical_range * 10000, 1),  # For non-JPY pairs
        }

    def _check_requirements(self, df: pd.DataFrame, quality_report: Dict) -> bool:
        """Check if data meets minimum requirements."""
        passed = True

        # Minimum candles
        if len(df) < self.config.data.min_candles:
            logger.error(f"Insufficient data: {len(df)} < {self.config.data.min_candles} required")
            passed = False

        # Quality threshold
        if quality_report['quality_score'] < 50:
            logger.error(f"Quality score too low: {quality_report['quality_score']}")
            passed = False

        # Critical issues
        if quality_report['issues']:
            logger.warning(f"Data has {len(quality_report['issues'])} quality issues")
            # Still pass but with warning

        return passed

    def _split_data(self, df: pd.DataFrame, back_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into back-test and forward-test portions."""
        split_idx = int(len(df) * back_ratio)

        df_back = df.iloc[:split_idx].copy()
        df_forward = df.iloc[split_idx:].copy()

        return df_back, df_forward
