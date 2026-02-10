"""
Stage 7: Report Generation

Generates a self-contained multi-tab HTML dashboard with interactive
Plotly charts. Delegates to pipeline.report package for data enrichment,
chart generation, and HTML assembly.
"""
import json
from typing import Dict, Any, List

from loguru import logger

from pipeline.config import PipelineConfig
from pipeline.state import PipelineState, _json_serializer
from pipeline.report.data_collector import collect_report_data
from pipeline.report.html_builder import build_report_html


class ReportStage:
    """Stage 7: HTML report generation."""

    name = "report"

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        state: PipelineState,
        candidates: List[Dict[str, Any]],
        data_result: Dict[str, Any],
        optimization_result: Dict[str, Any],
        walkforward_result: Dict[str, Any],
        stability_result: Dict[str, Any],
        montecarlo_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive HTML report."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 7: REPORT GENERATION")
        logger.info("=" * 70)

        # Collect and enrich data from all stages
        report_data = collect_report_data(
            state=state,
            config=self.config,
            candidates=candidates,
            data_result=data_result,
            optimization_result=optimization_result,
            walkforward_result=walkforward_result,
            stability_result=stability_result,
            montecarlo_result=montecarlo_result,
            confidence_result=confidence_result,
        )

        # Generate HTML dashboard
        html_content = build_report_html(report_data)

        # Save report
        report_path = state.run_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Report saved to: {report_path}")

        # Save JSON data (strip large arrays for readability)
        json_data = {k: v for k, v in report_data.items()
                     if k not in ('mc_raw_returns', 'mc_raw_max_dds')}
        json_path = state.run_dir / "report_data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=_json_serializer)

        summary = {
            'report_path': str(report_path),
            'json_path': str(json_path),
            'n_tabs': 7,
        }

        # Save stage output
        state.save_stage_output('report', summary)

        return {
            'report_path': report_path,
            'summary': summary,
        }
