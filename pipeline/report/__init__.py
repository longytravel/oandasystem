"""Report package for pipeline HTML dashboard generation."""
from pipeline.report.html_builder import build_report_html
from pipeline.report.data_collector import collect_report_data

__all__ = ['build_report_html', 'collect_report_data']
