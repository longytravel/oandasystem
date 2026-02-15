"""Build JSON data for the web leaderboard.

Generates web/data/leaderboard.json from pipeline run data, reusing the
same collect_runs() and collect_compare_data() functions from generate_index.py.

Usage:
    python scripts/build_web.py
    python scripts/build_web.py --pipelines-dir results/pipelines
"""
import json
import os
import argparse
from datetime import datetime
from pathlib import Path

from generate_index import collect_runs, collect_compare_data


def _make_serializable(obj):
    """Convert any non-JSON-serializable objects (datetimes, etc.) to strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return obj


def build_web_data(pipelines_dir: str, output_path: str) -> dict:
    """Build and write the leaderboard JSON file.

    Returns the data dict that was written, or None if no pipelines dir exists.
    """
    if not os.path.exists(pipelines_dir):
        print(f"No pipelines directory found at {pipelines_dir}")
        return None

    runs = collect_runs(pipelines_dir)
    compare = collect_compare_data(pipelines_dir, runs)

    payload = {
        'runs': _make_serializable(runs),
        'compare': _make_serializable(compare),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))

    file_size = os.path.getsize(output_path)
    print(f"Web data: {len(runs)} runs, {len(compare)} with compare data")
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return payload


def main():
    parser = argparse.ArgumentParser(description='Build web leaderboard JSON')
    parser.add_argument('--pipelines-dir', default=None,
                        help='Pipelines directory (default: results/pipelines)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path (default: web/data/leaderboard.json)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    pipelines_dir = args.pipelines_dir or str(project_root / 'results' / 'pipelines')
    output_path = args.output or str(project_root / 'web' / 'data' / 'leaderboard.json')

    build_web_data(pipelines_dir, output_path)


if __name__ == '__main__':
    main()
