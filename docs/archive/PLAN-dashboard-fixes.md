# Dashboard Data Fixes - Investigation Results and Plan

## Summary of Issues Found

There are **5 distinct bugs** causing missing/empty data in the HTML dashboard reports. All stem from two root causes: (A) key name mismatches between data producers and consumers, and (B) pipeline resume not preserving stage results for the report generator.

---

## Issue 1: Stability Chart Empty - Key Name Mismatch

**Severity**: HIGH - stability bars chart and detailed stability bars always empty
**Affected tabs**: Stability tab (chart + HTML bars)

**Root cause**: The stability stage (`s4_stability.py` via `unified_optimizer.py`) stores per-parameter data under the key `stability.params`, but both `chart_generators.py:stability_bars()` (line 657) and `html_builder.py:_build_stability_tab()` (line 381) read from `stability.per_param`.

**Evidence**:
- `report_data.json`: `best_candidate.stability.per_param` = 0 entries, `best_candidate.stability.params` = 32 entries
- All 13 generated reports show `per_param=0` and `params=N` (where N > 0 for recent runs)

**Fix**: In `chart_generators.py` line 657 and `html_builder.py` line 381, change `stability.get('per_param', {})` to `stability.get('per_param', stability.get('params', {}))` to support both key names. Alternatively, fix the data source to use `per_param` consistently.

**Preferred fix**: Change **two locations** in report consumers to fall back to `params` key:
1. `pipeline/report/chart_generators.py` line 657: `per_param = stability.get('per_param', stability.get('params', {}))`
2. `pipeline/report/html_builder.py` line 381: `per_param = stability.get('per_param', stability.get('params', {}))`

---

## Issue 2: Empty Stage Summaries After Pipeline Resume

**Severity**: MEDIUM - summary metrics show 0/empty in dashboard
**Affected tabs**: Walk-Forward (n_windows metric), Monte Carlo (iterations count), all tabs referencing summary data

**Root cause**: When the pipeline resumes from a later stage (e.g., `--resume-from confidence`), the `Pipeline.results` cache only contains results for stages that actually ran. Earlier stage results are `{}`. The report stage receives `self.results.get('walkforward', {})` which is `{}`, so `collect_report_data()` gets `walkforward_result.get('summary', {})` = `{}`.

**Evidence**:
- Both GREEN reports (20260206_*) were from resumed runs (time gaps between stages)
- Their `walkforward_summary`, `optimization_summary`, `stability_summary`, `montecarlo_summary` are all `{}` (0 keys)
- Older reports that ran straight through have populated summaries (4-6 keys)

**Fix**: In `pipeline/report/data_collector.py`, when a stage summary is empty, fall back to reading the summary from `state.stages[stage_name].summary` (which IS persisted in state.json).

Add to `collect_report_data()` after line 27:
```python
# Fall back to state summaries when stage results are empty (pipeline resume)
def _get_summary(result, stage_name):
    summary = result.get('summary', {})
    if not summary and state:
        stage_info = state.stages.get(stage_name)
        if stage_info and stage_info.summary:
            summary = stage_info.summary
    return summary
```

Then use `_get_summary(walkforward_result, 'walkforward')` instead of `walkforward_result.get('summary', {})` for each stage summary.

---

## Issue 3: walkforward_windows Empty After Resume

**Severity**: LOW (charts still work) - the `walkforward_windows` top-level key is empty
**Affected**: The `_build_wf_table()` won't render the WF detail table if it uses this data (but currently it uses `wf_results` from `best_candidate.walkforward.window_results` which IS populated)

**Root cause**: `data_collector.py` line 77 reads `walkforward_result.get('windows', [])` which is `[]` after resume (walkforward_result is `{}`). However, the WF window definitions (train/test date ranges) aren't the same as window_results (per-candidate results). The `walkforward_windows` key at top level is window *definitions*, not window *results*.

**Evidence**:
- `walkforward_windows` = 0 in both GREEN reports, but WF charts still render because they use `best_candidate.walkforward.window_results`
- The `_build_wf_table()` in html_builder uses `wf_results` (from best_candidate), not `walkforward_windows`

**Fix**: This is low priority since charts work via `best_candidate.walkforward.window_results`. However, for completeness, `data_collector.py` should fall back to extracting window data from best_candidate when walkforward_result is empty. The simplest fix: no change needed since it doesn't affect rendering.

---

## Issue 4: MC Raw Data Stripped from report_data.json (Not a Bug in HTML)

**Severity**: LOW (HTML report works, JSON export doesn't have MC arrays)

**Root cause**: `s7_report.py` lines 67-68 intentionally strip `mc_raw_returns` and `mc_raw_max_dds` from the JSON export for readability. The HTML report works because it reads these arrays from `data['mc_raw_returns']` which IS populated by `data_collector.py` from `best_candidate.montecarlo.raw_returns`.

**Evidence**: The MC histogram charts render correctly in the latest reports. The JSON file just doesn't have the raw arrays (by design).

**Fix**: No fix needed - this is working as designed.

---

## Issue 5: Old Reports (pre-20260206) Have No Trade Details

**Severity**: LOW (historical only) - affects only the 11 older runs

**Root cause**: The `_collect_trade_details()` method was added to `s5_montecarlo.py` on 2026-02-06. Older reports were generated before this feature existed and don't have trade data. The `_regenerate_trade_details()` fallback in `data_collector.py` requires `data_result.get('df_back')` which is only available when the data stage ran.

**Fix**: No code fix needed. If the user wants to regenerate old reports, they can re-run the pipeline from the report stage with `--resume-from report`. The `_regenerate_trade_details()` code already handles this case (line 32 of data_collector.py), as long as data is reloaded (which it is, per pipeline.py line 109-112).

---

## Implementation Plan

### Fix 1: Stability key mismatch (2 files, 2 lines each)

**File: `pipeline/report/chart_generators.py`** line 657:
```python
# Before:
per_param = stability.get('per_param', {})
# After:
per_param = stability.get('per_param', stability.get('params', {}))
```

**File: `pipeline/report/html_builder.py`** line 381:
```python
# Before:
per_param = stability.get('per_param', {})
# After:
per_param = stability.get('per_param', stability.get('params', {}))
```

### Fix 2: Stage summaries fall back to state (1 file)

**File: `pipeline/report/data_collector.py`**:
Add fallback logic to read summaries from `state.stages[name].summary` when stage result dicts are empty. This affects 5 summary fields: `optimization_summary`, `walkforward_summary`, `stability_summary`, `montecarlo_summary`, `confidence_summary`.

### Fix 3 (optional): walkforward_windows fallback

**File: `pipeline/report/data_collector.py`**:
When `walkforward_result.get('windows', [])` is empty, fall back to extracting from state or leave as-is since charts use best_candidate data.

---

## Testing

After applying fixes:
1. Re-run report stage only on existing GREEN run: `python scripts/run_pipeline.py GBP_USD H1 --resume-from report --run-dir results/pipelines/GBP_USD_H1_20260206_151217`
2. Open generated `report.html` and verify:
   - Stability tab: bar chart shows per-parameter stability ratios (was empty)
   - Stability tab: HTML bars show stability percentages (was empty)
   - Walk-Forward tab: summary metrics populated (n_windows, pass_rate)
   - All metric cards show actual values instead of 0
