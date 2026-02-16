# Web Dashboard & Vercel Deployment

## Overview

The OANDA Trading System has a unified web dashboard deployed at **https://oandasystem.vercel.app/**. It combines two previously separate interfaces into a single-page application with two tabs:

1. **Trading** (`#dashboard`) — Real-time strategy monitoring via VPS proxy
2. **Pipeline** (`#leaderboard`) — Historical pipeline run leaderboard with comprehensive detail panels

## Architecture

```
Vercel (oandasystem.vercel.app)               VPS (104.128.63.239:8080)
┌──────────────────────────────┐              ┌─────────────────────────┐
│  Static Frontend             │              │  FastAPI (unchanged)    │
│    /            → Dashboard  │              │  /api/status            │
│    /#leaderboard→ Pipeline   │              │  /api/trades/{id}       │
│                              │              │  /api/performance/{id}  │
│  External Rewrite (proxy)    │  ──proxy──>  │  /api/service/{id}/...  │
│    /api/vps/* → VPS /api/*   │              │                         │
│                              │              │                         │
│  Static Data                 │              │                         │
│    /data/leaderboard.json    │              │                         │
└──────────────────────────────┘              └─────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `web/index.html` | The entire SPA (~1350 lines). Dashboard + Leaderboard views, all JS inline |
| `web/data/leaderboard.json` | Generated JSON with all pipeline run data (~4.2MB for 204 runs) |
| `web/api/vps/[...path].js` | Serverless function (NOT currently used — replaced by external rewrite) |
| `scripts/build_web.py` | Generates `leaderboard.json` from pipeline results |
| `scripts/generate_index.py` | Generates the original static `results/pipelines/index.html` + calls `build_web.py` |
| `vercel.json` | Vercel config: output directory, rewrites (VPS proxy + SPA fallback) |
| `package.json` | Minimal Node.js config for Vercel |

## How the VPS Proxy Works

The VPS runs a FastAPI dashboard on port 8080 (HTTP). Vercel serves over HTTPS. To avoid mixed-content issues, Vercel's external rewrite proxies requests server-side:

```json
// vercel.json
{
  "rewrites": [
    { "source": "/api/vps/:path*", "destination": "http://104.128.63.239:8080/api/:path*" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

The browser only talks to Vercel (HTTPS). Vercel forwards `/api/vps/*` to the VPS (HTTP) on the server side. Order matters — the VPS rewrite must come before the SPA catch-all.

**Previous approach (abandoned):** A serverless function at `web/api/vps/[...path].js` was supposed to proxy requests, but Vercel's `outputDirectory: "web"` setting prevented the function from being discovered. The function file still exists but is unused.

**Requirements for VPS proxy to work:**
- VPS Windows Firewall must allow inbound TCP 8080
- VPS must be reachable from Vercel's edge network (no ISP/router-level blocking)
- FastAPI dashboard must be running on the VPS

## Leaderboard Data Pipeline

### Data Flow

```
Pipeline run completes
  → results/pipelines/{run_id}/state.json + report_data.json created
  → python scripts/build_web.py (or scripts/generate_index.py which calls it)
  → web/data/leaderboard.json regenerated
  → git add + commit + push
  → Vercel auto-deploys from GitHub push
```

### JSON Structure (`leaderboard.json`)

The JSON has three top-level sections:

```
{
  "runs": [...],           // 204 entries — table data for every pipeline run
  "compare": {...},        // 60 entries — equity curves for comparison overlay
  "detail": {...},         // 116 entries — comprehensive data for detail panel
  "generated_at": "..."
}
```

**`runs`** — One entry per pipeline run. Fields: run_id, pair, timeframe, strategy, description, score, rating, status, created, stages_done, back/forward sharpe/trades/return/max_dd, forward_back_ratio, win_rate, net_profit, n_candidates, total_time_min. Used for the main table.

**`compare`** — Keyed by run_id. Contains equity curve arrays (back_equity, forward_equity), basic metrics (R², returns, trades, sharpe, win_rate, profit_factor, stability_mean), and meta/decision data. Used for the multi-run comparison overlay.

**`detail`** — Keyed by run_id. The richest data, extracted from `report_data.json`. Contains:

| Section | Key Fields |
|---------|-----------|
| `montecarlo` | Return/DD percentiles (5th-95th), VaR95, expected shortfall, P(positive/above 5%/above 10%), bootstrap Sharpe/WR/PF confidence intervals, permutation p-value and significance |
| `confidence` | Total score, rating, recommendation, 6 component scores (backtest_quality, forward_back, walkforward, stability, montecarlo, sharpe), weights, raw_values |
| `walkforward` | Stats (n_windows, n_passed, pass_rate, OOS counts, mean sharpe/return), per-window array (trades, sharpe, sortino, R², return, max_dd, win_rate, PF, quality_score, passed, out_of_sample, ulcer) |
| `stability` | Rating (STABLE/MODERATE/FRAGILE), mean/min stability, n_stable/n_unstable/n_skipped |
| `trade_summary` | Total trades, net profit, gross profit/loss, avg win/loss, largest win/loss, max consecutive W/L, long/short split, exit_reason_counts, avg bars held, avg MFE/MAE |
| `drawdown_curve` | Array of {trade_num, timestamp, drawdown} points |
| `monthly_returns` | Nested dict: year → month → return% |
| `extended` | Back/forward sortino, ulcer, quality_score, forward metrics, combined_rank, full params dict |
| `config` | Initial capital, spread, data years, train/test/holdout months, trial counts |
| `decision` | Score, rating, recommendation |

### Data Collection Code

`scripts/build_web.py` has three collection functions:

1. **`collect_runs()`** (from `generate_index.py`) — Reads `state.json` from each run directory. Lightweight.
2. **`collect_compare_data()`** (from `generate_index.py`) — Reads `report_data.json` for equity curves and basic metrics. Used for comparison overlay.
3. **`collect_detail_data()`** (in `build_web.py`) — Reads `report_data.json` comprehensively. Extracts MC, confidence, WF, stability, trade summary, drawdown, monthly returns, extended metrics, config, and params.

## Frontend Structure (`web/index.html`)

### Views

The SPA has two views, toggled by the nav tabs:

- **Dashboard view** (`#dashboard`) — Fetches `/api/vps/status` every 15s. Shows summary cards (P&L, positions, trades, running count, errors) and an instances table with expandable performance/trades panels. Service control buttons (start/restart/stop). Shows "VPS Offline" banner when API unreachable.

- **Leaderboard view** (`#leaderboard`) — Loads `/data/leaderboard.json` on first visit. Shows summary strip, filter bar (pair, timeframe, rating, status), sortable table, checkbox-based comparison overlay, and run detail panel.

### Run Detail Panel

Clicking any table row opens a comprehensive detail panel with 10 collapsible sections:

1. **Header** — Run ID, pair/TF/strategy, score + rating badge, recommendation, config strip
2. **Performance Overview** — Back vs Forward table (Quality Score, Return, Trades, Win Rate, PF, Sharpe, Sortino, R², Max DD, Ulcer), F/B ratio banner, recovery factor
3. **Confidence Breakdown** — Total score display, 6 horizontal bars with score × weight
4. **Walk-Forward Analysis** — Pass rate summary, OOS count, per-window table with pass/fail dots
5. **Equity + Drawdown Charts** — Plotly equity curve (back solid + forward dashed) + drawdown area chart
6. **Monte Carlo** — 4 cards: return distribution, DD distribution, risk metrics, bootstrap CIs + permutation
7. **Stability** — Rating badge, mean/min stability, param counts
8. **Trade Analysis** — Metric cards grid + exit reason breakdown with bars
9. **Monthly Returns** — Year × month heatmap with color-coded cells
10. **Parameters** — Two-column table of all optimized parameter values (collapsed by default)

### Key JS Functions

| Function | Purpose |
|----------|---------|
| `switchView(v)` | Toggle between dashboard/leaderboard |
| `fetchDash()` | Fetch VPS status, render dashboard cards and instances |
| `loadLeaderboard()` | Fetch leaderboard.json, call buildLB() |
| `buildLB()` | Build summary strip, filters, table rows, bind events |
| `showRunDetail(runId)` | Build and show the 10-section detail panel |
| `rdHeader/rdPerformance/rdConfidence/rdWalkforward/rdMonteCarlo/rdStability/rdTrades/rdMonthly/rdParams` | Section builder functions, each returns HTML string |
| `rdRenderCharts(comp, det)` | Render Plotly equity + drawdown charts |
| `runComparison()` | Build equity overlay + metrics table for checked runs |
| `rdToggle(section)` | Toggle collapsed class on detail sections |

### Formatting Helpers

| Function | Purpose |
|----------|---------|
| `rdN(v, dec)` | Format number with commas (e.g., `1,234.56`) |
| `rdPct(v, dec)` | Format percentage where v is already in % |
| `rdPctR(v, dec)` | Format 0-1 ratio as percentage (multiplies by 100) |
| `rdDlr(v)` | Format dollar amount with $ sign |
| `rdSc(score)` | Score (0-100) → color (red/yellow/green) |
| `rdRc(rating)` | Rating string → [bg, fg] color pair |
| `rdVc(value)` | Value → green (positive) / red (negative) / grey (zero) |

## Deployment

### Auto-deploy via GitHub

The Vercel project is connected to `github.com/longytravel/oandasystem` on the `main` branch. Every push triggers an auto-deployment. Typical deploy time: ~30 seconds.

### Manual update workflow

```bash
# After pipeline runs, regenerate the leaderboard JSON:
cd scripts
python build_web.py

# Or use generate_index.py which also calls build_web.py:
python generate_index.py --inject-backlink

# Commit and push to trigger deploy:
cd ..
git add web/data/leaderboard.json
git commit -m "Update leaderboard data"
git push
```

### Vercel CLI

The CLI requires `--scope marks-projects-724fd891` for all commands in non-interactive environments. Linking is fiddly in CI; prefer GitHub push for deployments.

```bash
npx vercel whoami                    # Check auth
npx vercel project ls                # List projects (works without scope)
npx vercel deploy --prod --yes       # Manual deploy (needs scope)
```

## Known Issues & Limitations

1. **VPS proxy may be blocked**: The external rewrite requires the VPS to be reachable from Vercel's edge network. If the ISP or router blocks inbound connections on port 8080, the Trading tab shows "VPS Offline" even though the firewall rule exists.

2. **Serverless function not working**: The `web/api/vps/[...path].js` function is unused. With `outputDirectory: "web"`, Vercel doesn't discover serverless functions inside the output directory. The external rewrite approach works instead.

3. **Large JSON payload**: At 4.2MB, the leaderboard JSON takes a moment to load on slow connections. The `detail` section is the largest part. Could be optimized by lazy-loading detail data per-run if this becomes a problem.

4. **No real-time leaderboard updates**: Pipeline data is static — updated only when `build_web.py` is run and the JSON is committed/pushed. Live trading data (Trading tab) refreshes every 15 seconds.

5. **WebFetch caching**: The WebFetch tool has a 15-minute cache, which can make it appear that deployments haven't taken effect when testing.

## Design Decisions

- **Single HTML file**: Everything is in `web/index.html` (~1350 lines) to keep deployment simple — no build step, no bundler, no framework. Plotly.js and fonts loaded from CDNs.
- **External rewrite over serverless**: Simpler, no cold starts, no function discovery issues with `outputDirectory`.
- **Three-tier JSON**: `runs` (table), `compare` (overlay), `detail` (drill-down) balances load time with data richness. Only `detail` was added in the Feb 15 update.
- **Collapsible sections**: The detail panel shows a lot of data. Sections collapse to let users focus on what matters. Parameters collapsed by default since they're reference data.
- **Dark theme**: Consistent with the VPS FastAPI dashboard and pipeline report HTML. Colors: `#0a0e1a` background, `#111827` cards, `#1e293b` borders, `#3b82f6` accents.
