# Build: Agent Team Dashboard (Local Web UI)

## What to Build

A **local web-based dashboard** that monitors Claude Code agent team activity in real-time. Opens in a browser window alongside the terminal so you can see all teammates, tasks, and progress at a glance.

## Why

Claude Code agent teams have limited visibility on Windows (no tmux split-pane support). This dashboard fills that gap by reading the team/task files from disk and presenting them in a clean web UI.

## Technical Constraints

- **Windows 11** (no WSL, no tmux)
- **Python 3.x** already installed (used for the OANDA trading system)
- Must run as a simple `python team_dashboard.py` command
- Single file or minimal files - keep it simple
- No heavy dependencies - use stdlib + maybe `watchdog` for file monitoring

## Data Sources

Claude Code stores team/task data locally:

- **Team configs**: `~/.claude/teams/{team-name}/config.json`
  - Contains `members` array with: `name`, `agentId`, `agentType`
  - Team metadata
- **Task lists**: `~/.claude/tasks/{team-name}/`
  - Individual task JSON files
  - Each task has: `id`, `subject`, `description`, `status` (pending/in_progress/completed), `owner`, `blockedBy`, `blocks`, `activeForm`
  - Status progression: pending -> in_progress -> completed

**IMPORTANT**: Before building, first explore `~/.claude/teams/` and `~/.claude/tasks/` to check the exact file structure and JSON schemas. Read any existing files to confirm the data format. The schemas described above are approximate - verify them from actual files.

## Architecture

```
[File System Watcher] --> [Python HTTP Server] --> [Browser Dashboard]
     watches                  serves API +            auto-refreshes
  ~/.claude/teams/            static HTML              via polling
  ~/.claude/tasks/
```

### Backend (Python)

- Single Python script: `scripts/team_dashboard.py`
- Built-in `http.server` or lightweight framework (no Flask needed if keeping it simple)
- Endpoints:
  - `GET /` - serves the HTML dashboard
  - `GET /api/teams` - returns all team configs as JSON
  - `GET /api/tasks/{team-name}` - returns all tasks for a team as JSON
- Polls the filesystem every 1-2 seconds for changes
- Auto-opens browser on startup (`webbrowser.open`)
- Runs on `localhost:8099` (pick a port that doesn't conflict)

### Frontend (Embedded HTML/JS/CSS)

- Single HTML page with embedded CSS and JS (served by the Python script)
- **Dark theme** matching the existing pipeline report aesthetic (#1a1a2e background, etc.)
- Auto-polls `/api/tasks` every 2 seconds via `fetch()`
- No build step, no npm, no framework - vanilla HTML/CSS/JS

## UI Layout

```
+----------------------------------------------------------+
|  CLAUDE TEAM DASHBOARD          [team-name]   [auto-refresh] |
+----------------------------------------------------------+
|                                                          |
|  TEAMMATES                                               |
|  +----------+ +----------+ +----------+ +----------+     |
|  | Agent 1  | | Agent 2  | | Agent 3  | | Agent 4  |     |
|  | name     | | name     | | name     | | name     |     |
|  | type     | | type     | | type     | | type     |     |
|  | [status] | | [status] | | [status] | | [status] |     |
|  +----------+ +----------+ +----------+ +----------+     |
|                                                          |
|  TASKS                                                   |
|  +------------------------------------------------------+|
|  | # | Subject        | Owner   | Status      | Blocks ||
|  |---|----------------|---------|-------------|--------||
|  | 1 | Research auth  | agent-1 | completed   | 3, 4   ||
|  | 2 | Build backend  | agent-2 | in_progress | 5      ||
|  | 3 | Write tests    |         | pending     |        ||
|  | 4 | Build frontend | agent-3 | in_progress |        ||
|  | 5 | Integration    |         | pending     |        ||
|  +------------------------------------------------------+|
|                                                          |
|  DEPENDENCY GRAPH (optional/stretch)                     |
|  [visual node graph of task dependencies]                |
|                                                          |
+----------------------------------------------------------+
```

## Visual Design

- **Background**: Dark (#0d1117 or similar GitHub-dark)
- **Cards**: Slightly lighter panels with subtle borders
- **Status colors**:
  - Pending: gray/dim
  - In Progress: blue/cyan with pulse animation
  - Completed: green with checkmark
  - Blocked: orange/amber
- **Typography**: Monospace font (Consolas, JetBrains Mono, or system monospace)
- **Animations**: Subtle pulse on in_progress items, smooth transitions on status changes
- Task rows highlight when status changes (brief flash)

## Implementation Steps

1. **Explore data** - Read actual files in `~/.claude/teams/` and `~/.claude/tasks/` to confirm JSON schemas
2. **Build backend** - Python HTTP server that reads team/task files and serves JSON API
3. **Build frontend** - HTML dashboard with polling, dark theme, status indicators
4. **Wire up** - Embed the HTML in the Python script (or serve from adjacent file)
5. **Test** - Start a team session in one terminal, run dashboard in another, verify it updates live
6. **Polish** - Animations, dependency visualization, error handling for missing files

## Stretch Goals (only if time permits)

- Dependency graph visualization (simple SVG/canvas)
- Task timeline showing when each task started/completed
- Log viewer showing recent file changes
- Sound/notification on task completion
- Filter by status or owner

## How to Run

```bash
python scripts/team_dashboard.py
# Opens browser to http://localhost:8099
# Keep running while using agent teams in another terminal
```

## Use the Agent Team Feature to Build This

Spin up an agent team to build this in parallel:
- **1 agent**: Explore the `~/.claude/teams/` and `~/.claude/tasks/` file structure, document exact schemas
- **1 agent**: Build the Python backend server
- **1 agent**: Build the HTML/CSS/JS frontend dashboard
- Coordinate via shared task list, then integrate

Use `--teammate-mode in-process` with Shift+Up/Down to monitor the team while they build it (meta!).
