FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
COPY deploy/requirements-live.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for live trading
COPY config/ config/
COPY live/ live/
COPY strategies/ strategies/
COPY optimization/numba_backtest.py optimization/numba_backtest.py
COPY scripts/run_live.py scripts/run_live.py
COPY scripts/run_monitor.py scripts/run_monitor.py

# Create .env placeholder (actual credentials via env_file in compose)
RUN touch .env

ENTRYPOINT ["python", "scripts/run_live.py"]
