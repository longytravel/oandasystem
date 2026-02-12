@echo off
cd /d C:\Trading\oandasystem
python scripts/run_live.py --strategy rsi_v1 --pair USD_CHF --timeframe H1 --params-file instances\rsi_v1_USD_CHF_H1\config.json --instance-dir instances\rsi_v1_USD_CHF_H1 --instance-id rsi_v1_USD_CHF_H1 --yes
pause
