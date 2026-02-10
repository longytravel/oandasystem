@echo off
cd /d C:\Trading\oandasystem
python scripts/run_live.py --strategy rsi_v3 --pair GBP_USD --timeframe H1 --params-file instances\rsi_v3_GBP_USD_H1\config.json --instance-dir instances\rsi_v3_GBP_USD_H1 --instance-id rsi_v3_GBP_USD_H1 --yes
pause
