@echo off
cd /d C:\Trading\oandasystem
python scripts/run_live.py --strategy rsi_v3 --pair GBP_USD --timeframe M15 --params-file instances\rsi_v3_GBP_USD_M15\config.json --instance-dir instances\rsi_v3_GBP_USD_M15 --instance-id rsi_v3_GBP_USD_M15 --yes
pause
