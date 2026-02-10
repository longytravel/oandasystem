@echo off
cd /d C:\Trading\oandasystem
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --instance-dir instances\rsi_v3_GBP_USD_M15 --instance-id rsi_v3_GBP_USD_M15 --yes
pause
