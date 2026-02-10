@echo off
cd /d C:\Trading\oandasystem
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_H1_20260206_151217 --instance-dir instances\rsi_v3_GBP_USD_H1 --instance-id rsi_v3_GBP_USD_H1 --yes
pause
