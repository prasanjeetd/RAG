@echo off
call .venv\Scripts\activate
python src\ingest.py --data .\data
