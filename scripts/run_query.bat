@echo off
call .venv\Scripts\activate
python src\query.py --q "%*"
