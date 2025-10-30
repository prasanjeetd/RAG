@echo off
py -3.12 -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
echo Done.
