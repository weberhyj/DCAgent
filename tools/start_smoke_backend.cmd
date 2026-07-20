@echo off
cd /d "%~dp0..\backend"
set DATABASE_URL=sqlite+pysqlite:///:memory:
set LLM_PROVIDER=template
uv run --project . --group dev python -m uvicorn app.main:app --host 127.0.0.1 --port 8015
