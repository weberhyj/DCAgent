@echo off
cd /d "%~dp0.."
set VITE_API_PROXY_TARGET=http://127.0.0.1:8015
pnpm.cmd --filter dc-agent-admin-frontend dev -- --port 5178
