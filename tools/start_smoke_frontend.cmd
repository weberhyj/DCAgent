@echo off
cd /d "%~dp0..\frontend"
set VITE_API_PROXY_TARGET=http://127.0.0.1:8015
npm.cmd run dev -- --port 5177
