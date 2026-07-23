# Backend Runtime Dependencies Design

## Goal

Allow browser clients from any origin, add the production process and access-log runtime dependencies, and keep development-only tooling separated from application dependencies.

## Design

The shared FastAPI builder in `backend/app/main.py` will configure `CORSMiddleware` with `allow_origins=["*"]` while preserving the existing credentials, methods, and headers settings. The same builder will call `asynctor.contrib.fastapi.config_access_log(app)` so development, test, and production app instances receive identical access-log configuration.

`gunicorn` and `asynctor` will be regular project dependencies because deployed application processes import or execute them. `fastapi-cli` will be added to the dev dependency group. Ruff configuration will remain in `pyproject.toml`, but the Ruff package itself will be removed from the dev group so developers install it independently with `uv tool install ruff`.

## Verification

Tests will inspect the FastAPI middleware configuration, mock the access-log integration to prove it receives the created app, and parse `pyproject.toml` to enforce dependency placement. The lockfile will be regenerated with uv, then focused tests, the backend suite, and global Ruff checks will run.
