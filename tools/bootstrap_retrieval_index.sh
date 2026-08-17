#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${DCAGENT_PYTHON:-/srv/dcagent/venv/bin/python}"
ENV_FILE="${DCAGENT_ENV_FILE:-/etc/dc-agent/dcagent.env}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "retrieval bootstrap Python runtime is unavailable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

cd "$REPO_ROOT/backend"
exec "$PYTHON_BIN" -m app.retrieval_bootstrap "$@"
