#!/usr/bin/env bash
set -Eeuo pipefail

# Start the native Ubuntu chain in dependency order. The retrieval bootstrap is
# deliberately placed between llama.cpp and the API: a failed first build must
# leave the application start command untouched so an operator can fix the
# dependency or model configuration and rerun this script.

SUPERVISORCTL="${DCAGENT_SUPERVISORCTL:-supervisorctl}"
BOOTSTRAP_SCRIPT="${DCAGENT_RETRIEVAL_BOOTSTRAP:-}"
EMBEDDING_PROGRAM="${DCAGENT_EMBEDDING_PROGRAM:-dcagent-llama-embedding}"
RERANKER_PROGRAM="${DCAGENT_RERANKER_PROGRAM:-dcagent-llama-reranker}"
LLM_PROGRAM="${DCAGENT_LLM_PROGRAM:-dcagent-ollama-llm}"
API_PROGRAM="${DCAGENT_API_PROGRAM:-dcagent-api}"
WORKER_PROGRAM="${DCAGENT_WORKER_PROGRAM:-dcagent-structured-worker}"
INGESTION_PROGRAM="${DCAGENT_INGESTION_PROGRAM:-dcagent-ingestion-worker}"

if [[ -z "$BOOTSTRAP_SCRIPT" ]]; then
  SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_retrieval_index.sh"
fi

die() {
  echo "dcagent chain start failed: $*" >&2
  exit 1
}

command -v "$SUPERVISORCTL" >/dev/null 2>&1 || die "supervisorctl is unavailable"
[[ -x "$BOOTSTRAP_SCRIPT" ]] || die "retrieval bootstrap script is not executable: $BOOTSTRAP_SCRIPT"

echo "Starting llama.cpp Embedding/Reranker and Ollama LLM services..."
"$SUPERVISORCTL" start "$EMBEDDING_PROGRAM" "$RERANKER_PROGRAM" "$LLM_PROGRAM"

echo "Reconciling the Qdrant retrieval publication..."
"$BOOTSTRAP_SCRIPT"

echo "Starting API, knowledge ingestion worker, and structured worker..."
"$SUPERVISORCTL" start "$API_PROGRAM" "$INGESTION_PROGRAM" "$WORKER_PROGRAM"

echo "Supervisor status:"
"$SUPERVISORCTL" status \
  "$EMBEDDING_PROGRAM" "$RERANKER_PROGRAM" "$LLM_PROGRAM" \
  "$API_PROGRAM" "$INGESTION_PROGRAM" "$WORKER_PROGRAM"
