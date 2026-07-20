ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}

ARG DCAGENT_UID
ARG DCAGENT_GID
USER root

WORKDIR /app

COPY artifacts/wheels /wheels
COPY backend/pyproject.toml backend/uv.lock ./
ENV UV_NO_INDEX=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy
RUN uv --version && uv sync --frozen --offline --no-install-project --no-dev --group offline --find-links=/wheels \
    && rm -rf /root/.cache/uv
ENV PATH="/app/.venv/bin:$PATH"

COPY backend/app ./app
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic

RUN case "$DCAGENT_UID" in ''|*[!0-9]*) exit 1 ;; esac \
    && case "$DCAGENT_GID" in ''|*[!0-9]*) exit 1 ;; esac \
    && test "$DCAGENT_UID" -ge 1 \
    && test "$DCAGENT_UID" -le 2147483647 \
    && test "$DCAGENT_GID" -ge 1 \
    && test "$DCAGENT_GID" -le 2147483647 \
    && groupadd --gid "$DCAGENT_GID" dcagent \
    && useradd --uid "$DCAGENT_UID" --gid "$DCAGENT_GID" --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin dcagent \
    && test "$(id -u dcagent)" = "$DCAGENT_UID" \
    && test "$(id -g dcagent)" = "$DCAGENT_GID" \
    && install -d -o dcagent -g dcagent /app/uploads/knowledge
ENV HOME=/nonexistent
USER dcagent

CMD ["python", "-m", "app.ingestion_worker"]
