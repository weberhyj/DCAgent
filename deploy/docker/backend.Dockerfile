ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}

ARG DCAGENT_UID
ARG DCAGENT_GID
USER root

WORKDIR /app

COPY artifacts/wheels /wheels
COPY backend/requirements.txt backend/requirements-offline.txt ./
RUN python -m pip install --no-index --find-links=/wheels --require-hashes -r requirements-offline.txt \
    && rm -rf /root/.cache/pip

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
    && useradd --uid "$DCAGENT_UID" --gid "$DCAGENT_GID" --create-home --shell /usr/sbin/nologin dcagent \
    && test "$(id -u dcagent)" = "$DCAGENT_UID" \
    && test "$(id -g dcagent)" = "$DCAGENT_GID" \
    && chown -R dcagent:dcagent /app
USER dcagent

CMD ["python", "-m", "uvicorn", "app.main:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
