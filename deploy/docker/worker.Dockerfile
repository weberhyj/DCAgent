# syntax=docker/dockerfile:1
ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

COPY artifacts/wheels /wheels
COPY backend/requirements.txt backend/requirements-offline.txt ./
RUN python -m pip install --no-index --find-links=/wheels --require-hashes -r requirements-offline.txt \
    && rm -rf /root/.cache/pip

COPY backend/app ./app
COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic

RUN useradd --create-home --shell /usr/sbin/nologin dcagent \
    && chown -R dcagent:dcagent /app
USER dcagent

CMD ["python", "-m", "app.ingestion_worker"]
