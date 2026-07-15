ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

COPY artifacts/wheels /wheels
COPY backend/requirements.txt backend/requirements-offline.txt ./
RUN python -m pip install --no-index --find-links=/wheels --require-hashes -r requirements-offline.txt \
    && rm -rf /root/.cache/pip

COPY backend/app ./app

RUN useradd --create-home --shell /usr/sbin/nologin dcagent \
    && chown -R dcagent:dcagent /app
USER dcagent

CMD ["python", "-m", "uvicorn", "app.embedding_service:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8081"]
