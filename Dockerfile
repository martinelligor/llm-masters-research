FROM --platform=linux/arm64 python:3.11-slim-bullseye

ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=TRUE \
    PIP_NO_CACHE_DIR=TRUE \
    PIP_TIMEOUT=300 \
    PIP_RETRIES=5 \
    DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
    --no-install-recommends \
    python3-venv \
    git && \
    apt auto-clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/poetry && \
    /opt/poetry/bin/pip install poetry

RUN mkdir /src
COPY pyproject.toml /src/
COPY poetry.lock /src/
COPY README.md /src/
COPY llm_api/ /src/llm_api/
COPY data/ /src/data/

RUN cd /src && \
    /opt/poetry/bin/poetry install --only main --no-root && \
    pip install . && \
    python -m nltk.downloader -d /app/data/nltk_data punkt_tab stopwords && \
    rm -rf /root/.cache

EXPOSE 8099

CMD [ "python", "-m", "uvicorn", "llm_api.server.app:app", "--reload", "--host", "0.0.0.0", "--port", "8099", "--timeout-keep-alive", "240" ]