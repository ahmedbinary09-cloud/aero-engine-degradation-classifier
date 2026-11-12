FROM python:3.14-slim-bookworm

# xgboost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

# uv.lock* = copied if present (created by `uv sync` on your machine)
COPY ".python-version" "pyproject.toml" "uv.lock*" "./"
RUN uv sync --no-install-project --no-dev

COPY "predict.py" "health.py" "model.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
