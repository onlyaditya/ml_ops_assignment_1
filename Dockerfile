# Serving image for the heart-disease API: model + FastAPI, nothing else.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements-serve.txt changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY models/model.joblib models/model_metadata.json models/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# slim image has no curl, so probe with Python's standard library
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
