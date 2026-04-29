# AURA — multi-stage friendly single image; run API and/or Streamlit via command override.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Dataset + models: mount volumes at runtime (see docker-compose.yml)
RUN mkdir -p /app/logs /app/models

EXPOSE 8000 8501

# Default: FastAPI (override CMD for Streamlit)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
