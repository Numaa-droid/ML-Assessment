FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
# xgboost-cpu is the same pinned version and module without the ~470MB of
# NVIDIA GPU libraries the Linux xgboost wheel pulls in.
COPY requirements.txt .
RUN sed 's/^xgboost==/xgboost-cpu==/' requirements.txt > requirements-docker.txt \
    && pip install -r requirements-docker.txt

COPY churn/ churn/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
