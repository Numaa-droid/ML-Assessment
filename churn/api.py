"""FastAPI inference service.

Run from the project root:  uvicorn churn.api:app --port 8000
Interactive docs:           http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from churn import config
from churn.predict import ChurnPredictor
from churn.schemas import CustomerProfile, HealthResponse, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup and fail fast if the artifact is missing.
    model_path = Path(os.environ.get("CHURN_MODEL_PATH", config.MODEL_PATH))
    app.state.predictor = ChurnPredictor.load(model_path)
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Scores a single raw customer profile and returns the churn prediction and probability.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor: ChurnPredictor = request.app.state.predictor
    return HealthResponse(
        status="ok", model_name=predictor.model_name, model_version=predictor.model_version
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerProfile, request: Request) -> PredictionResponse:
    # Sync handler: FastAPI runs it in a threadpool, so CPU-bound inference
    # does not block the event loop.
    return request.app.state.predictor.predict(customer)
