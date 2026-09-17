"""
Credit Card Fraud Detection — Inference API
=============================================
Serves the trained scikit-learn pipeline (RobustScaler preprocessing +
tuned RandomForestClassifier) saved in model/Model_Pipeline.pkl as a
REST API built on FastAPI.

The pipeline was trained on the Kaggle mlg-ulb/creditcardfraud dataset:
Time, V1-V28 (PCA-anonymized), and Amount as inputs; Class (0 = legit,
1 = fraud) as the target. See PROCESS.md and README.md for the full
methodology and evaluation numbers.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# Resolved relative to this file (not the process cwd) so the app finds the
# model and frontend files regardless of where uvicorn is launched from — a
# real bug in an earlier project of mine came from exactly the opposite
# assumption.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "Model_Pipeline.pkl"

# Exact column order the pipeline's ColumnTransformer was fit on.
FEATURE_ORDER: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

_pipeline = None  # set during lifespan startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    _pipeline = joblib.load(MODEL_PATH)
    yield
    _pipeline = None


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Scores a credit card transaction's fraud probability using a "
        "RandomForestClassifier trained on the Kaggle mlg-ulb/creditcardfraud "
        "dataset. See /docs for interactive testing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# Illustrative values near the dataset's column means — NOT a real transaction.
_EXAMPLE_ROW = {
    "Time": 94813.86, "V1": -0.0057, "V2": 0.0033, "V3": 0.0040, "V4": 0.0007,
    "V5": -0.0025, "V6": 0.0017, "V7": -0.0006, "V8": -0.0002, "V9": -0.0009,
    "V10": -0.0012, "V11": 0.0016, "V12": 0.0002, "V13": 0.0001, "V14": 0.0002,
    "V15": 0.0003, "V16": -0.0004, "V17": -0.0002, "V18": 0.0001, "V19": 0.0004,
    "V20": 0.0001, "V21": 0.0001, "V22": 0.0002, "V23": 0.0000, "V24": 0.0001,
    "V25": 0.0001, "V26": -0.0001, "V27": 0.0000, "V28": 0.0001, "Amount": 88.35,
}


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")

    model_config = {"json_schema_extra": {"example": _EXAMPLE_ROW}}


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = legitimate, 1 = fraudulent")
    label: str
    fraud_probability: float = Field(..., description="Model's predicted probability of fraud, 0-1")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """Serves the frontend. The JSON status check lives at /health."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css", include_in_schema=False)
def styles():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/script.js", include_in_schema=False)
def script():
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok" if _pipeline is not None else "unavailable", model_loaded=_pipeline is not None)


def _predict_one(row: dict) -> PredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    frame = pd.DataFrame([row], columns=FEATURE_ORDER)
    try:
        pred = int(_pipeline.predict(frame)[0])
        proba = float(_pipeline.predict_proba(frame)[0][1])
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        prediction=pred,
        label="Fraudulent" if pred == 1 else "Legitimate",
        fraud_probability=round(proba, 6),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(transaction: Transaction):
    """Score a single transaction."""
    return _predict_one(transaction.model_dump())


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["inference"])
def predict_batch(transactions: List[Transaction]):
    """Score a batch of transactions in one call."""
    if not transactions:
        raise HTTPException(status_code=400, detail="Provide at least one transaction")
    return [_predict_one(t.model_dump()) for t in transactions]
