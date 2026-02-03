import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Any

from .schemas import PredictRequest, PredictResponse, ModelInfo
from .model_loader import model_wrapper
from .utils import compute_uncertainty, explain_prediction
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Student Performance Prediction API")
logger = logging.getLogger("uvicorn.error")

# allow local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    try:
        model_wrapper.load()
        logger.info("Loaded model")
    except Exception as e:
        logger.error("Model load failed: %s", e)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    if not model_wrapper.meta:
        raise HTTPException(status_code=404, detail="No model metadata found")
    return model_wrapper.meta


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model_wrapper.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    meta = model_wrapper.meta
    expected_features = meta.get('features')
    payload = req.features

    # Validate input has required features
    if expected_features:
        missing = [f for f in expected_features if f not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Build DataFrame in correct feature order
    X_df = pd.DataFrame([ [payload.get(f) for f in expected_features] ], columns=expected_features)

    try:
        pred = float(model_wrapper.model.predict(X_df)[0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    uncertainty = compute_uncertainty(model_wrapper.model, X_df)
    top_feats = explain_prediction(model_wrapper.model, X_df, top_k=5)

    # build response
    top_feats_resp = []
    for f, val, contrib in top_feats:
        top_feats_resp.append({"feature": f, "value": val, "contribution": contrib})

    return {"prediction": pred, "uncertainty": uncertainty, "top_features": top_feats_resp}
