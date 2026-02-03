from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
    # Accept arbitrary features; validate keys in endpoint code
    features: Dict[str, Optional[float]] = Field(..., description="Feature name -> value")


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictResponse(BaseModel):
    prediction: float
    uncertainty: Optional[float]
    top_features: List[FeatureContribution]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    best_algo: str
    metrics: Dict[str, float]
    features: List[str]
