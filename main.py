
from typing import Dict, Optional
from contextlib import asynccontextmanager
from utils import get_prediction_confidence,load_models
from fastapi import FastAPI, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import numpy as np
import joblib
import os 


# Global variables for models
vectorizer = None
metrics = None
models = {}


# ---------- Pydantic Models ----------
class RequestModel(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

    @field_validator('text')
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

class PredictionResult(BaseModel):
    prediction: str
    confidence: Optional[float]
    metrics_on_test_data: Dict[str, float]

class ResponseModel(BaseModel):
    input_text: str
    model_results: Dict[str, PredictionResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models,vectorizer,metrics
    models, vectorizer, metrics = load_models()
    yield

# ---------- FastAPI Setup ----------


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="Simple sentiment analysis using multiple ML models",
    lifespan=lifespan
)



# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }

@app.post("/predict_all", response_model=ResponseModel)
def predict_all_models(data: RequestModel):
    X = vectorizer.transform([data.text])
    results = {}

    for name, model in models.items():
        try:
            pred = model.predict(X)[0]
            label = "positive" if pred == 1 else "negative"
            confidence = get_prediction_confidence(model, X)

            results[name] = PredictionResult(
                prediction=label,
                confidence=confidence,
                metrics_on_test_data={
                    "accuracy": round(float(metrics[name]['accuracy']),2),
                    "f1_score": round(float(metrics[name]['f1_score']),2)
                }
            )
        except:
            continue

    if not results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No model was able to make a prediction"
        )

    return ResponseModel(input_text=data.text, model_results=results)

@app.post("/predict/{model_name}")
def predict_single_model(model_name: str, data: RequestModel):
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )

    try:
        X = vectorizer.transform([data.text])
        model = models[model_name]
        pred = model.predict(X)[0]
        label = "positive" if pred == 1 else "negative"
        confidence = get_prediction_confidence(model, X)

        return {
            "input_text": data.text,
            "model_name": model_name,
            "prediction": label,
            "confidence": confidence,
            "metrics_on_test_data": {
                "accuracy": round(float(metrics[model_name]['accuracy']),2),
                "f1_score": round(float(metrics[model_name]['f1_score']),2)
            }
        }
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction with {model_name}"
        )

