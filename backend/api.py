from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model_infer import load_model, predict
import os

app = FastAPI()

# Try to load the model, use None if not found
try:
    model = load_model("models/model.pkl")
    print("Loaded pre-trained model")
except FileNotFoundError:
    print("No pre-trained model found. Using None.")
    model = None

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first by making a POST request to /train"
        )
    X = pd.DataFrame([req.features])
    preds = predict(model, X)
    return {"predictions": preds.tolist()}

@app.post("/train")
def train_model():
    try:
        # Import here to avoid circular imports
        from train_model import train_model as train
        train()
        # Reload the model after training
        global model
        model = load_model("models/model.pkl")
        return {"status": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
