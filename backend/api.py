from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model_infer import load_model, predict

app = FastAPI()
model = load_model("model.pkl")

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    X = pd.DataFrame([req.features])
    preds = predict(model, X)
    return {"predictions": preds.tolist()}
