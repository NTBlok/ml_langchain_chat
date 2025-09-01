import joblib

def load_model(model_path: str):
    return joblib.load(model_path)

def predict(model, X):
    return model.predict(X)
