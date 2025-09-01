import joblib

def train_model(X, y, model_cls, model_params: dict, save_path: str):
    model = model_cls(**model_params)
    model.fit(X, y)
    joblib.dump(model, save_path)
    return model
