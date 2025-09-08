import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    # Create a simple dataset
    data = {
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 3, 4, 5, 6, 7],
        'target': [0, 0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    # Split features and target
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    return model_path

if __name__ == '__main__':
    train_model()
