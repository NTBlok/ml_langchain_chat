import requests
import pandas as pd

def fetch_data(api_url: str, params: dict = None) -> pd.DataFrame:
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()
    # This will be project-specific: convert JSON to DataFrame
    return pd.json_normalize(data)
