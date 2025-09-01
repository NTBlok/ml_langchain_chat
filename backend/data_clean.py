import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder cleaning - override per project
    df = df.dropna()
    return df
