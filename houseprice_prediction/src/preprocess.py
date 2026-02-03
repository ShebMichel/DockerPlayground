import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent  
DATA_PATH = BASE_DIR / "data" / "housing.csv"

def preprocess_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    X = df.drop("price", axis=1)
    y = df["price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
