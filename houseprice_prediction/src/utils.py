import json
import pandas as pd

def load_metrics():
    with open("models/metrics.json", "r") as f:
        return json.load(f)

def load_predictions():
    return pd.read_csv("models/predictions.csv")


