import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def predict_price(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

