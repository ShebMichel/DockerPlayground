import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = preprocess_data("data/housing.csv")

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save model
joblib.dump(model, "models/model.pkl")

# Save metrics
metrics = {
    "rmse": rmse,
    "r2": r2
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model trained")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Save prediction to csv
results = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})

results.to_csv("models/predictions.csv", index=False)
