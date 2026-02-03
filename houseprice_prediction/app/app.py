import streamlit as st
import matplotlib.pyplot as plt
from src.predict import predict_price
from src.utils import load_metrics, load_predictions

st.set_page_config(page_title="ML Model Deployment", layout="wide")

st.title("🏠 House Price Prediction")

# -----------------------------
# Sidebar - Model Metrics
# -----------------------------
st.sidebar.header("📊 Model Performance")

metrics = load_metrics()
st.sidebar.metric("RMSE", f"{metrics['rmse']:.2f}")
st.sidebar.metric("R² Score", f"{metrics['r2']:.2f}")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("🔢 Input Features")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area (sq ft)", min_value=100)

with col2:
    rooms = st.number_input("Number of rooms", min_value=1)

with col3:
    age = st.number_input("House age (years)", min_value=0)

if st.button("Predict Price"):
    price = predict_price([area, rooms, age])
    st.success(f"💰 Estimated Price: ${price:,.2f}")

# -----------------------------
# Visualization Section
# -----------------------------
st.subheader("📈 Model Predictions vs Actual")

df = load_predictions()

fig, ax = plt.subplots()
ax.scatter(df["actual"], df["predicted"], alpha=0.6)
ax.plot(
    [df["actual"].min(), df["actual"].max()],
    [df["actual"].min(), df["actual"].max()],
)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Prices")

st.pyplot(fig)
