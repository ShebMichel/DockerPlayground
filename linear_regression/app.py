# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import logging
# # Initialize Flask App
# app = Flask(__name__)
# # Configure Logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# # Load Model
# MODEL_PATH = "model.pkl"
# try:
#     model = joblib.load(MODEL_PATH)
#     logging.info(f"✅ Model loaded successfully from {MODEL_PATH}")
# except Exception as e:
#     logging.error(f"❌ Error loading model: {e}")
#     model = None

# @app.route("/", methods=["GET"])
# def home():
#     """Root endpoint - API Welcome Message."""
#     return jsonify({
#         "message": "🚀 Welcome to the ML Prediction API!",
#         "endpoints": {
#             "Predict": "/predict (POST)"
#         }
#     })

# @app.route("/predict", methods=["POST"])
# def predict():
#     """Endpoint to make predictions using the trained model."""
#     if model is None:
#         logging.error("Prediction request failed: Model not loaded.")
#         return jsonify({"error": "Model not loaded"}), 500
#     try:
#         # Get JSON data from request
#         data = request.json.get("input")
#         if not data:
#             logging.warning("Prediction request failed: Missing 'input' data.")
#             return jsonify({"error": "Missing 'input' data"}), 400
#         # Convert input to NumPy array and make prediction
#         prediction = model.predict(np.array(data).reshape(-1, 1))
#         logging.info(f"Prediction successful: {prediction.tolist()}")
#         return jsonify({"prediction": prediction.tolist()})
#     except Exception as e:
#         logging.error(f"Prediction error: {e}")
#         return jsonify({"error": "Invalid input format"}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import logging

# Initialize Flask App
app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Model
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# Simple HTML template for frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Prediction App</title>
</head>
<body>
    <h1>🚀 ML Prediction WebApp</h1>
    <p>Enter numbers separated by commas (e.g., 1, 2, 3) and click Predict:</p>
    <input type="text" id="inputData" size="50">
    <button onclick="makePrediction()">Predict</button>
    <h3>Prediction:</h3>
    <pre id="result"></pre>

    <script>
        async function makePrediction() {
            const input = document.getElementById('inputData').value;
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input: input.split(',').map(x => parseFloat(x.trim()))})
            });
            const data = await response.json();
            document.getElementById('result').innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """Serve HTML frontend for testing predictions."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to make predictions using the trained model."""
    if model is None:
        logging.error("Prediction request failed: Model not loaded.")
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json.get("input")
        if data is None or len(data) == 0:
            return jsonify({"error": "Missing or empty 'input' data"}), 400

        try:
            input_array = np.array(data, dtype=float).reshape(-1, 1)
        except ValueError:
            return jsonify({"error": "All input values must be numeric"}), 400

        prediction = model.predict(input_array)
        logging.info(f"Prediction successful: {prediction.tolist()}")
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Something went wrong during prediction"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
