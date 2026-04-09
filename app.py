from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Absolute path fix (prevents path issues in Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'logistic_regression_battery_state_model.joblib')

# Load model safely
if not os.path.exists(model_path):
    print("❌ Model file not found at:", model_path)
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", str(e))
        model = None


# Home route
@app.route('/')
def home():
    return "API is running 🚀"


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)

        expected_features = ['voltage', 'current_percentage', 'temperature']

        # Validate input
        if not all(feature in data for feature in expected_features):
            return jsonify({'error': 'Missing input data'}), 400

        # Create DataFrame
        input_df = pd.DataFrame([{
            'voltage': float(data['voltage']),
            'current_percentage': float(data['current_percentage']),
            'temperature': float(data['temperature'])
        }])

        # Prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        return jsonify({
            'prediction': str(prediction[0]),
            'probabilities': {
                str(cls): float(prob)
                for cls, prob in zip(model.classes_, prediction_proba[0])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 🔥 CRITICAL FIX FOR RENDER (THIS WAS MISSING)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting app on port {port}")
    app.run(host="0.0.0.0", port=port)
