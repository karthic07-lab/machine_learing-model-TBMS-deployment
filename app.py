from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model safely
model_path = 'logistic_regression_battery_state_model.joblib'

if not os.path.exists(model_path):
    print("❌ Model file not found!")
    model = None
else:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")

@app.route('/')
def home():
    return "Battery Prediction API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)

        expected_features = ['voltage', 'current_percentage', 'temperature']

        # Validate input
        if not all(feature in data for feature in expected_features):
            return jsonify({
                'error': f'Missing features. Expected: {expected_features}'
            }), 400

        # Create DataFrame
        input_df = pd.DataFrame([{
            'voltage': data['voltage'],
            'current_percentage': data['current_percentage'],
            'temperature': data['temperature']
        }])

        # Use pipeline directly
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        result = {
            'prediction': str(prediction[0]),
            'probabilities': {
                str(cls): float(prob)
                for cls, prob in zip(model.classes_, prediction_proba[0])
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500        return jsonify({'error': str(e)}), 500


