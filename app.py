from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
# Load the full trained pipeline
loaded_model_pipeline = joblib.load('logistic_regression_battery_state_model.joblib')

# Extract the preprocessor from the pipeline
preprocessor = loaded_model_pipeline.named_steps['preprocessor']

# The StandardScaler is the 'num' transformer within the preprocessor
# We need to get the fitted transformer, which is stored in the `named_transformers_` attribute
scaler = preprocessor.named_transformers_['num']

# Save the extracted scaler
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)

print(f"Fitted StandardScaler successfully exported to {scaler_filename}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    data = request.get_json(force=True)

    # Expected input features
    expected_features = ['voltage', 'current_percentage', 'temperature']

    # Validate input
    if not all(feature in data for feature in expected_features):
        return jsonify({
            'error': f'Missing one or more required features. Expected: {expected_features}'
        }), 400

    try:
        # Convert input data into DataFrame
        input_df = pd.DataFrame([{
            'voltage': data['voltage'],
            'current_percentage': data['current_percentage'],
            'temperature': data['temperature']
        }])

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        predicted_class = prediction[0]

        # Convert probabilities to normal Python types (important!)
        class_probabilities = {
            str(cls): float(prob)
            for cls, prob in zip(model.classes_, prediction_proba[0])
        }

        return jsonify({
            'prediction': str(predicted_class),
            'probabilities': class_probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
