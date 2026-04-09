from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'logistic_regression_battery_state_model.joblib')

if not os.path.exists(model_path):
    print("❌ Model file not found:", model_path)
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Model load error:", str(e))
        model = None


# ✅ UI PAGE (BROWSER FRIENDLY)
@app.route('/')
def home():
    return render_template_string("""
    <html>
    <head>
        <title>Battery Health Predictor</title>
        <style>
            body { font-family: Arial; text-align: center; margin-top: 50px; }
            input { padding: 10px; margin: 5px; }
            button { padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h2>🔋 Battery Health Prediction</h2>
        <form action="/predict_form" method="post">
            <input type="number" step="any" name="voltage" placeholder="Voltage" required><br>
            <input type="number" step="any" name="current_percentage" placeholder="Current %" required><br>
            <input type="number" step="any" name="temperature" placeholder="Temperature" required><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """)


# ✅ FORM PREDICTION (NO ERROR IN BROWSER)
@app.route('/predict_form', methods=['POST'])
def predict_form():
    if model is None:
        return "<h3>❌ Model not loaded</h3>"

    try:
        voltage = float(request.form['voltage'])
        current = float(request.form['current_percentage'])
        temperature = float(request.form['temperature'])

        input_df = pd.DataFrame([{
            'voltage': voltage,
            'current_percentage': current,
            'temperature': temperature
        }])

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        result = f"""
        <h3>✅ Prediction: {prediction[0]}</h3>
        <p>Probabilities:</p>
        <ul>
        """

        for cls, prob in zip(model.classes_, prediction_proba[0]):
            result += f"<li>{cls}: {round(prob, 3)}</li>"

        result += "</ul><br><a href='/'>⬅ Back</a>"

        return result

    except Exception as e:
        return f"<h3>❌ Error: {str(e)}</h3>"


# ✅ API (FOR POSTMAN / FRONTEND)
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)

        input_df = pd.DataFrame([{
            'voltage': float(data['voltage']),
            'current_percentage': float(data['current_percentage']),
            'temperature': float(data['temperature'])
        }])

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


# 🔥 REQUIRED FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)
