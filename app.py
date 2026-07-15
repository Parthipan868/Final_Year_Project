import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

FEATURE_COLUMNS = [
    "reading_accuracy",
    "word_omission_count",
    "pronunciation_error_count",
    "reading_speed_wpm",
]

model = CatBoostClassifier()
model.load_model("catboost_dyslexia_model.cbm")

label_encoder = joblib.load("risk_label_encoder.pkl")


def get_recommendation(risk_label: str) -> str:
    if risk_label == "Low":
        return (
            "Low risk detected. Continue regular reading practice "
            "and routine monitoring."
        )

    if risk_label == "Medium":
        return (
            "Some reading difficulties were observed. Provide guided "
            "reading support and repeat the assessment later."
        )

    return (
        "High dyslexia risk detected. A professional educational "
        "assessment is recommended. EEG screening may be considered "
        "as the next stage where clinically appropriate."
    )


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No feature data was received."
            }), 400

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": f"Missing features: {missing_features}"
            }), 400

        feature_values = {
            feature: float(data[feature])
            for feature in FEATURE_COLUMNS
        }

        input_data = pd.DataFrame(
            [feature_values],
            columns=FEATURE_COLUMNS,
        )

        encoded_prediction = int(
            np.asarray(
                model.predict(input_data)
            ).reshape(-1)[0]
        )

        risk_label = label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]

        probabilities = model.predict_proba(input_data)[0]

        probability_result = {
            label_encoder.inverse_transform([index])[0]:
                round(float(probability) * 100, 2)
            for index, probability in enumerate(probabilities)
        }

        return jsonify({
            "risk_label": risk_label,
            "probabilities": probability_result,
            "recommendation": get_recommendation(risk_label),
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:
        return jsonify({
            "error": f"Prediction failed: {error}"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )