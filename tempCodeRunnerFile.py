from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ✅ Load Level-1 models
ada = joblib.load("models/AdaBoost_model.pkl")
gb = joblib.load("models/GradientBoosting_model.pkl")
knn = joblib.load("models/KNN_model.pkl")
lasso = joblib.load("models/Lasso_model.pkl")
rf = joblib.load("models/RandomForest_model.pkl")
svr = joblib.load("models/SVR_model.pkl")

# ✅ Load Level-2 Meta Model (Decision Tree)
meta_model = joblib.load("models/meta_decision_tree.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON input"}), 400

        # ✅ Expected input fields
        required_fields = [
            "Airline", "Departure_Airport", "Arrival_Airport",
            "Scheduled_Dep_Hour", "Scheduled_Arr_Hour",
            "Day_of_Week", "Month", "Holiday_Season", "Weather_Condition",
            "Temperature", "Wind_Speed", "Visibility", "Rain_mm",
            "Departure_Traffic", "Arrival_Traffic"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # ✅ Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        # ✅ Level-1 Predictions (Base Models)
        level1_features = np.column_stack([
            ada.predict(input_df),
            gb.predict(input_df),
            knn.predict(input_df),
            lasso.predict(input_df),
            rf.predict(input_df),
            svr.predict(input_df)
        ])

        # ✅ Level-2 Final Prediction
        final_pred = meta_model.predict(level1_features)[0]

        return jsonify({
            "final_delay_prediction": float(final_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
