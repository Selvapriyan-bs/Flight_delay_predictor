from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model pipeline
model_pipeline = joblib.load("models/RandomForest_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ✅ Normalize user inputs
    data = {k: (v.strip() if isinstance(v, str) else v) for k, v in data.items()}

    # ✅ Mapping for categorical standardization
    category_mappings = {
        "Airline": {
            "air india": "Air India",
            "indigo": "IndiGo",
            "spicejet": "SpiceJet",
            "vistara": "Vistara",
            "goair": "GoAir",
            "go air": "GoAir"
        }
    }

    # ✅ Normalize airline input
    if "Airline" in data:
        key = data["Airline"].lower()
        if key in category_mappings["Airline"]:
            data["Airline"] = category_mappings["Airline"][key]
        else:
            return jsonify({"error": f"Invalid Airline: {data['Airline']}"}), 400

    # ✅ Valid categories (update based on your training data)
    valid_airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoAir"]
    valid_weather = ["Clear", "Rain", "Fog", "Storm", "Thunderstorm"]
    valid_traffic = ["Low", "Medium", "High"]
    valid_airports = ["DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "GOI"]

    # ✅ Validation checks
    if data["Airline"] not in valid_airlines:
        return jsonify({"error": "Invalid Airline"}), 400

    if data["Weather_Condition"] not in valid_weather:
        return jsonify({"error": "Invalid Weather_Condition"}), 400

    if data["Departure_Airport"] not in valid_airports:
        return jsonify({"error": "Invalid Departure_Airport"}), 400

    if data["Arrival_Airport"] not in valid_airports:
        return jsonify({"error": "Invalid Arrival_Airport"}), 400

    if data["Departure_Traffic"] not in valid_traffic:
        return jsonify({"error": "Invalid Departure_Traffic"}), 400

    if data["Arrival_Traffic"] not in valid_traffic:
        return jsonify({"error": "Invalid Arrival_Traffic"}), 400

    # ✅ Convert Boolean / numeric fields
    data["Holiday_Season"] = 1 if data["Holiday_Season"] == "Yes" else 0

    numeric_columns = [
        "Scheduled_Dep_Hour", "Scheduled_Arr_Hour", "Day_of_Week", "Month",
        "Temperature", "Wind_Speed", "Visibility", "Rain_mm"
    ]
    for col in numeric_columns:
        data[col] = float(data[col])

    # ✅ Convert to DataFrame
    input_df = pd.DataFrame([data])

    # ✅ Predict
    prediction = model_pipeline.predict(input_df)[0]

    return jsonify({
    "predicted_delay_minutes": round(float(prediction), 2)
})

if __name__ == '__main__':
    app.run(debug=True)
