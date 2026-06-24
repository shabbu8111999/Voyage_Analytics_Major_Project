
import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)

SAVED_DIR = ROOT / "models" / "saved"

# loading flight price prediction files
print("Loading flight price model...")
model          = joblib.load(SAVED_DIR / "model.pkl")
label_encoders = joblib.load(SAVED_DIR / "label_encoders.pkl")
scaler         = joblib.load(SAVED_DIR / "scaler.pkl")

# loading flight type classifier files
print("Loading flight type classifier...")
classifier_model    = joblib.load(SAVED_DIR / "classifier_model.pkl")
classifier_encoders = joblib.load(SAVED_DIR / "classifier_encoders.pkl")
classifier_features = joblib.load(SAVED_DIR / "classifier_features.pkl")

# loading hotel catalog
print("Loading hotel catalog...")
hotel_catalog = pd.read_csv(ROOT / "data" / "processed" / "hotel_catalog.csv")

print("All files loaded. Server is ready.\n")


# field definitions for flight price prediction
REQUIRED_FIELDS = [
    "flightType", "agency", "gender", "company",
    "time", "distance", "age",
    "month", "day_of_week", "is_weekend", "quarter"
]
TEXT_FIELDS    = ["flightType", "agency", "gender", "company"]
NUMERIC_FIELDS = ["time", "distance", "age", "month", "day_of_week"]
FEATURE_ORDER  = [
    "flightType", "time", "distance", "agency",
    "company", "gender", "age", "month",
    "day_of_week", "is_weekend", "quarter"
]


# home route — serves the main UI
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        flight_types       = list(label_encoders["flightType"].classes_),
        agencies           = list(label_encoders["agency"].classes_),
        genders            = list(label_encoders["gender"].classes_),
        companies_flight   = list(label_encoders["company"].classes_),
        companies_classify = list(classifier_encoders["company"].classes_),
    )


# health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"})


# flight price prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row      = {field: [data[field]] for field in REQUIRED_FIELDS}
    input_df = pd.DataFrame(row)

    for col in TEXT_FIELDS:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            valid = list(le.classes_)
            return jsonify({"error": f"Invalid value '{data[col]}' for '{col}'. Valid: {valid}"}), 400

    input_df[NUMERIC_FIELDS] = scaler.transform(input_df[NUMERIC_FIELDS])
    input_df = input_df[FEATURE_ORDER]

    predicted_price = model.predict(input_df)[0]

    return jsonify({
        "predicted_price": round(float(predicted_price), 2),
        "currency"       : "BRL",
        "status"         : "success"
    })


# flight type classification
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    input_data = {}
    for col in classifier_features:
        if col == "company_enc":
            le = classifier_encoders["company"]
            try:
                input_data[col] = [int(le.transform([data["company"]])[0])]
            except ValueError:
                return jsonify({"error": f"Invalid company: {data['company']}"}), 400
        else:
            try:
                input_data[col] = [float(data[col])]
            except (KeyError, TypeError, ValueError):
                return jsonify({"error": f"Missing or invalid field: {col}"}), 400

    input_df     = pd.DataFrame(input_data)[classifier_features]
    pred_encoded = classifier_model.predict(input_df)[0]
    pred_label   = classifier_encoders["flightType"].inverse_transform([pred_encoded])[0]

    return jsonify({
        "predicted_flight_type": pred_label,
        "status"               : "success"
    })


# hotel recommendation
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    budget = float(data.get("budget_per_night", 0))
    days   = int(data.get("num_days", 1))

    affordable = hotel_catalog[hotel_catalog["price_per_night"] <= budget].copy()

    if affordable.empty:
        return jsonify({"error": f"No hotels found within {budget} BRL per night"}), 404

    affordable["estimated_total"] = (affordable["price_per_night"] * days).round(2)
    affordable = affordable.sort_values("total_bookings", ascending=False)
    results    = affordable[["name", "place", "price_per_night",
                              "estimated_total", "total_bookings"]].head(3)

    return jsonify({
        "recommendations": results.to_dict("records"),
        "budget"         : budget,
        "days"           : days,
        "status"         : "success"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open browser → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)