"""
Diabetes prediction web app.

Loads the trained pipeline from diabetes_model.pkl (produced by diabetes_v2.py)
and serves a simple form-based UI. Empty form fields become NaN and are
handled automatically by the pipeline's imputer.

Local dev  : python app.py                   (http://localhost:5000)
Production : gunicorn -w 2 -b 0.0.0.0:8000 app:app
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request


MODEL_PATH = Path("diabetes_model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Train the model first: python diabetes_v2.py"
    )

artifact = joblib.load(MODEL_PATH)
PIPELINE      = artifact["pipeline"]
THRESHOLD     = artifact["threshold"]
FEATURE_ORDER = artifact["feature_order"]
MODEL_NAME    = artifact["model_name"]
TEST_METRICS  = artifact["test_metrics"]

print(f"[startup] {MODEL_NAME} | threshold={THRESHOLD} | features={FEATURE_ORDER}")

app = Flask(__name__)

# Form metadata: (label, placeholder, min, max, step) per feature
FEATURE_META = {
    "Pregnancies":              ("Pregnancies",              "e.g. 3",     0,   17,  1),
    "Glucose":                  ("Glucose (mg/dL)",          "e.g. 120",   0,   250, 1),
    "BloodPressure":            ("Blood Pressure (mm Hg)",   "e.g. 70",    0,   140, 1),
    "SkinThickness":            ("Skin Thickness (mm)",      "e.g. 25",    0,   100, 1),
    "BMI":                      ("BMI",                      "e.g. 28.5",  0,   70,  0.1),
    "DiabetesPedigreeFunction": ("Diabetes Pedigree",        "e.g. 0.5",   0,   3,   0.01),
    "Age":                      ("Age (years)",              "e.g. 35",    0,   120, 1),
}


def parse_input(form):
    """Form → single-row DataFrame in the correct column order.
    Blanks become NaN so the pipeline's imputer fills them with training median."""
    row = {}
    for feat in FEATURE_ORDER:
        raw = form.get(feat, "").strip()
        row[feat] = float(raw) if raw else np.nan
    return pd.DataFrame([row], columns=FEATURE_ORDER)


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    submitted = {f: "" for f in FEATURE_ORDER}

    if request.method == "POST":
        try:
            submitted = {f: request.form.get(f, "") for f in FEATURE_ORDER}
            X = parse_input(request.form)
            proba = float(PIPELINE.predict_proba(X)[0, 1])
            flagged = proba >= THRESHOLD
            imputed = [f for f in FEATURE_ORDER if not submitted[f].strip()]
            result = {
                "probability_pct": f"{proba * 100:.1f}%",
                "flagged":         flagged,
                "verdict":         "Flag for follow-up test" if flagged else "Low risk — no flag",
                "threshold_pct":   f"{THRESHOLD * 100:.0f}%",
                "imputed_fields":  imputed,
            }
        except ValueError:
            error = "All entered values must be numbers."
        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        features=FEATURE_META,
        feature_order=FEATURE_ORDER,
        submitted=submitted,
        result=result,
        error=error,
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        test_metrics=TEST_METRICS,
    )


@app.route("/health")
def health():
    """Simple health check — useful for deployment monitoring."""
    return {"status": "ok", "model": MODEL_NAME}, 200


if __name__ == "__main__":
    # Dev server only — use gunicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)
