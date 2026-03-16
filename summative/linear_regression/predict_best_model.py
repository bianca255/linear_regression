import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def load_input(input_path=None):
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    sample_path = BASE_DIR / "sample_input.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError("Provide --input payload.json or create sample_input.json in this folder")


def main():
    parser = argparse.ArgumentParser(description="Predict student final grade using the saved best model")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to JSON payload (single record). Defaults to sample_input.json",
    )
    args = parser.parse_args()

    model = joblib.load(BASE_DIR / "best_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    with open(BASE_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    payload = load_input(args.input)
    row_df = pd.DataFrame([payload])

    missing_cols = [c for c in feature_columns if c not in row_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    row_df = row_df[feature_columns]
    row_processed = scaler.transform(row_df)
    prediction = float(model.predict(row_processed)[0])

    print("Model used: best_model.pkl")
    print(f"Predicted G3: {prediction:.2f}")


if __name__ == "__main__":
    main()
