"""Score v2 customer records with the saved churn model and decision threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "telco_churn_v2_40_20_20_model.joblib"
RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "support_tickets_30d",
    "support_tickets_90d",
    "complaints_90d",
    "late_payments_6m",
    "billing_disputes_6m",
    "data_usage_change_pct_mom",
    "minutes_change_pct_mom",
    "plan_downgrade_90d",
    "days_since_plan_change",
    "days_to_renewal",
    "contract_end_date",
    "dropped_calls_30d",
    "area_outages_90d",
    "outage_hours_90d",
    "nps_score",
    "nps_category",
    "competitor_offers_in_area",
    "competitor_promo_active",
]
NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "support_tickets_30d",
    "support_tickets_90d",
    "complaints_90d",
    "late_payments_6m",
    "billing_disputes_6m",
    "data_usage_change_pct_mom",
    "minutes_change_pct_mom",
    "plan_downgrade_90d",
    "days_since_plan_change",
    "days_to_renewal",
    "dropped_calls_30d",
    "area_outages_90d",
    "outage_hours_90d",
    "nps_score",
    "competitor_offers_in_area",
    "competitor_promo_active",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Telco v2 customer records with the final churn model."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV containing the required v2 customer input columns.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Saved model package. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Predictions CSV path. Defaults beside the input file.",
    )
    return parser.parse_args()


def normalise_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]):
        values = result[column].astype("string").str.strip().replace({"": pd.NA})
        # The saved sklearn imputer expects np.nan, rather than pandas' scalar NA.
        result[column] = values.astype(object).where(values.notna(), np.nan)
    return result


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Supply a nonempty table with unique column names")
    missing_columns = sorted(set(RAW_FEATURE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required v2 column(s): " + ", ".join(missing_columns)
        )
    features = normalise_text_columns(frame.loc[:, RAW_FEATURE_COLUMNS])
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise")
        values = features[column].dropna()
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column}: values must be finite or missing")
        if column not in {"data_usage_change_pct_mom", "minutes_change_pct_mom"} and (values < 0).any():
            raise ValueError(f"{column}: values must be nonnegative")
    for column in ["SeniorCitizen", "plan_downgrade_90d", "competitor_promo_active"]:
        if not features[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column}: use 0, 1 or a missing value")
    if not features["nps_score"].dropna().between(0, 10).all():
        raise ValueError("nps_score: this extract uses the individual 0–10 response scale")

    contract_end_date = pd.to_datetime(features["contract_end_date"], errors="coerce")
    invalid_dates = (
        contract_end_date.isna().sum() - features["contract_end_date"].isna().sum()
    )
    if invalid_dates:
        raise ValueError(
            f"Input CSV contains {invalid_dates} invalid contract_end_date value(s)."
        )
    features = features.drop(columns=["customerID", "Churn"], errors="ignore")
    features = features.drop(columns="contract_end_date")
    features["support_ticket_escalation"] = (
        features["support_tickets_30d"] - features["support_tickets_90d"] / 3
    )
    features["customer_friction_events"] = (
        features["complaints_90d"]
        + features["late_payments_6m"]
        + features["billing_disputes_6m"]
        + features["plan_downgrade_90d"]
    )
    features["service_reliability_events"] = (
        features["dropped_calls_30d"] + features["area_outages_90d"]
    )
    features["tenure_log"] = np.log1p(features["tenure"])
    features["monthly_charge_per_tenure"] = features["MonthlyCharges"] / (
        features["tenure"] + 1
    )
    if np.isinf(features.select_dtypes(include=np.number).to_numpy(dtype=float)).any():
        raise ValueError("Numeric feature construction overflowed; check source magnitudes and units")
    return features


def load_model_package(model_path: Path) -> dict:
    package = joblib.load(model_path)
    required_keys = {
        "model",
        "selected_model",
        "threshold",
        "feature_columns",
    }
    missing_keys = required_keys - set(package)
    if missing_keys:
        raise ValueError(
            "The model package is missing: " + ", ".join(sorted(missing_keys))
        )
    if list(package["model"].classes_) != [0, 1]:
        raise ValueError("The churn model must use classes 0 and 1")
    if not np.isfinite(package["threshold"]) or not 0 <= package["threshold"] <= 1:
        raise ValueError("The decision threshold must be between zero and one")
    return package


def predict_customers(package, raw_data):
    """Return probabilities and decisions in input order, before any presentation sorting."""
    features = build_feature_frame(raw_data)
    missing = set(package["feature_columns"]) - set(features)
    if missing:
        raise ValueError("Feature construction did not produce: " + ", ".join(sorted(missing)))
    features = features.loc[:, package["feature_columns"]]
    if list(features.columns) != list(package["model"].feature_names_in_):
        raise ValueError("Model feature order differs from its package")
    probabilities = package["model"].predict_proba(features)[:, 1]
    return pd.DataFrame({
        "churn_probability": probabilities,
        "predicted_churn": (probabilities >= package["threshold"]).astype(int),
    }, index=raw_data.index)


def main() -> None:
    args = parse_arguments()
    input_path = args.input_csv.resolve()
    model_path = args.model.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model package was not found: {model_path}")
    raw_data = pd.read_csv(input_path)
    package = load_model_package(model_path)
    scored = predict_customers(package, raw_data)
    probabilities = scored["churn_probability"].to_numpy()
    threshold = float(package["threshold"])
    predictions = scored["predicted_churn"].to_numpy()

    identifier = (
        raw_data["customerID"]
        if "customerID" in raw_data.columns
        else pd.Series(range(1, len(raw_data) + 1), name="record_id")
    )
    results = pd.DataFrame(
        {
            identifier.name: identifier,
            "churn_probability": probabilities,
            "decision_threshold": threshold,
            "predicted_churn": predictions,
            "predicted_label": pd.Series(predictions).map({0: "No Churn", 1: "Churn"}),
        }
    ).sort_values("churn_probability", ascending=False)
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_predictions.csv")
    )
    if output_path in {input_path, model_path} or output_path.exists():
        raise ValueError("Choose a new output path; existing inputs and outputs are not overwritten")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Scored {len(results)} customer record(s).")
    print(f"Model: {model_path}")
    print(f"Predictions: {output_path}")
    print(f"Decision threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
