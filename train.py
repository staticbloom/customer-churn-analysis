"""Train a separate churn candidate, or evaluate a saved model on labeled data."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from predict import DEFAULT_MODEL_PATH, RAW_FEATURE_COLUMNS, build_feature_frame, load_model_package, predict_customers


def churn_metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix_0_1": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Matching v2 inputs plus Churn=Yes/No")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; never overwrites a release")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evaluate-only", action="store_true", help="Score all supplied labels without fitting or selecting a threshold")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")

    data = pd.read_csv(args.input_csv, dtype={"customerID": "string"})
    if "Churn" not in data:
        raise ValueError("Labeled input must include Churn=Yes/No")
    features = build_feature_frame(data)
    labels = data["Churn"].astype("string").str.strip().map({"No": 0, "Yes": 1})
    if labels.isna().any() or set(labels) != {0, 1}:
        raise ValueError("Churn must contain both Yes and No, without missing labels")
    labels = labels.astype(int)
    if "customerID" in data:
        customer_ids = data["customerID"].astype("string").str.strip()
        if customer_ids.isna().any() or customer_ids.eq("").any() or customer_ids.duplicated().any():
            raise ValueError("customerID must be nonblank and unique after trimming whitespace")
        data["customerID"] = customer_ids
    if features.duplicated().any():
        raise ValueError("Resolve repeated feature records before splitting or evaluating")
    package = load_model_package(args.model)
    features = features.loc[:, package["feature_columns"]]
    split = pd.Series("evaluation", index=data.index)
    if args.evaluate_only:
        test_indices = data.index.to_numpy()
    else:
        if labels.value_counts().min() < 5:
            raise ValueError("Need at least five examples of each class for the three-way split")
        development, test_indices = train_test_split(data.index, test_size=0.2, stratify=labels, random_state=42)
        train_indices, validation = train_test_split(
            development, test_size=0.25, stratify=labels.loc[development], random_state=42,
        )
        model = clone(package["model"])
        model.fit(features.loc[train_indices], labels.loc[train_indices])
        validation_probability = model.predict_proba(features.loc[validation])[:, 1]
        thresholds = np.linspace(0.05, 0.95, 901)
        scores = [f1_score(labels.loc[validation], validation_probability >= t, zero_division=0) for t in thresholds]
        package = {
            "model": model, "selected_model": package["selected_model"],
            "parameters": model.named_steps["model"].get_params(),
            "threshold": float(thresholds[int(np.argmax(scores))]),
            "feature_columns": list(features.columns),
            "source_filename": args.input_csv.name,
        }
        split.loc[train_indices] = "train"
        split.loc[validation] = "validation"
        split.loc[test_indices] = "test"

    scored = predict_customers(package, data.loc[test_indices])
    metrics = churn_metrics(labels.loc[test_indices], scored.churn_probability, package["threshold"])
    report = {
        "status": "evaluation_only" if args.evaluate_only else "candidate_not_promoted",
        "source_filename": args.input_csv.name,
        "source_sha256": hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
        "reference_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_columns": RAW_FEATURE_COLUMNS,
        "threshold": float(package["threshold"]),
        "split_counts": {str(k): int(v) for k, v in split.value_counts().items()},
        "test_metrics": metrics,
        "evaluation_note": "Data independence and pre-churn field availability require source verification.",
        "protocol": "No fitting" if args.evaluate_only else "Seed 42; stratified 60/20/20; fit on train only; choose F1 threshold on validation; evaluate test once; no final refit",
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]},
    }
    if not args.evaluate_only:
        baseline = np.full(len(test_indices), labels.loc[train_indices].mean())
        report["training_prevalence_baseline"] = churn_metrics(labels.loc[test_indices], baseline, 0.5)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not args.evaluate_only:
        model_path = args.output_dir / "model.joblib"
        joblib.dump(package, model_path)
        reloaded = predict_customers(load_model_package(model_path), data.loc[test_indices])
        np.testing.assert_allclose(reloaded.churn_probability, scored.churn_probability, rtol=1e-12)
        report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    scored.assign(actual_churn=labels.loc[test_indices]).to_csv(args.output_dir / "evaluation_predictions.csv", index_label="source_row_index")
    split.rename("split").to_csv(args.output_dir / "split.csv", index_label="source_row_index")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {report['status']} to {args.output_dir}")


if __name__ == "__main__":
    main()
