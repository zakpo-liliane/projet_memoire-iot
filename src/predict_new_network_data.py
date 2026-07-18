from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final" / "best_model_bundle.joblib"
REPORT_DIR = ROOT / "reports" / "inference"

LABEL_CANDIDATES = [
    "label1_encoded",
    "label1",
    "target",
    "class",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the final saved IDS model on new network data and optionally "
            "compute metrics if a label column is available."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a CSV or pickle file containing new network samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory where predictions and reports will be saved.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Optional path to a CSV file containing the ground-truth labels for the "
            "same rows as the input file."
        ),
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)
    if suffixes.endswith(".pkl") or suffixes.endswith(".pkl.gz"):
        return pd.read_pickle(path, compression="gzip" if suffixes.endswith(".gz") else None)
    raise ValueError(f"Unsupported input format: {path.suffixes}")


def load_labels(path: Path) -> pd.Series:
    labels = pd.read_csv(path).iloc[:, 0]
    return pd.to_numeric(labels, errors="raise").astype("int32")


def infer_label_column(df: pd.DataFrame) -> str | None:
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, str | None]:
    label_col = infer_label_column(df)
    if label_col is None:
        return df.copy(), None, None

    target = df[label_col]
    features = df.drop(columns=[label_col]).copy()

    if label_col == "label1":
        # Project convention: attack=0, benign=1.
        if target.dtype.name in {"category", "object"}:
            normalized = target.astype(str).str.lower().str.strip()
            mapped = normalized.map(
                {
                    "attack": 0,
                    "attacking": 0,
                    "malicious": 0,
                    "benign": 1,
                    "normal": 1,
                    "good": 1,
                }
            )
            if mapped.isna().any():
                raise ValueError(
                    "Could not map label1 values to binary classes. "
                    "Please provide label1_encoded or an explicit binary target."
                )
            target = mapped.astype("int32")
        else:
            target = pd.to_numeric(target, errors="raise").astype("int32")
    else:
        target = pd.to_numeric(target, errors="raise").astype("int32")

    return features, target, label_col


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = df.copy()

    # Drop columns that are clearly non-features if they survived ingestion.
    for col in [
        "label1",
        "label2",
        "label3",
        "label4",
        "label_full",
        "timestamp",
        "timestamp_start",
        "timestamp_end",
    ]:
        if col in aligned.columns:
            aligned = aligned.drop(columns=[col])

    # Keep the expected model features only, filling any missing ones.
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0.0

    extra_cols = [col for col in aligned.columns if col not in feature_names]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)

    aligned = aligned[feature_names]
    aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aligned


def predict(bundle: dict[str, object], features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = bundle["scaler"]
    classifier = bundle["classifier"]

    x_scaled = scaler.transform(features)
    y_pred = classifier.predict(x_scaled)

    if hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(x_scaled)[:, 1]
    elif hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(x_scaled)
        y_proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_proba = y_pred.astype(float)

    return y_pred.astype("int32"), y_proba.astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "attack_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "attack_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "attack_f1_score": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "attack_roc_auc": roc_auc_score((y_true == 0).astype("int32"), 1.0 - y_proba),
    }
    return metrics


def save_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, float] | None,
    bundle: dict[str, object],
    input_path: Path,
    label_col: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "new_data_predictions.csv", index=False)

    summary = {
        "model_name": bundle["model_name"],
        "input_file": str(input_path),
        "samples": int(len(predictions)),
        "label_column": label_col,
        "has_ground_truth": metrics is not None,
    }
    if metrics is not None:
        summary.update(metrics)

    (output_dir / "new_data_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if metrics is not None:
        pd.DataFrame([summary]).to_csv(output_dir / "new_data_metrics.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))


def main() -> None:
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Final model bundle not found: {MODEL_PATH}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    feature_names = list(bundle["feature_names"])

    raw_df = load_data(args.input)
    features_df, y_true, label_col = extract_target(raw_df)
    if y_true is None and args.labels is not None:
        y_true = load_labels(args.labels)
        label_col = args.labels.name
    aligned = align_features(features_df, feature_names)
    y_pred, y_proba = predict(bundle, aligned)

    result_df = pd.DataFrame(
        {
            "predicted_class": y_pred,
            "predicted_label": np.where(y_pred == 0, "attack", "benign"),
            "attack_probability": y_proba,
            "benign_probability": 1.0 - y_proba,
        }
    )

    metrics = None
    if y_true is not None:
        metrics = compute_metrics(y_true.to_numpy(), y_pred, y_proba)
        cm = confusion_matrix(y_true.to_numpy(), y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["true_attack", "true_benign"],
            columns=["pred_attack", "pred_benign"],
        )
        cm_df.to_csv(args.output_dir / "new_data_confusion_matrix.csv", index=True)
        result_df.insert(0, "true_class", y_true.to_numpy())

    result_df.insert(0, "row_id", np.arange(len(result_df), dtype="int32"))
    save_outputs(args.output_dir, result_df, metrics, bundle, args.input, label_col)


if __name__ == "__main__":
    main()
