from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "outputs" / "splits"
BASELINE_MODEL_DIR = ROOT / "models" / "baselines"
DL_MODEL_DIR = ROOT / "models" / "deep_learning"
TUNING_MODEL_DIR = DL_MODEL_DIR / "tuning"
BASELINE_REPORT_DIR = ROOT / "reports" / "evaluation" / "baselines"
DL_REPORT_DIR = ROOT / "reports" / "evaluation" / "deep_learning"
FIGURE_DIR = ROOT / "reports" / "evaluation" / "figures"
PREDICTION_DIR = ROOT / "reports" / "evaluation" / "predictions"

SEED = 42
CLASS_LABELS = ["attack", "benign"]
BEST_MODEL_ORDER = ["cnn", "lstm", "cnn_lstm", "cnn_lstm_attention", "autoencoder"]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_deep_models import to_temporal_tensor  # noqa: E402


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    x_test = pd.read_pickle(SPLIT_DIR / "X_test_scaled.pkl.gz", compression="gzip")
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv").iloc[:, 0].astype("int32")
    return x_test, y_test


def load_baseline_model(path: Path):
    return joblib.load(path)


def load_deep_model(model_name: str) -> tuple[tf.keras.Model, str]:
    tuning_best_path = TUNING_MODEL_DIR / f"{model_name}_best.keras"
    tuned_cfg_path = TUNING_MODEL_DIR / f"{model_name}_best_config.csv"
    base_path = DL_MODEL_DIR / f"{model_name}.keras"

    if tuning_best_path.exists():
        config_name = "best"
        if tuned_cfg_path.exists():
            config_name = pd.read_csv(tuned_cfg_path).iloc[0]["best_config"]
        return tf.keras.models.load_model(tuning_best_path), config_name
    if base_path.exists():
        return tf.keras.models.load_model(base_path), "base"
    raise FileNotFoundError(f"No saved model found for {model_name}.")


def available_deep_models() -> list[str]:
    models: list[str] = []
    for model_name in BEST_MODEL_ORDER:
        tuning_best_path = TUNING_MODEL_DIR / f"{model_name}_best.keras"
        base_path = DL_MODEL_DIR / f"{model_name}.keras"
        if tuning_best_path.exists() or base_path.exists():
            models.append(model_name)
    return models


def write_confusion_matrix(
    report_dir: Path,
    figure_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_attack", "true_benign"],
        columns=["pred_attack", "pred_benign"],
    )
    cm_df.to_csv(report_dir / f"{model_name}_confusion_matrix_test.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name} confusion matrix - test")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{model_name}_confusion_matrix_test.png", dpi=160)
    plt.close(fig)
    return cm_df


def write_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "model": model_name,
            "row_id": np.arange(len(y_true), dtype="int32"),
            "y_true": y_true.astype("int32"),
            "y_pred": y_pred.astype("int32"),
            "y_score": y_scores.astype(float),
        }
    )
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTION_DIR / f"{model_name}_predictions_test.csv", index=False)
    return pred_df


def write_roc_curve(
    report_dir: Path,
    figure_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        report_dir / f"{model_name}_roc_curve_test.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"{model_name} ROC curve - test")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{model_name}_roc_curve_test.png", dpi=160)
    plt.close(fig)
    return roc_auc


def baseline_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, train_seconds: float | None = None) -> dict[str, object]:
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "attack_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "attack_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "attack_f1_score": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "attack_roc_auc": roc_auc_score((y_true == 0).astype("int32"), 1.0 - y_scores),
    }
    if train_seconds is not None:
        metrics["train_seconds"] = train_seconds
    return metrics


def evaluate_baselines(x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, object]]:
    results = []
    for model_path in sorted(BASELINE_MODEL_DIR.glob("*.joblib")):
        model_name = model_path.stem
        model = load_baseline_model(model_path)
        y_pred = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(x_test)[:, 1]
        else:
            decision = model.decision_function(x_test)
            y_scores = expit(decision)

        write_confusion_matrix(BASELINE_REPORT_DIR, FIGURE_DIR, model_name, y_test.to_numpy(), y_pred)
        write_roc_curve(BASELINE_REPORT_DIR, FIGURE_DIR, model_name, y_test.to_numpy(), y_scores)
        write_predictions(model_name, y_test.to_numpy(), y_pred, y_scores)
        results.append(baseline_metrics(model_name, y_test.to_numpy(), y_pred, y_scores))
    return results


def evaluate_cnn_like(
    model_name: str,
    model: tf.keras.Model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    x_test_seq = to_temporal_tensor(x_test)
    y_scores = model.predict(x_test_seq, batch_size=512, verbose=0).ravel()
    y_pred = (y_scores >= 0.5).astype("int32")
    write_confusion_matrix(DL_REPORT_DIR, FIGURE_DIR, model_name, y_test.to_numpy(), y_pred)
    write_roc_curve(DL_REPORT_DIR, FIGURE_DIR, model_name, y_test.to_numpy(), y_scores)
    write_predictions(model_name, y_test.to_numpy(), y_pred, y_scores)
    return baseline_metrics(model_name, y_test.to_numpy(), y_pred, y_scores)


def evaluate_autoencoder(
    model: tf.keras.Model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    best_cfg_path = TUNING_MODEL_DIR / "autoencoder_best_config.csv"
    threshold_path = None
    threshold = 0.5

    if best_cfg_path.exists():
        best_cfg = pd.read_csv(best_cfg_path).iloc[0]["best_config"]
        threshold_path = TUNING_MODEL_DIR / f"autoencoder_{best_cfg}_threshold.csv"
        if threshold_path.exists():
            threshold = float(pd.read_csv(threshold_path).iloc[0]["threshold"])

    x_test_matrix = x_test.to_numpy(dtype=np.float32)
    reconstructed = model.predict(x_test_matrix, batch_size=512, verbose=0)
    reconstruction_error = np.mean(np.square(x_test_matrix - reconstructed), axis=1)
    benign_scores = 1.0 - (
        (reconstruction_error - reconstruction_error.min())
        / (reconstruction_error.max() - reconstruction_error.min() + 1e-8)
    )
    y_pred = (reconstruction_error <= threshold).astype("int32")

    write_confusion_matrix(DL_REPORT_DIR, FIGURE_DIR, "autoencoder", y_test.to_numpy(), y_pred)
    write_roc_curve(DL_REPORT_DIR, FIGURE_DIR, "autoencoder", y_test.to_numpy(), benign_scores)
    write_predictions("autoencoder", y_test.to_numpy(), y_pred, benign_scores)
    return baseline_metrics("autoencoder", y_test.to_numpy(), y_pred, benign_scores)


def save_comparison(results: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    sort_col = "attack_f1_score" if "attack_f1_score" in df.columns else "f1_score"
    df = df.sort_values(sort_col, ascending=False)
    df.to_csv(FIGURE_DIR / "all_model_metrics_test.csv", index=False)

    score_cols = ["accuracy", "precision", "recall", "f1_score"]
    plot_df = df.melt(id_vars="model", value_vars=score_cols, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Model comparison on test set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_metrics_comparison_test.png", dpi=160)
    plt.close(fig)

    best_df = df.head(1).copy()
    best_df.to_csv(FIGURE_DIR / "best_model_test.csv", index=False)
    return df


def save_summary(df: pd.DataFrame) -> None:
    summary = [
        "# Step 11 Test Evaluation",
        "",
        "The table below summarizes the test-set performance of all evaluated models.",
        "",
        "```csv",
        df.round(4).to_csv(index=False).strip(),
        "```",
    ]
    (FIGURE_DIR / "test_evaluation_summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_test, y_test = load_test_data()

    results: list[dict[str, object]] = []
    results.extend(evaluate_baselines(x_test, y_test))

    for model_name in available_deep_models():
        if model_name == "autoencoder":
            model, _ = load_deep_model(model_name)
            results.append(evaluate_autoencoder(model, x_test, y_test))
        else:
            model, _ = load_deep_model(model_name)
            results.append(evaluate_cnn_like(model_name, model, x_test, y_test))

    comparison_df = save_comparison(results)
    save_summary(comparison_df)

    print(comparison_df.to_string(index=False))
    print(f"Saved test evaluation to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
