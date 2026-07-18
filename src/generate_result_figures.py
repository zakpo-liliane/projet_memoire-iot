from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
BASELINE_MODEL_DIR = ROOT / "models" / "baselines"
DL_MODEL_DIR = ROOT / "models" / "deep_learning"
BASELINE_REPORT_DIR = ROOT / "reports" / "baselines"
DL_REPORT_DIR = ROOT / "reports" / "deep_learning"
FIGURE_DIR = ROOT / "reports" / "figures"

CLASS_LABELS = ["attack", "benign"]


def load_validation() -> tuple[pd.DataFrame, pd.Series]:
    x_val = pd.read_pickle(SPLIT_DIR / "X_val_scaled.pkl.gz", compression="gzip")
    y_val = pd.read_csv(SPLIT_DIR / "y_val.csv").iloc[:, 0].astype("int32")
    return x_val, y_val


def save_metric_comparison() -> None:
    baseline_path = BASELINE_REPORT_DIR / "baseline_metrics_validation.csv"
    dl_path = DL_REPORT_DIR / "deep_learning_metrics_validation.csv"

    frames = []
    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path))
    if dl_path.exists():
        frames.append(pd.read_csv(dl_path))
    if not frames:
        raise FileNotFoundError("No metrics files found.")

    metrics_df = pd.concat(frames, ignore_index=True, sort=False)
    metrics_df = metrics_df.drop_duplicates(subset=["model"], keep="last")
    metrics_df.to_csv(FIGURE_DIR / "all_model_metrics_validation.csv", index=False)

    score_cols = ["accuracy", "precision", "recall", "f1_score"]
    plot_df = metrics_df.melt(id_vars="model", value_vars=score_cols, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Model comparison on validation set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_metrics_comparison_validation.png", dpi=160)
    plt.close(fig)

    f1_df = metrics_df.sort_values("f1_score", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=f1_df, x="f1_score", y="model", color="#2F80ED", ax=ax)
    ax.set_title("F1-score comparison on validation set")
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "f1_score_comparison_validation.png", dpi=160)
    plt.close(fig)


def save_baseline_confusion_matrices(x_val: pd.DataFrame, y_val: pd.Series) -> None:
    for model_path in sorted(BASELINE_MODEL_DIR.glob("*.joblib")):
        model_name = model_path.stem
        model = joblib.load(model_path)
        y_pred = model.predict(x_val)
        cm = confusion_matrix(y_val, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in CLASS_LABELS],
            columns=[f"pred_{label}" for label in CLASS_LABELS],
        )
        cm_df.to_csv(BASELINE_REPORT_DIR / f"{model_name}_confusion_matrix_validation.csv")

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{model_name} confusion matrix - validation")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{model_name}_confusion_matrix_validation.png", dpi=160)
        plt.close(fig)


def save_cnn_figures(x_val: pd.DataFrame, y_val: pd.Series) -> None:
    cnn_path = DL_MODEL_DIR / "cnn.keras"
    cm_path = DL_REPORT_DIR / "cnn_confusion_matrix_validation.csv"

    if cm_path.exists():
        cm_df = pd.read_csv(cm_path, index_col=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title("CNN confusion matrix - validation")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "cnn_confusion_matrix_validation.png", dpi=160)
        plt.close(fig)

    history_path = DL_REPORT_DIR / "cnn_training_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history_df["epoch"], history_df["loss"], label="train_loss")
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
        axes[0].set_title("CNN loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[1].plot(history_df["epoch"], history_df["accuracy"], label="train_accuracy")
        axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
        axes[1].set_title("CNN accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "cnn_training_history.png", dpi=160)
        plt.close(fig)

    if cnn_path.exists():
        try:
            model = tf.keras.models.load_model(cnn_path)
        except ValueError as exc:
            print(f"Skipped CNN ROC curve: {exc}")
            return

        x_val_seq = x_val.to_numpy(dtype=np.float32).reshape(len(x_val), x_val.shape[1], 1)
        y_scores = model.predict(x_val_seq, batch_size=512, verbose=0).ravel()
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="CNN").plot(ax=ax)
        ax.set_title("CNN ROC curve - validation")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "cnn_roc_curve_validation.png", dpi=160)
        plt.close(fig)

        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(FIGURE_DIR / "cnn_roc_curve_validation.csv", index=False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_val, y_val = load_validation()
    save_metric_comparison()
    save_baseline_confusion_matrices(x_val, y_val)
    save_cnn_figures(x_val, y_val)

    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
