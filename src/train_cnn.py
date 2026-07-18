from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
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
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "models" / "deep_learning"
REPORT_DIR = ROOT / "reports" / "deep_learning"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 512


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = pd.read_pickle(SPLIT_DIR / "X_train_scaled_smote.pkl.gz", compression="gzip")
    y_train = pd.read_csv(SPLIT_DIR / "y_train_smote.csv").iloc[:, 0].astype("int32")
    x_val = pd.read_pickle(SPLIT_DIR / "X_val_scaled.pkl.gz", compression="gzip")
    y_val = pd.read_csv(SPLIT_DIR / "y_val.csv").iloc[:, 0].astype("int32")

    x_train_seq = x_train.to_numpy(dtype=np.float32).reshape(len(x_train), x_train.shape[1], 1)
    x_val_seq = x_val.to_numpy(dtype=np.float32).reshape(len(x_val), x_val.shape[1], 1)
    return x_train_seq, y_train.to_numpy(), x_val_seq, y_val.to_numpy()


def save_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        REPORT_DIR / "cnn_roc_curve_validation.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("CNN ROC curve - validation")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cnn_roc_curve_validation.png", dpi=150)
    plt.close(fig)


def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def save_training_plot(history_df: pd.DataFrame) -> None:
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
    fig.savefig(REPORT_DIR / "cnn_training_history.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(cm_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("CNN confusion matrix - validation")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cnn_confusion_matrix_validation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = load_data()
    model = build_model((x_train.shape[1], x_train.shape[2]))

    checkpoint_path = MODEL_DIR / "cnn.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )
    train_seconds = time.time() - start

    y_scores = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred = (y_scores >= 0.5).astype("int32")
    y_attack = (y_val == 0).astype("int32")
    attack_scores = 1.0 - y_scores

    metrics = pd.DataFrame(
        [
            {
                "model": "cnn",
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1_score": f1_score(y_val, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_scores),
                "attack_precision": precision_score(y_val, y_pred, pos_label=0, zero_division=0),
                "attack_recall": recall_score(y_val, y_pred, pos_label=0, zero_division=0),
                "attack_f1_score": f1_score(y_val, y_pred, pos_label=0, zero_division=0),
                "macro_f1_score": f1_score(y_val, y_pred, average="macro", zero_division=0),
                "attack_roc_auc": roc_auc_score(y_attack, attack_scores),
                "train_seconds": train_seconds,
            }
        ]
    )

    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_attack", "true_benign"],
        columns=["pred_attack", "pred_benign"],
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    metrics.to_csv(REPORT_DIR / "cnn_metrics_validation.csv", index=False)
    cm_df.to_csv(REPORT_DIR / "cnn_confusion_matrix_validation.csv")
    history_df.to_csv(REPORT_DIR / "cnn_training_history.csv", index=False)
    save_training_plot(history_df)
    save_confusion_matrix_plot(cm_df)
    save_roc_curve(y_val, y_scores)

    pd.DataFrame(
        [
            {
                "model": "cnn",
                "train_shape": str(x_train.shape),
                "validation_shape": str(x_val.shape),
                "input_shape": str((x_train.shape[1], x_train.shape[2])),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "target": "label1_encoded",
                "task": "binary classification: attack vs benign",
            }
        ]
    ).to_csv(REPORT_DIR / "cnn_experiment_setup.csv", index=False)

    print(metrics.to_string(index=False))
    print(cm_df.to_string())
    print(f"Saved model to: {checkpoint_path}")


if __name__ == "__main__":
    main()
