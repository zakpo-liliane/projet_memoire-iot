from __future__ import annotations

import argparse
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
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "models" / "deep_learning"
REPORT_DIR = ROOT / "reports" / "deep_learning"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 512

MODEL_ORDER = ["cnn", "lstm", "autoencoder", "cnn_lstm", "cnn_lstm_attention"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the deep learning models used in step 9 of the IDS thesis."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        choices=MODEL_ORDER,
        help="Models to train, in the desired order.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--val-sample-size", type=int, default=20000)
    parser.add_argument(
        "--balance-method",
        choices=["smote", "class_weight", "both"],
        default="smote",
        help="How to handle class imbalance for supervised deep learning models.",
    )
    parser.add_argument(
        "--loss-function",
        choices=["binary_crossentropy", "focal_loss"],
        default="binary_crossentropy",
        help="Loss function used for supervised deep learning models.",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use the full training and validation splits instead of reduced samples.",
    )
    parser.add_argument("--keras-verbose", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count for TensorFlow. Keep it low for reproducibility on large tabular sets.",
    )
    return parser.parse_args()


def set_runtime(threads: int) -> None:
    tf.keras.utils.set_random_seed(SEED)
    if threads and threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        tf.config.threading.set_intra_op_parallelism_threads(threads)


def load_supervised_train() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_pickle(
        SPLIT_DIR / "X_train_scaled_smote.pkl.gz",
        compression="gzip",
    ).reset_index(drop=True)
    y = pd.read_csv(SPLIT_DIR / "y_train_smote.csv").iloc[:, 0].astype("int32").reset_index(
        drop=True
    )
    return x, y


def load_train_raw() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_pickle(
        SPLIT_DIR / "X_train_scaled.pkl.gz",
        compression="gzip",
    ).reset_index(drop=True)
    y = pd.read_csv(SPLIT_DIR / "y_train.csv").iloc[:, 0].astype("int32").reset_index(drop=True)
    return x, y


def load_validation() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_pickle(
        SPLIT_DIR / "X_val_scaled.pkl.gz",
        compression="gzip",
    ).reset_index(drop=True)
    y = pd.read_csv(SPLIT_DIR / "y_val.csv").iloc[:, 0].astype("int32").reset_index(drop=True)
    return x, y


def balanced_sample(
    x: pd.DataFrame,
    y: pd.Series,
    sample_size: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size is None or sample_size >= len(x):
        return x, y

    joined = x.copy()
    joined["_target"] = y.values
    per_class = max(sample_size // joined["_target"].nunique(), 1)

    sampled_frames = []
    for _, class_frame in joined.groupby("_target"):
        sampled_frames.append(
            class_frame.sample(n=min(len(class_frame), per_class), random_state=SEED)
        )

    sampled_df = pd.concat(sampled_frames).sample(frac=1.0, random_state=SEED)
    sampled_y = sampled_df.pop("_target").astype("int32")
    return sampled_df, sampled_y


def to_temporal_tensor(x: pd.DataFrame) -> np.ndarray:
    matrix = x.to_numpy(dtype=np.float32)
    return matrix.reshape(len(matrix), matrix.shape[1], 1)


def compute_class_weights(y: pd.Series) -> dict[int, float]:
    labels = np.asarray(y, dtype=np.int32)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return alpha_factor * modulating_factor * ce

    return loss


def supervised_loss(loss_name: str):
    if loss_name == "focal_loss":
        return binary_focal_loss()
    return "binary_crossentropy"


def supervised_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def build_experiment_note(
    full_train_size: int,
    used_train_size: int,
    full_val_size: int,
    used_val_size: int,
    balance_method: str,
    loss_function: str,
) -> str:
    if used_train_size == full_train_size and used_val_size == full_val_size:
        base_note = "Full train and validation datasets used."
    else:
        base_note = "Reduced stratified subsets used for faster deep learning experiments."
    return f"{base_note} Balance method: {balance_method}. Loss function: {loss_function}."


def build_binary_metrics(
    model_name: str,
    y_true: np.ndarray,
    benign_scores: np.ndarray,
    train_seconds: float,
) -> dict[str, float]:
    y_pred = (benign_scores >= 0.5).astype("int32")
    y_attack = (y_true == 0).astype("int32")
    attack_scores = 1.0 - benign_scores

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "attack_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "attack_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "attack_f1_score": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, benign_scores),
        "attack_roc_auc": roc_auc_score(y_attack, attack_scores),
        "train_seconds": train_seconds,
    }


def save_confusion_matrix(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_attack", "true_benign"],
        columns=["pred_attack", "pred_benign"],
    )
    cm_df.to_csv(REPORT_DIR / f"{model_name}_confusion_matrix_validation.csv")
    return cm_df


def save_roc_curve(model_name: str, y_true: np.ndarray, benign_scores: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, benign_scores)
    roc_auc = auc(fpr, tpr)

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        REPORT_DIR / f"{model_name}_roc_curve_validation.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"{model_name.upper()} ROC curve - validation")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"{model_name}_roc_curve_validation.png", dpi=160)
    plt.close(fig)


def save_training_history(model_name: str, history: tf.keras.callbacks.History) -> pd.DataFrame:
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(REPORT_DIR / f"{model_name}_training_history.csv", index=False)

    plot_cols = [col for col in ["loss", "val_loss", "accuracy", "val_accuracy"] if col in history_df]
    if plot_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if "loss" in history_df:
            axes[0].plot(history_df["epoch"], history_df["loss"], label="train_loss")
        if "val_loss" in history_df:
            axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
        axes[0].set_title(f"{model_name.upper()} loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        if "accuracy" in history_df:
            axes[1].plot(history_df["epoch"], history_df["accuracy"], label="train_accuracy")
        if "val_accuracy" in history_df:
            axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
        axes[1].set_title(f"{model_name.upper()} accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"{model_name}_training_history.png", dpi=160)
        plt.close(fig)

    return history_df


def common_callbacks(model_name: str) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def build_cnn(input_shape: tuple[int, int], loss_name: str = "binary_crossentropy") -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss=supervised_loss(loss_name), metrics=supervised_metrics())
    return model


def build_lstm(input_shape: tuple[int, int], loss_name: str = "binary_crossentropy") -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss=supervised_loss(loss_name), metrics=supervised_metrics())
    return model


def build_cnn_lstm(input_shape: tuple[int, int], loss_name: str = "binary_crossentropy") -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss=supervised_loss(loss_name), metrics=supervised_metrics())
    return model


def build_cnn_lstm_attention(
    input_shape: tuple[int, int],
    loss_name: str = "binary_crossentropy",
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_attention")
    model.compile(optimizer="adam", loss=supervised_loss(loss_name), metrics=supervised_metrics())
    return model


def build_autoencoder(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(128, activation="relu")(inputs)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.Dense(64, activation="relu")(encoded)
    latent = tf.keras.layers.Dense(16, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(64, activation="relu")(latent)
    decoded = tf.keras.layers.Dense(128, activation="relu")(decoded)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(decoded)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_temporal_model(
    model_name: str,
    builder,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    epochs: int,
    batch_size: int,
    keras_verbose: int,
    class_weight: dict[int, float] | None = None,
    loss_name: str = "binary_crossentropy",
) -> tuple[dict[str, float], pd.DataFrame]:
    x_train_seq = to_temporal_tensor(x_train)
    x_val_seq = to_temporal_tensor(x_val)
    model = builder((x_train_seq.shape[1], x_train_seq.shape[2]), loss_name)

    start = time.time()
    history = model.fit(
        x_train_seq,
        y_train.to_numpy(),
        validation_data=(x_val_seq, y_val.to_numpy()),
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbose,
        callbacks=common_callbacks(model_name),
        class_weight=class_weight,
    )
    train_seconds = time.time() - start

    benign_scores = model.predict(x_val_seq, batch_size=batch_size, verbose=0).ravel()
    y_pred = (benign_scores >= 0.5).astype("int32")
    metrics = build_binary_metrics(model_name, y_val.to_numpy(), benign_scores, train_seconds)
    save_confusion_matrix(model_name, y_val.to_numpy(), y_pred)
    save_roc_curve(model_name, y_val.to_numpy(), benign_scores)
    history_df = save_training_history(model_name, history)
    return metrics, history_df


def train_autoencoder(
    x_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    sample_size: int | None,
    epochs: int,
    batch_size: int,
    keras_verbose: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    benign_train = x_train_raw.loc[y_train_raw == 1]
    if sample_size is not None:
        benign_train = benign_train.sample(
            n=min(len(benign_train), max(sample_size // 2, 1)),
            random_state=SEED,
        )
    benign_val = x_val.loc[y_val == 1]

    model = build_autoencoder(x_train_raw.shape[1])
    callbacks = common_callbacks("autoencoder")

    start = time.time()
    history = model.fit(
        benign_train.to_numpy(dtype=np.float32),
        benign_train.to_numpy(dtype=np.float32),
        validation_data=(
            benign_val.to_numpy(dtype=np.float32),
            benign_val.to_numpy(dtype=np.float32),
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbose,
        callbacks=callbacks,
    )
    train_seconds = time.time() - start

    val_matrix = x_val.to_numpy(dtype=np.float32)
    reconstructed = model.predict(val_matrix, batch_size=batch_size, verbose=0)
    reconstruction_error = np.mean(np.square(val_matrix - reconstructed), axis=1)
    benign_errors = reconstruction_error[y_val.to_numpy() == 1]
    threshold = float(np.quantile(benign_errors, 0.95))

    benign_scores = 1.0 - (
        (reconstruction_error - reconstruction_error.min())
        / (reconstruction_error.max() - reconstruction_error.min() + 1e-8)
    )
    y_pred = (reconstruction_error <= threshold).astype("int32")
    metrics = build_binary_metrics("autoencoder", y_val.to_numpy(), benign_scores, train_seconds)
    metrics["threshold"] = threshold

    save_confusion_matrix("autoencoder", y_val.to_numpy(), y_pred)
    save_roc_curve("autoencoder", y_val.to_numpy(), benign_scores)
    pd.DataFrame([{"threshold": threshold}]).to_csv(
        REPORT_DIR / "autoencoder_threshold.csv",
        index=False,
    )
    history_df = save_training_history("autoencoder", history)
    return metrics, history_df


def save_metrics(results: list[dict[str, float]]) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    metrics_path = REPORT_DIR / "deep_learning_metrics_validation.csv"

    if metrics_path.exists():
        existing_df = pd.read_csv(metrics_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True, sort=False)

    results_df = results_df.drop_duplicates(subset=["model"], keep="last")
    if "attack_f1_score" in results_df.columns:
        results_df = results_df.sort_values(by="attack_f1_score", ascending=False)
    else:
        results_df = results_df.sort_values(by="f1_score", ascending=False)

    results_df.to_csv(metrics_path, index=False)
    return results_df


def save_experiment_setup(
    models: list[str],
    epochs: int,
    batch_size: int,
    full_train_size: int,
    used_train_size: int,
    full_validation_size: int,
    used_validation_size: int,
    train_shape: tuple[int, int],
    validation_shape: tuple[int, int],
    balance_method: str,
    loss_function: str,
) -> None:
    pd.DataFrame(
        [
            {
                "models": ",".join(models),
                "epochs": epochs,
                "batch_size": batch_size,
                "train_sample_size": used_train_size,
                "validation_sample_size": used_validation_size,
                "train_shape": str(train_shape),
                "validation_shape": str(validation_shape),
                "target": "label1_encoded",
                "task": "binary classification: attack vs benign",
                "note": build_experiment_note(
                    full_train_size,
                    used_train_size,
                    full_validation_size,
                    used_validation_size,
                    balance_method,
                    loss_function,
                ),
            }
        ]
    ).to_csv(REPORT_DIR / "deep_learning_experiment_setup.csv", index=False)


def write_summary_report(results_df: pd.DataFrame) -> None:
    metrics_text = results_df.round(4).to_csv(index=False)
    lines = [
        "# Deep Learning Step 9 Summary",
        "",
        "Ce bloc couvre l'implémentation des modeles deep learning du memoire.",
        "",
        "## Modeles",
        "",
        "- CNN: extraction de motifs locaux sur les features normalisées.",
        "- LSTM: apprentissage de dependances séquentielles sur le vecteur de features.",
        "- Autoencoder: detection d'anomalies à partir des erreurs de reconstruction sur le trafic normal.",
        "- CNN + LSTM: modele hybride combinant convolution et memoire sequentielle.",
        "- CNN + LSTM + Attention: extension hybride integrant un mecanisme d'attention multi-tete.",
        "",
        "## Comparaison validation",
        "",
        "```csv",
        metrics_text.strip(),
        "```",
        "",
        "## Critere de selection",
        "",
        "Le meilleur modele est retenu sur la base de `attack_f1_score` pour privilegier la detection des attaques.",
    ]
    (REPORT_DIR / "deep_learning_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_runtime(args.threads)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_smote, y_train_smote = load_supervised_train()
    full_train_size = len(x_train_smote)
    train_sample_size = None if args.full_data else args.sample_size
    val_sample_size = None if args.full_data else args.val_sample_size
    x_train_smote, y_train_smote = balanced_sample(
        x_train_smote,
        y_train_smote,
        train_sample_size,
    )

    x_train_raw, y_train_raw = load_train_raw()
    x_val, y_val = load_validation()
    full_val_size = len(x_val)
    x_val, y_val = balanced_sample(x_val, y_val, val_sample_size)

    if args.balance_method == "class_weight":
        x_supervised_train = x_train_raw
        y_supervised_train = y_train_raw
        class_weights = compute_class_weights(y_train_raw)
    elif args.balance_method == "both":
        x_supervised_train = x_train_smote
        y_supervised_train = y_train_smote
        class_weights = compute_class_weights(y_train_raw)
    else:
        x_supervised_train = x_train_smote
        y_supervised_train = y_train_smote
        class_weights = None

    print(f"Supervised DL train subset: {x_supervised_train.shape}")
    print(f"Validation set: {x_val.shape}")
    if class_weights is not None:
        print(f"Class weights enabled: {class_weights}")

    results: list[dict[str, float]] = []

    for model_name in args.models:
        print(f"Training {model_name}...")
        if model_name == "cnn":
            metrics, _ = train_temporal_model(
                "cnn",
                build_cnn,
                x_supervised_train,
                y_supervised_train,
                x_val,
                y_val,
                args.epochs,
                args.batch_size,
                args.keras_verbose,
                class_weight=class_weights,
                loss_name=args.loss_function,
            )
        elif model_name == "lstm":
            metrics, _ = train_temporal_model(
                "lstm",
                build_lstm,
                x_supervised_train,
                y_supervised_train,
                x_val,
                y_val,
                args.epochs,
                args.batch_size,
                args.keras_verbose,
                class_weight=class_weights,
                loss_name=args.loss_function,
            )
        elif model_name == "cnn_lstm":
            metrics, _ = train_temporal_model(
                "cnn_lstm",
                build_cnn_lstm,
                x_supervised_train,
                y_supervised_train,
                x_val,
                y_val,
                args.epochs,
                args.batch_size,
                args.keras_verbose,
                class_weight=class_weights,
                loss_name=args.loss_function,
            )
        elif model_name == "cnn_lstm_attention":
            metrics, _ = train_temporal_model(
                "cnn_lstm_attention",
                build_cnn_lstm_attention,
                x_supervised_train,
                y_supervised_train,
                x_val,
                y_val,
                args.epochs,
                args.batch_size,
                args.keras_verbose,
                class_weight=class_weights,
                loss_name=args.loss_function,
            )
        else:
            metrics, _ = train_autoencoder(
                x_train_raw,
                y_train_raw,
                x_val,
                y_val,
                train_sample_size,
                args.epochs,
                args.batch_size,
                args.keras_verbose,
            )

        results.append(metrics)
        current_df = save_metrics(results)
        print(pd.DataFrame([metrics]).to_string(index=False))
        print(f"Current best model: {current_df.iloc[0]['model']}")

    results_df = save_metrics(results)
    save_experiment_setup(
        args.models,
        args.epochs,
        args.batch_size,
        full_train_size,
        len(x_supervised_train),
        full_val_size,
        len(x_val),
        x_supervised_train.shape,
        x_val.shape,
        balance_method=args.balance_method,
        loss_function=args.loss_function,
    )
    write_summary_report(results_df)

    best_row = results_df.iloc[0]
    pd.DataFrame([best_row]).to_csv(REPORT_DIR / "deep_learning_best_model.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"Saved metrics to: {REPORT_DIR / 'deep_learning_metrics_validation.csv'}")
    print(f"Best model by attack_f1_score: {best_row['model']}")


if __name__ == "__main__":
    main()
