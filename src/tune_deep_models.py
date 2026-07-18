from __future__ import annotations

import argparse
import itertools
import time
import sys
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
SRC_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "deep_learning" / "tuning"
REPORT_DIR = ROOT / "reports" / "deep_learning" / "tuning"

SEED = 42
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 512
MODEL_ORDER = ["cnn", "lstm", "autoencoder", "cnn_lstm", "cnn_lstm_attention"]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_deep_models import (  # noqa: E402
    balanced_sample,
    build_binary_metrics,
    build_autoencoder,
    compute_class_weights,
    load_supervised_train,
    load_train_raw,
    load_validation,
    supervised_loss,
    supervised_metrics,
    set_runtime,
    to_temporal_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the deep learning models for step 10 of the IDS thesis."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        choices=MODEL_ORDER,
        help="Models to tune, in the desired order.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sample-size", type=int, default=50000)
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
    parser.add_argument("--keras-verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count for TensorFlow.",
    )
    return parser.parse_args()


def set_seed() -> None:
    tf.keras.utils.set_random_seed(SEED)


def build_metrics_frame(
    model_name: str,
    config_name: str,
    params: dict[str, object],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    train_seconds: float,
) -> dict[str, object]:
    metrics = build_binary_metrics(model_name, y_true, y_scores, train_seconds)
    metrics["config_name"] = config_name
    for key, value in params.items():
        metrics[f"param_{key}"] = value
    return metrics


def save_confusion_matrix(report_dir: Path, model_name: str, config_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_attack", "true_benign"],
        columns=["pred_attack", "pred_benign"],
    )
    cm_df.to_csv(report_dir / f"{model_name}_{config_name}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name.upper()} {config_name} confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(report_dir / f"{model_name}_{config_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_roc_curve(report_dir: Path, model_name: str, config_name: str, y_true: np.ndarray, y_scores: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        report_dir / f"{model_name}_{config_name}_roc_curve.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} {config_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"{model_name.upper()} {config_name} ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / f"{model_name}_{config_name}_roc_curve.png", dpi=160)
    plt.close(fig)


def save_training_history(report_dir: Path, model_name: str, config_name: str, history: tf.keras.callbacks.History) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(report_dir / f"{model_name}_{config_name}_training_history.csv", index=False)


def compile_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def build_cnn(
    input_shape: tuple[int, int],
    params: dict[str, object],
    loss_name: str = "binary_crossentropy",
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(int(params["filters1"]), 5, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(int(params["filters2"]), 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(int(params["dense_units"]), activation="relu"),
            tf.keras.layers.Dropout(float(params["dropout"])),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=compile_optimizer(float(params["learning_rate"])),
        loss=supervised_loss(loss_name),
        metrics=supervised_metrics(),
    )
    return model


def build_lstm(
    input_shape: tuple[int, int],
    params: dict[str, object],
    loss_name: str = "binary_crossentropy",
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(int(params["lstm_units_1"]), return_sequences=True),
            tf.keras.layers.Dropout(float(params["dropout"])),
            tf.keras.layers.LSTM(int(params["lstm_units_2"])),
            tf.keras.layers.Dense(int(params["dense_units"]), activation="relu"),
            tf.keras.layers.Dropout(float(params["dropout"])),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=compile_optimizer(float(params["learning_rate"])),
        loss=supervised_loss(loss_name),
        metrics=supervised_metrics(),
    )
    return model


def build_cnn_lstm(
    input_shape: tuple[int, int],
    params: dict[str, object],
    loss_name: str = "binary_crossentropy",
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(int(params["filters1"]), 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(int(params["filters2"]), 3, padding="same", activation="relu"),
            tf.keras.layers.LSTM(int(params["lstm_units"])),
            tf.keras.layers.Dense(int(params["dense_units"]), activation="relu"),
            tf.keras.layers.Dropout(float(params["dropout"])),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=compile_optimizer(float(params["learning_rate"])),
        loss=supervised_loss(loss_name),
        metrics=supervised_metrics(),
    )
    return model


def build_cnn_lstm_attention(
    input_shape: tuple[int, int],
    params: dict[str, object],
    loss_name: str = "binary_crossentropy",
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(int(params["filters1"]), 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(int(params["filters2"]), 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LSTM(int(params["lstm_units"]), return_sequences=True)(x)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=int(params["attention_heads"]),
        key_dim=int(params["attention_key_dim"]),
    )(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(int(params["dense_units"]), activation="relu")(x)
    x = tf.keras.layers.Dropout(float(params["dropout"]))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_attention")
    model.compile(
        optimizer=compile_optimizer(float(params["learning_rate"])),
        loss=supervised_loss(loss_name),
        metrics=supervised_metrics(),
    )
    return model


def build_autoencoder_tuned(input_dim: int, params: dict[str, object]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(int(params["encoder_units_1"]), activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(int(params["encoder_units_2"]), activation="relu")(x)
    latent = tf.keras.layers.Dense(int(params["latent_dim"]), activation="relu")(x)
    x = tf.keras.layers.Dense(int(params["decoder_units_1"]), activation="relu")(latent)
    x = tf.keras.layers.Dense(int(params["decoder_units_2"]), activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    model.compile(
        optimizer=compile_optimizer(float(params["learning_rate"])),
        loss="mse",
    )
    return model


def search_space(model_name: str) -> list[dict[str, object]]:
    if model_name == "cnn":
        return [
            {"filters1": 16, "filters2": 32, "dense_units": 32, "dropout": 0.2, "learning_rate": 1e-3},
            {"filters1": 32, "filters2": 64, "dense_units": 64, "dropout": 0.3, "learning_rate": 1e-3},
        ]
    if model_name == "lstm":
        return [
            {"lstm_units_1": 32, "lstm_units_2": 16, "dense_units": 16, "dropout": 0.2, "learning_rate": 1e-3},
            {"lstm_units_1": 64, "lstm_units_2": 32, "dense_units": 32, "dropout": 0.3, "learning_rate": 1e-3},
        ]
    if model_name == "cnn_lstm":
        return [
            {"filters1": 16, "filters2": 32, "lstm_units": 16, "dense_units": 16, "dropout": 0.2, "learning_rate": 1e-3},
            {"filters1": 32, "filters2": 64, "lstm_units": 32, "dense_units": 32, "dropout": 0.3, "learning_rate": 1e-3},
        ]
    if model_name == "cnn_lstm_attention":
        return [
            {
                "filters1": 16,
                "filters2": 32,
                "lstm_units": 16,
                "attention_heads": 2,
                "attention_key_dim": 16,
                "dense_units": 16,
                "dropout": 0.2,
                "learning_rate": 1e-3,
            },
            {
                "filters1": 32,
                "filters2": 64,
                "lstm_units": 32,
                "attention_heads": 4,
                "attention_key_dim": 16,
                "dense_units": 32,
                "dropout": 0.3,
                "learning_rate": 1e-3,
            },
        ]
    return [
        {"encoder_units_1": 64, "encoder_units_2": 32, "latent_dim": 8, "decoder_units_1": 32, "decoder_units_2": 64, "learning_rate": 1e-3, "threshold_quantile": 0.90},
        {"encoder_units_1": 128, "encoder_units_2": 64, "latent_dim": 16, "decoder_units_1": 64, "decoder_units_2": 128, "learning_rate": 1e-3, "threshold_quantile": 0.95},
    ]


def common_callbacks(model_name: str, config_name: str) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / f"{model_name}_{config_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def evaluate_temporal_model(
    model_name: str,
    config_name: str,
    params: dict[str, object],
    builder,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    epochs: int,
    batch_size: int,
    keras_verbose: int,
    class_weight: dict[int, float] | None,
    loss_name: str,
) -> tuple[dict[str, object], tf.keras.Model]:
    x_train_seq = to_temporal_tensor(x_train)
    x_val_seq = to_temporal_tensor(x_val)
    model = builder((x_train_seq.shape[1], x_train_seq.shape[2]), params, loss_name)

    start = time.time()
    history = model.fit(
        x_train_seq,
        y_train.to_numpy(),
        validation_data=(x_val_seq, y_val.to_numpy()),
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbose,
        callbacks=common_callbacks(model_name, config_name),
        class_weight=class_weight,
    )
    train_seconds = time.time() - start

    y_scores = model.predict(x_val_seq, batch_size=batch_size, verbose=0).ravel()
    y_pred = (y_scores >= 0.5).astype("int32")
    metrics = build_metrics_frame(model_name, config_name, params, y_val.to_numpy(), y_scores, train_seconds)

    save_confusion_matrix(REPORT_DIR, model_name, config_name, y_val.to_numpy(), y_pred)
    save_roc_curve(REPORT_DIR, model_name, config_name, y_val.to_numpy(), y_scores)
    save_training_history(REPORT_DIR, model_name, config_name, history)
    return metrics, model


def evaluate_autoencoder(
    config_name: str,
    params: dict[str, object],
    x_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    sample_size: int | None,
    epochs: int,
    batch_size: int,
    keras_verbose: int,
) -> tuple[dict[str, object], tf.keras.Model]:
    benign_train = x_train_raw.loc[y_train_raw == 1]
    if sample_size is not None:
        benign_train = benign_train.sample(
            n=min(len(benign_train), max(sample_size // 2, 1)),
            random_state=SEED,
        )
    benign_val = x_val.loc[y_val == 1]

    model = build_autoencoder_tuned(x_train_raw.shape[1], params)
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
        callbacks=common_callbacks("autoencoder", config_name),
    )
    train_seconds = time.time() - start

    val_matrix = x_val.to_numpy(dtype=np.float32)
    reconstructed = model.predict(val_matrix, batch_size=batch_size, verbose=0)
    reconstruction_error = np.mean(np.square(val_matrix - reconstructed), axis=1)
    benign_errors = reconstruction_error[y_val.to_numpy() == 1]
    threshold = float(np.quantile(benign_errors, float(params["threshold_quantile"])))

    benign_scores = 1.0 - (
        (reconstruction_error - reconstruction_error.min())
        / (reconstruction_error.max() - reconstruction_error.min() + 1e-8)
    )
    y_pred = (reconstruction_error <= threshold).astype("int32")
    metrics = build_metrics_frame("autoencoder", config_name, params, y_val.to_numpy(), benign_scores, train_seconds)
    metrics["threshold"] = threshold

    save_confusion_matrix(REPORT_DIR, "autoencoder", config_name, y_val.to_numpy(), y_pred)
    save_roc_curve(REPORT_DIR, "autoencoder", config_name, y_val.to_numpy(), benign_scores)
    save_training_history(REPORT_DIR, "autoencoder", config_name, history)
    pd.DataFrame([{"config_name": config_name, "threshold": threshold, **params}]).to_csv(
        REPORT_DIR / f"autoencoder_{config_name}_threshold.csv",
        index=False,
    )
    return metrics, model


def save_best_model(report_dir: Path, model_dir: Path, model_name: str, config_name: str, model: tf.keras.Model) -> None:
    model.save(model_dir / f"{model_name}_best.keras")
    pd.DataFrame([{"model": model_name, "best_config": config_name}]).to_csv(
        report_dir / f"{model_name}_best_config.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    set_runtime(args.threads)
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_smote, y_train_smote = load_supervised_train()
    x_train_raw, y_train_raw = load_train_raw()
    x_val, y_val = load_validation()

    train_sample_size = None if args.full_data else args.sample_size
    val_sample_size = None if args.full_data else args.val_sample_size
    x_train_smote, y_train_smote = balanced_sample(x_train_smote, y_train_smote, train_sample_size)
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

    results: list[dict[str, object]] = []
    best_per_model: dict[str, tuple[str, dict[str, object], tf.keras.Model, float]] = {}

    for model_name in args.models:
        print(f"Tuning {model_name}...")
        for index, params in enumerate(search_space(model_name), start=1):
            config_name = f"cfg{index}"
            if model_name == "cnn":
                metrics, model = evaluate_temporal_model(
                    model_name,
                    config_name,
                    params,
                    build_cnn,
                    x_supervised_train,
                    y_supervised_train,
                    x_val,
                    y_val,
                    args.epochs,
                    args.batch_size,
                    args.keras_verbose,
                    class_weights,
                    args.loss_function,
                )
            elif model_name == "lstm":
                metrics, model = evaluate_temporal_model(
                    model_name,
                    config_name,
                    params,
                    build_lstm,
                    x_supervised_train,
                    y_supervised_train,
                    x_val,
                    y_val,
                    args.epochs,
                    args.batch_size,
                    args.keras_verbose,
                    class_weights,
                    args.loss_function,
                )
            elif model_name == "cnn_lstm":
                metrics, model = evaluate_temporal_model(
                    model_name,
                    config_name,
                    params,
                    build_cnn_lstm,
                    x_supervised_train,
                    y_supervised_train,
                    x_val,
                    y_val,
                    args.epochs,
                    args.batch_size,
                    args.keras_verbose,
                    class_weights,
                    args.loss_function,
                )
            elif model_name == "cnn_lstm_attention":
                metrics, model = evaluate_temporal_model(
                    model_name,
                    config_name,
                    params,
                    build_cnn_lstm_attention,
                    x_supervised_train,
                    y_supervised_train,
                    x_val,
                    y_val,
                    args.epochs,
                    args.batch_size,
                    args.keras_verbose,
                    class_weights,
                    args.loss_function,
                )
            else:
                metrics, model = evaluate_autoencoder(
                    config_name,
                    params,
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
            score = float(metrics["attack_f1_score"])
            current_best = best_per_model.get(model_name)
            if current_best is None or score > current_best[3]:
                best_per_model[model_name] = (config_name, params, model, score)
                save_best_model(REPORT_DIR, MODEL_DIR, model_name, config_name, model)

            print(pd.DataFrame([metrics]).to_string(index=False))
            print(f"Current best config for {model_name}: {best_per_model[model_name][0]}")

    results_df = pd.DataFrame(results)
    sort_col = "attack_f1_score" if "attack_f1_score" in results_df.columns else "f1_score"
    results_df = results_df.sort_values(sort_col, ascending=False)
    results_df.to_csv(REPORT_DIR / "hyperparameter_tuning_results.csv", index=False)

    best_rows = []
    for model_name, (config_name, params, _, score) in best_per_model.items():
        row = {
            "model": model_name,
            "best_config": config_name,
            "attack_f1_score": score,
        }
        row.update(params)
        best_rows.append(row)
    best_df = pd.DataFrame(best_rows).sort_values("attack_f1_score", ascending=False)
    best_df.to_csv(REPORT_DIR / "hyperparameter_tuning_best_configs.csv", index=False)

    summary_lines = [
        "# Step 10 Hyperparameter Tuning",
        "",
        "This step tunes the deep learning models on the validation split.",
        f"Balance method: {args.balance_method}. Loss function: {args.loss_function}.",
        "",
        "## Best configs",
        "",
        "```csv",
        best_df.round(4).to_csv(index=False).strip(),
        "```",
        "",
        "## Search results",
        "",
        "```csv",
        results_df.round(4).to_csv(index=False).strip(),
        "```",
    ]
    (REPORT_DIR / "hyperparameter_tuning_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(best_df.to_string(index=False))
    print(f"Saved tuning results to: {REPORT_DIR / 'hyperparameter_tuning_results.csv'}")
    print(f"Saved best configs to: {REPORT_DIR / 'hyperparameter_tuning_best_configs.csv'}")


if __name__ == "__main__":
    main()
