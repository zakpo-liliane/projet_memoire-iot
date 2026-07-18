from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DL_MODEL_DIR = ROOT / "models" / "deep_learning"
TUNING_MODEL_DIR = DL_MODEL_DIR / "tuning"
OUTPUT_DIR = ROOT / "models" / "edge"
REPORT_DIR = ROOT / "reports" / "edge"
EXPORT_PRIORITY = ["cnn", "autoencoder", "cnn_lstm_attention", "cnn_lstm", "lstm"]


def available_model_name() -> str:
    for model_name in EXPORT_PRIORITY:
        tuned_best = TUNING_MODEL_DIR / f"{model_name}_best.keras"
        base_model = DL_MODEL_DIR / f"{model_name}.keras"
        if tuned_best.exists() or base_model.exists():
            return model_name
    raise FileNotFoundError("No exportable deep model found.")


def load_keras_model(model_name: str) -> tf.keras.Model:
    tuned_best = TUNING_MODEL_DIR / f"{model_name}_best.keras"
    base_model = DL_MODEL_DIR / f"{model_name}.keras"
    if tuned_best.exists():
        return tf.keras.models.load_model(tuned_best)
    if base_model.exists():
        return tf.keras.models.load_model(base_model)
    raise FileNotFoundError(f"No Keras model found for {model_name}.")


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = available_model_name()
    model = load_keras_model(model_name)
    output_path = OUTPUT_DIR / f"{model_name}_quantized.tflite"
    export_tflite(model, output_path)

    summary = pd.DataFrame(
        [
            {
                "exported_model": model_name,
                "export_path": str(output_path),
                "quantization": "float16",
                "optimization": "DEFAULT",
                "note": "exportable edge model selected automatically from available artifacts",
            }
        ]
    )
    summary.to_csv(REPORT_DIR / "edge_deployment_summary.csv", index=False)
    (REPORT_DIR / "edge_deployment_summary.md").write_text(
        "# Edge Deployment Summary\n\n"
        f"- Exported model: `{model_name}`\n"
        f"- Exported quantized model: `{output_path}`\n"
        "- Quantization: float16 with default TFLite optimizations.\n"
        "- Exportable edge model selected automatically from the available deep-learning artifacts.\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
