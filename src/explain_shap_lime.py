from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final" / "best_model_bundle.joblib"
SPLIT_DIR = ROOT / "outputs" / "splits"
OUTPUT_DIR = ROOT / "reports" / "explainability"
SAMPLE_SIZE = 1000


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_model_and_data():
    bundle = joblib.load(MODEL_PATH)
    x_train = pd.read_pickle(SPLIT_DIR / "X_train_raw.pkl.gz", compression="gzip")
    x_test = pd.read_pickle(SPLIT_DIR / "X_test_raw.pkl.gz", compression="gzip")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").iloc[:, 0].astype("int32")
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv").iloc[:, 0].astype("int32")
    feature_names = list(bundle["feature_names"])
    scaler = bundle["scaler"]
    classifier = bundle["classifier"]
    x_train_scaled = scaler.transform(x_train[feature_names])
    x_test_scaled = scaler.transform(x_test[feature_names])
    return classifier, feature_names, x_train_scaled, x_test_scaled, y_train, y_test


def run_shap(classifier, feature_names: list[str], x_test_scaled: np.ndarray) -> str:
    import shap

    sample = x_test_scaled[: min(SAMPLE_SIZE, len(x_test_scaled))]
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    shap_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(OUTPUT_DIR / "best_model_shap_importance.csv", index=False)

    top_df = shap_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1], color="#2F80ED")
    ax.set_title("Top SHAP importances for the best model")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_model_shap_importance.png", dpi=160)
    plt.close(fig)
    return "SHAP executed successfully."


def run_lime(
    classifier,
    feature_names: list[str],
    x_train_scaled: np.ndarray,
    x_test_scaled: np.ndarray,
) -> str:
    from lime.lime_tabular import LimeTabularExplainer

    class_names = ["attack", "benign"]
    explainer = LimeTabularExplainer(
        training_data=x_train_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        x_test_scaled[0],
        classifier.predict_proba,
        num_features=15,
    )
    lime_df = pd.DataFrame(
        explanation.as_list(),
        columns=["local_rule", "contribution"],
    )
    lime_df.to_csv(OUTPUT_DIR / "best_model_lime_example.csv", index=False)
    return "LIME executed successfully for one representative test instance."


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classifier, feature_names, x_train_scaled, x_test_scaled, _, _ = load_model_and_data()

    statuses = []
    if has_package("shap"):
        statuses.append({"method": "SHAP", "status": run_shap(classifier, feature_names, x_test_scaled)})
    else:
        statuses.append({"method": "SHAP", "status": "Not executed: package `shap` is not installed."})

    if has_package("lime"):
        statuses.append({"method": "LIME", "status": run_lime(classifier, feature_names, x_train_scaled, x_test_scaled)})
    else:
        statuses.append({"method": "LIME", "status": "Not executed: package `lime` is not installed."})

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(OUTPUT_DIR / "shap_lime_status.csv", index=False)
    lines = [
        "# SHAP/LIME Explainability Status",
        "",
        "This report records whether SHAP and LIME explanations were executed for the final binary IDS model.",
        "",
        "```csv",
        status_df.to_csv(index=False).strip(),
        "```",
    ]
    (OUTPUT_DIR / "shap_lime_status.md").write_text("\n".join(lines), encoding="utf-8")
    print(status_df.to_string(index=False))


if __name__ == "__main__":
    main()
