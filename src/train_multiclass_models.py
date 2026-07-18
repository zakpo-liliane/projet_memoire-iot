from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
PREPARED_PATH = ROOT / "outputs" / "cic_iiot_2025_prepared.pkl.gz"
MAPPING_PATH = ROOT / "outputs" / "label_mappings.csv"
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "models" / "multiclass"
REPORT_DIR = ROOT / "reports" / "multiclass"

TARGETS = ["label2_encoded", "label3_encoded", "label4_encoded"]
MODEL_NAMES = ["decision_tree", "random_forest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate multiclass IDS models on the full CIC-IIoT-2025 split."
    )
    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def load_features(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = pd.read_pickle(SPLIT_DIR / "X_train_scaled.pkl.gz", compression="gzip")
    x_eval = pd.read_pickle(SPLIT_DIR / f"X_{'val' if split == 'validation' else 'test'}_scaled.pkl.gz", compression="gzip")
    return x_train, x_eval


def load_target(prepared: pd.DataFrame, target: str, x_train: pd.DataFrame, x_eval: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    y_train = prepared.loc[x_train.index, target].astype("int32")
    y_eval = prepared.loc[x_eval.index, target].astype("int32")
    return y_train, y_eval


def label_names(target: str) -> dict[int, str]:
    label_column = target.replace("_encoded", "")
    mapping_df = pd.read_csv(MAPPING_PATH)
    target_mapping = mapping_df[mapping_df["label_column"] == label_column]
    return {
        int(row.encoded_value): str(row.original_value)
        for row in target_mapping.itertuples(index=False)
    }


def make_model(model_name: str):
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=42,
            max_depth=24,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
        )
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=42,
    )


def save_confusion_matrix(target: str, model_name: str, y_true: pd.Series, y_pred, names: dict[int, str]) -> None:
    labels = sorted(set(y_true.unique()).union(set(pd.Series(y_pred).unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    display_names = [names.get(int(label), str(label)) for label in labels]
    cm_df = pd.DataFrame(cm, index=display_names, columns=display_names)
    cm_df.to_csv(REPORT_DIR / f"{target}_{model_name}_confusion_matrix.csv")

    if len(labels) <= 15:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_df, cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{target} {model_name} confusion matrix")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"{target}_{model_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)


def evaluate_target_model(
    target: str,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> dict[str, object]:
    model = make_model(model_name)
    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start

    y_pred = model.predict(x_eval)
    names = label_names(target)
    report = classification_report(
        y_eval,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_DIR / f"{target}_{model_name}_classification_report.csv")
    save_confusion_matrix(target, model_name, y_eval, y_pred, names)

    model_path = MODEL_DIR / f"{target}_{model_name}.joblib"
    joblib.dump(model, model_path)

    return {
        "target": target,
        "model": model_name,
        "classes": int(y_eval.nunique()),
        "train_rows": int(len(x_train)),
        "eval_rows": int(len(x_eval)),
        "accuracy": accuracy_score(y_eval, y_pred),
        "macro_f1": f1_score(y_eval, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
        "train_seconds": train_seconds,
        "artifact": str(model_path),
    }


def write_summary(results_df: pd.DataFrame, split: str) -> None:
    lines = [
        "# Multiclass Evaluation",
        "",
        f"Evaluation split: `{split}`.",
        "The models use the full existing CIC-IIoT-2025 train split and the full evaluation split.",
        "",
        "```csv",
        results_df.round(4).to_csv(index=False).strip(),
        "```",
    ]
    (REPORT_DIR / "multiclass_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prepared = pd.read_pickle(PREPARED_PATH, compression="gzip")
    x_train, x_eval = load_features(args.split)

    results: list[dict[str, object]] = []
    for target in args.targets:
        y_train, y_eval = load_target(prepared, target, x_train, x_eval)
        for model_name in args.models:
            print(f"Training multiclass {model_name} for {target}...")
            metrics = evaluate_target_model(target, model_name, x_train, y_train, x_eval, y_eval)
            results.append(metrics)
            print(pd.DataFrame([metrics]).to_string(index=False))

    results_df = pd.DataFrame(results).sort_values(["target", "weighted_f1"], ascending=[True, False])
    results_df.to_csv(REPORT_DIR / "multiclass_metrics.csv", index=False)
    write_summary(results_df, args.split)

    print(results_df.to_string(index=False))
    print(f"Saved multiclass reports to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
