from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "models" / "baselines"
REPORT_DIR = ROOT / "reports" / "baselines"

N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stratified cross-validation for the classical baseline models."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size used to make cross-validation faster.",
    )
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    x_train = pd.read_pickle(SPLIT_DIR / "X_train_raw.pkl.gz", compression="gzip")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").iloc[:, 0].astype("int32")
    return x_train, y_train


def stratified_sample(
    x_train: pd.DataFrame, y_train: pd.Series, sample_size: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size is None or sample_size >= len(y_train):
        return x_train, y_train

    index_frame = pd.DataFrame({"row_index": range(len(y_train)), "target": y_train.to_numpy()})
    sampled_index = (
        index_frame.groupby("target", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(1, round(sample_size * len(group) / len(index_frame))),
                random_state=42,
            )
        )
        .sample(frac=1.0, random_state=42)["row_index"]
        .to_numpy()
    )
    sampled_x = x_train.iloc[sampled_index].reset_index(drop=True)
    sampled_y = y_train.iloc[sampled_index].astype("int32").reset_index(drop=True)
    return sampled_x, sampled_y


def make_models() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
        "linear_svm": LinearSVC(
            random_state=42,
            max_iter=5000,
        ),
    }


def evaluate_split(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "attack_f1_score": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_data()
    x_train, y_train = stratified_sample(x_train, y_train, args.sample_size)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for model_name, model_template in make_models().items():
        fold_metrics: list[dict[str, float]] = []
        for fold_index, (train_idx, val_idx) in enumerate(cv.split(x_train, y_train), start=1):
            x_tr = x_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            x_va = x_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            scaler = StandardScaler()
            x_tr_scaled = scaler.fit_transform(x_tr)
            x_va_scaled = scaler.transform(x_va)

            smote = SMOTE(random_state=42)
            x_tr_bal, y_tr_bal = smote.fit_resample(x_tr_scaled, y_tr)

            model = clone(model_template)
            model.fit(x_tr_bal, y_tr_bal)
            y_pred = model.predict(x_va_scaled)

            metrics = evaluate_split(y_va, y_pred)
            metrics.update({"model": model_name, "fold": fold_index})
            fold_metrics.append(metrics)
            fold_rows.append(metrics)

        fold_df = pd.DataFrame(fold_metrics)
        summary_rows.append(
            {
                "model": model_name,
                "n_splits": N_SPLITS,
                "accuracy_mean": fold_df["accuracy"].mean(),
                "accuracy_std": fold_df["accuracy"].std(ddof=0),
                "precision_mean": fold_df["precision"].mean(),
                "recall_mean": fold_df["recall"].mean(),
                "f1_score_mean": fold_df["f1_score"].mean(),
                "attack_f1_score_mean": fold_df["attack_f1_score"].mean(),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("attack_f1_score_mean", ascending=False)

    fold_df.to_csv(REPORT_DIR / "baseline_cross_validation_folds.csv", index=False)
    summary_df.to_csv(REPORT_DIR / "baseline_cross_validation_summary.csv", index=False)

    report_lines = [
        "# Baseline Cross-Validation",
        "",
        "This report summarizes 5-fold stratified cross-validation on the training split.",
        f"Rows used: {len(y_train)}.",
        "",
        "```csv",
        summary_df.round(4).to_csv(index=False).strip(),
        "```",
    ]
    (REPORT_DIR / "baseline_cross_validation_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Saved cross-validation results to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
