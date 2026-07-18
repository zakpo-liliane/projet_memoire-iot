from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "models" / "baselines"
REPORT_DIR = ROOT / "reports" / "baselines"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train = pd.read_pickle(SPLIT_DIR / "X_train_scaled_smote.pkl.gz", compression="gzip")
    y_train = pd.read_csv(SPLIT_DIR / "y_train_smote.csv").iloc[:, 0]
    x_val = pd.read_pickle(SPLIT_DIR / "X_val_scaled.pkl.gz", compression="gzip")
    y_val = pd.read_csv(SPLIT_DIR / "y_val.csv").iloc[:, 0]
    return x_train, y_train, x_val, y_val


def evaluate_model(name: str, model, x_train, y_train, x_val, y_val) -> dict[str, float]:
    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start

    preds = model.predict(x_val)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1_score": f1_score(y_val, preds, zero_division=0),
        "train_seconds": train_seconds,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    return metrics


def main() -> None:
    x_train, y_train, x_val, y_val = load_data()

    models = {
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

    results: list[dict[str, float]] = []
    for name, model in models.items():
        print(f"Training {name}...")
        metrics = evaluate_model(name, model, x_train, y_train, x_val, y_val)
        results.append(metrics)
        print(metrics)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    results_df.to_csv(REPORT_DIR / "baseline_metrics_validation.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"Saved baseline metrics to: {REPORT_DIR / 'baseline_metrics_validation.csv'}")


if __name__ == "__main__":
    main()
