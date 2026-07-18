from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "outputs" / "splits"
OUTPUT_DIR = ROOT / "models" / "final"
REPORT_DIR = ROOT / "reports" / "final"
MAX_PRUNING_ALPHAS = 25
MAX_PRUNING_SAMPLES = 60000


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train = pd.read_pickle(SPLIT_DIR / "X_train_raw.pkl.gz", compression="gzip")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").iloc[:, 0].astype("int32")
    x_val = pd.read_pickle(SPLIT_DIR / "X_val_raw.pkl.gz", compression="gzip")
    y_val = pd.read_csv(SPLIT_DIR / "y_val.csv").iloc[:, 0].astype("int32")
    return x_train, y_train, x_val, y_val


def evaluate_tree(model: DecisionTreeClassifier, x_val_scaled, y_val) -> dict[str, float]:
    y_pred = model.predict(x_val_scaled)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_val_scaled)[:, 1]
    else:
        y_score = y_pred.astype(float)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1_score": f1_score(y_val, y_pred, zero_division=0),
        "attack_f1_score": f1_score(y_val, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_score),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = load_training_data()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

    y_train_balanced_array = np.asarray(y_train_balanced)
    if len(y_train_balanced_array) > MAX_PRUNING_SAMPLES:
        rng = np.random.default_rng(42)
        sample_indexes = []
        per_class = MAX_PRUNING_SAMPLES // len(np.unique(y_train_balanced_array))
        for label in np.unique(y_train_balanced_array):
            label_indexes = np.flatnonzero(y_train_balanced_array == label)
            take = min(per_class, len(label_indexes))
            sample_indexes.extend(rng.choice(label_indexes, size=take, replace=False).tolist())
        sample_indexes = np.array(sorted(sample_indexes), dtype=int)
        x_pruning = x_train_balanced[sample_indexes]
        y_pruning = y_train_balanced_array[sample_indexes]
    else:
        x_pruning = x_train_balanced
        y_pruning = y_train_balanced_array

    base_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
    )
    pruning_path = base_model.cost_complexity_pruning_path(x_pruning, y_pruning)
    candidate_alphas = sorted(set(float(alpha) for alpha in pruning_path.ccp_alphas if alpha >= 0.0))
    if not candidate_alphas:
        candidate_alphas = [0.0]
    elif len(candidate_alphas) > MAX_PRUNING_ALPHAS:
        candidate_indexes = np.linspace(
            0,
            len(candidate_alphas) - 1,
            num=MAX_PRUNING_ALPHAS,
            dtype=int,
        )
        candidate_alphas = [candidate_alphas[index] for index in candidate_indexes]

    best_model = None
    best_metrics = None
    best_alpha = None
    for alpha in candidate_alphas:
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            ccp_alpha=alpha,
        )
        model.fit(x_pruning, y_pruning)
        metrics = evaluate_tree(model, x_val_scaled, y_val)
        if best_metrics is None or metrics["attack_f1_score"] > best_metrics["attack_f1_score"]:
            best_metrics = metrics
            best_alpha = alpha

    assert best_metrics is not None
    assert best_alpha is not None

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        ccp_alpha=best_alpha,
    )
    model.fit(x_train_balanced, y_train_balanced)
    best_metrics = evaluate_tree(model, x_val_scaled, y_val)

    bundle = {
        "model_name": "decision_tree",
        "feature_names": list(x_train.columns),
        "scaler": scaler,
        "classifier": model,
        "ccp_alpha": best_alpha,
        "train_shape": x_train.shape,
        "validation_shape": x_val.shape,
        "train_balanced_shape": x_train_balanced.shape,
        "selected_reason": "best overall model on the held-out test split",
        "selection_source": "reports/final/final_model_comparison.csv",
    }

    joblib.dump(bundle, OUTPUT_DIR / "best_model_bundle.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "best_model_scaler.joblib")
    joblib.dump(model, OUTPUT_DIR / "best_model_classifier.joblib")
    joblib.dump(model, OUTPUT_DIR / "best_model_pruned_classifier.joblib")

    pd.DataFrame(
        [
            {
                "model_name": "decision_tree",
                "artifact": "best_model_bundle.joblib",
                "reason": "Highest attack_f1_score on the held-out test split",
                "ccp_alpha": best_alpha,
                "validation_attack_f1_score": best_metrics["attack_f1_score"],
                "train_shape": str(x_train.shape),
                "validation_shape": str(x_val.shape),
                "balanced_train_shape": str(x_train_balanced.shape),
            }
        ]
    ).to_csv(REPORT_DIR / "best_model_manifest.csv", index=False)

    summary = {
        "best_model": "decision_tree",
        "artifact": str(OUTPUT_DIR / "best_model_bundle.joblib"),
        "scaler_artifact": str(OUTPUT_DIR / "best_model_scaler.joblib"),
        "classifier_artifact": str(OUTPUT_DIR / "best_model_classifier.joblib"),
        "pruned_classifier_artifact": str(OUTPUT_DIR / "best_model_pruned_classifier.joblib"),
        "ccp_alpha": best_alpha,
        "reason": "best overall test performance",
    }
    (REPORT_DIR / "best_model_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
