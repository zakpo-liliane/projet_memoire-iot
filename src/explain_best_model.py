from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final" / "best_model_bundle.joblib"
SPLIT_DIR = ROOT / "outputs" / "splits"
OUTPUT_DIR = ROOT / "reports" / "explainability"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    x_test = pd.read_pickle(SPLIT_DIR / "X_test_raw.pkl.gz", compression="gzip")
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv").iloc[:, 0].astype("int32")
    return x_test, y_test


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    feature_names = list(bundle["feature_names"])
    scaler = bundle["scaler"]
    classifier = bundle["classifier"]

    x_test, y_test = load_data()
    x_test_aligned = x_test[feature_names].copy()
    x_test_scaled = scaler.transform(x_test_aligned)

    if hasattr(classifier, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "tree_importance": classifier.feature_importances_,
            }
        ).sort_values("tree_importance", ascending=False)
    else:
        importance_df = pd.DataFrame({"feature": feature_names, "tree_importance": 0.0})

    perm = permutation_importance(
        classifier,
        x_test_scaled,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1",
    )
    importance_df["permutation_importance_mean"] = perm.importances_mean
    importance_df["permutation_importance_std"] = perm.importances_std
    importance_df = importance_df.sort_values("permutation_importance_mean", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / "best_model_feature_importance.csv", index=False)

    top_df = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top_df, x="permutation_importance_mean", y="feature", ax=ax, color="#2F80ED")
    ax.set_title("Top feature importances for the best model")
    ax.set_xlabel("Permutation importance mean")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_model_feature_importance.png", dpi=160)
    fig.savefig(OUTPUT_DIR / "best_model_permutation_importance.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Explainability Summary",
        "",
        "The final decision tree model was analyzed using tree-based feature importance and permutation importance on the held-out test set.",
        "",
        "## Top features",
        "",
        "```csv",
        top_df.round(6).to_csv(index=False).strip(),
        "```",
    ]
    (OUTPUT_DIR / "best_model_explainability_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(top_df.to_string(index=False))


if __name__ == "__main__":
    main()
