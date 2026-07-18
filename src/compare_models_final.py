from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TEST_METRICS_PATH = ROOT / "reports" / "evaluation" / "figures" / "all_model_metrics_test.csv"
OUTPUT_DIR = ROOT / "reports" / "final"


def load_metrics() -> pd.DataFrame:
    if not TEST_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing test metrics file: {TEST_METRICS_PATH}")
    return pd.read_csv(TEST_METRICS_PATH)


def select_best(df: pd.DataFrame) -> pd.DataFrame:
    sort_col = "attack_f1_score" if "attack_f1_score" in df.columns else "f1_score"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def split_families(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classical = df[df["model"].isin(["decision_tree", "random_forest", "linear_svm"])].copy()
    deep = df[df["model"].isin(["cnn", "lstm", "cnn_lstm", "cnn_lstm_attention", "autoencoder"])].copy()
    return classical, deep


def save_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = df.sort_values("attack_f1_score", ascending=False)
    sns.barplot(data=plot_df, x="attack_f1_score", y="model", color="#2F80ED", ax=ax)
    ax.set_title("Final model ranking on the test set")
    ax.set_xlabel("Attack F1-score")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_model_ranking_test.png", dpi=160)
    plt.close(fig)


def save_report(
    df: pd.DataFrame,
    best_overall: pd.Series,
    best_classical: pd.Series | None,
    best_deep: pd.Series | None,
) -> None:
    lines = [
        "# Step 12 Final Comparison",
        "",
        "## Best overall model",
        "",
        f"- Model: `{best_overall['model']}`",
        f"- Attack F1-score: `{best_overall['attack_f1_score']:.4f}`",
        f"- Accuracy: `{best_overall['accuracy']:.4f}`",
        f"- ROC-AUC: `{best_overall['roc_auc']:.4f}`",
        "",
        "## Best classical model",
        "",
        f"- Model: `{best_classical['model']}`" if best_classical is not None else "- Model: `N/A`",
        f"- Attack F1-score: `{best_classical['attack_f1_score']:.4f}`" if best_classical is not None else "- Attack F1-score: `N/A`",
        "",
        "## Best deep learning model",
        "",
        f"- Model: `{best_deep['model']}`" if best_deep is not None else "- Model: `N/A`",
        f"- Attack F1-score: `{best_deep['attack_f1_score']:.4f}`" if best_deep is not None else "- Attack F1-score: `N/A`",
        "",
        "## Full ranking",
        "",
        "```csv",
        df.round(4).to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        "The classical tree-based models outperform the deep learning models on the held-out test split.",
        "The decision tree is selected as the final best model for this project because it has the highest attack F1-score and the strongest overall test performance.",
    ]
    (OUTPUT_DIR / "final_model_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = load_metrics()
    ranking_df = select_best(metrics_df)
    classical_df, deep_df = split_families(ranking_df)

    best_overall = ranking_df.iloc[0]
    best_classical = classical_df.iloc[0] if not classical_df.empty else None
    best_deep = deep_df.iloc[0] if not deep_df.empty else None

    ranking_df.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)
    pd.DataFrame([best_overall]).to_csv(OUTPUT_DIR / "final_best_overall.csv", index=False)
    if best_classical is not None:
        pd.DataFrame([best_classical]).to_csv(OUTPUT_DIR / "final_best_classical.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "final_best_classical.csv", index=False)
    if best_deep is not None:
        pd.DataFrame([best_deep]).to_csv(OUTPUT_DIR / "final_best_deep_learning.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "final_best_deep_learning.csv", index=False)
    save_plot(ranking_df)
    save_report(ranking_df, best_overall, best_classical, best_deep)

    print(ranking_df.to_string(index=False))
    print(f"Best overall: {best_overall['model']}")


if __name__ == "__main__":
    main()
