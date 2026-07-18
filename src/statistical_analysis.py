from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, f_oneway, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = ROOT / "reports" / "evaluation" / "predictions"
CV_FOLDS_PATH = ROOT / "reports" / "baselines" / "baseline_cross_validation_folds.csv"
OUTPUT_DIR = ROOT / "reports" / "evaluation" / "statistical_tests"


def load_predictions() -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob("*_predictions_test.csv"))
    if not files:
        raise FileNotFoundError(
            f"No prediction files found in {PREDICTION_DIR}. Run evaluation first."
        )
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def compute_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_names = sorted(df["model"].unique())

    pivot_pred = df.pivot(index="row_id", columns="model", values="y_pred")
    pivot_true = df.groupby("row_id")["y_true"].first()

    for model_a, model_b in combinations(model_names, 2):
        a = pivot_pred[model_a].to_numpy()
        b = pivot_pred[model_b].to_numpy()
        y_true = pivot_true.to_numpy()

        correct_a = a == y_true
        correct_b = b == y_true
        b01 = int(np.sum(~correct_a & correct_b))
        b10 = int(np.sum(correct_a & ~correct_b))
        n = b01 + b10
        p_value = float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "b01": b01,
                "b10": b10,
                "n_discordant": n,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows).sort_values("p_value", ascending=True)


def load_cross_validation() -> pd.DataFrame:
    if not CV_FOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing cross-validation file: {CV_FOLDS_PATH}. Run cross_validate_baselines first."
        )
    return pd.read_csv(CV_FOLDS_PATH)


def compute_anova_and_wilcoxon(cv_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attack_groups = [group["attack_f1_score"].to_numpy() for _, group in cv_df.groupby("model")]
    anova_stat, anova_p = f_oneway(*attack_groups)

    anova_df = pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                "statistic": float(anova_stat),
                "p_value": float(anova_p),
                "groups": int(cv_df["model"].nunique()),
                "folds_per_group": int(cv_df.groupby("model").size().min()),
            }
        ]
    )

    rows: list[dict[str, object]] = []
    pivot = cv_df.pivot(index="fold", columns="model", values="attack_f1_score").sort_index()
    for model_a, model_b in combinations(sorted(cv_df["model"].unique()), 2):
        paired = pivot[[model_a, model_b]].dropna()
        a = paired[model_a].to_numpy()
        b = paired[model_b].to_numpy()
        if len(a) == 0 or len(b) == 0:
            stat, p_value = float("nan"), float("nan")
        else:
            stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "statistic": float(stat),
                "p_value": float(p_value),
                "n_pairs": int(min(len(a), len(b))),
            }
        )

    wilcoxon_df = pd.DataFrame(rows).sort_values("p_value", ascending=True)
    return anova_df, wilcoxon_df


def save_report(mcnemar_df: pd.DataFrame, anova_df: pd.DataFrame, wilcoxon_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mcnemar_df.to_csv(OUTPUT_DIR / "mcnemar_pairwise.csv", index=False)
    anova_df.to_csv(OUTPUT_DIR / "anova_summary.csv", index=False)
    wilcoxon_df.to_csv(OUTPUT_DIR / "wilcoxon_pairwise.csv", index=False)

    lines = [
        "# Statistical Analysis",
        "",
        "## McNemar test on test-set predictions",
        "",
        "```csv",
        mcnemar_df.round(4).to_csv(index=False).strip(),
        "```",
        "",
        "## One-way ANOVA on baseline cross-validation folds",
        "",
        "```csv",
        anova_df.round(4).to_csv(index=False).strip(),
        "```",
        "",
        "## Wilcoxon signed-rank tests on baseline cross-validation folds",
        "",
        "```csv",
        wilcoxon_df.round(4).to_csv(index=False).strip(),
        "```",
    ]
    (OUTPUT_DIR / "statistical_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    preds = load_predictions()
    mcnemar_df = compute_mcnemar(preds)

    if CV_FOLDS_PATH.exists():
        cv_df = load_cross_validation()
        anova_df, wilcoxon_df = compute_anova_and_wilcoxon(cv_df)
    else:
        anova_df = pd.DataFrame(
            [{"test": "one_way_anova", "statistic": np.nan, "p_value": np.nan, "groups": 0, "folds_per_group": 0}]
        )
        wilcoxon_df = pd.DataFrame(columns=["model_a", "model_b", "statistic", "p_value", "n_pairs"])

    save_report(mcnemar_df, anova_df, wilcoxon_df)
    print(mcnemar_df.to_string(index=False))


if __name__ == "__main__":
    main()
