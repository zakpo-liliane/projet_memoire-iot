from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "cic_iiot_2025_prepared.pkl.gz"
REPORT_DIR = ROOT / "reports" / "eda"


def save_class_distribution(df: pd.DataFrame, column: str) -> None:
    counts = df[column].astype(str).value_counts().reset_index()
    counts.columns = [column, "count"]
    counts.to_csv(REPORT_DIR / f"{column}_distribution.csv", index=False)

    plt.figure(figsize=(10, 5))
    top_counts = counts.head(15)
    sns.barplot(data=top_counts, x="count", y=column, palette="Blues_r")
    plt.title(f"Distribution of {column} (top 15)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{column}_distribution.png", dpi=200)
    plt.close()


def save_numeric_summary(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_df = df[numeric_cols]
    numeric_df.describe().T.to_csv(REPORT_DIR / "numeric_summary.csv")

    variances = numeric_df.var().sort_values(ascending=False)
    top_features = variances.head(10).index.tolist()

    sample_df = numeric_df.sample(min(len(numeric_df), 100_000), random_state=42)
    sample_df[top_features].hist(figsize=(16, 10), bins=30)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_numeric_histograms.png", dpi=200)
    plt.close()

    corr = sample_df[top_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Top-Variance Features")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_feature_correlation_heatmap.png", dpi=200)
    plt.close()

    return top_features


def save_binary_target_plot(df: pd.DataFrame) -> None:
    counts = df["label1"].astype(str).value_counts().reset_index()
    counts.columns = ["label1", "count"]

    plt.figure(figsize=(6, 4))
    sns.barplot(data=counts, x="label1", y="count", palette="Set2")
    plt.title("Binary Target Distribution")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "label1_binary_distribution.png", dpi=200)
    plt.close()


def save_report(df: pd.DataFrame, top_features: list[str]) -> None:
    lines = []
    lines.append("# EDA Summary")
    lines.append("")
    lines.append(f"- Nombre de lignes : {len(df)}")
    lines.append(f"- Nombre de colonnes : {df.shape[1]}")
    lines.append(f"- Nombre de variables numeriques : {len(df.select_dtypes(include=['number']).columns)}")
    lines.append(f"- Nombre de variables categorielles : {len(df.select_dtypes(include=['category', 'object', 'string']).columns)}")
    lines.append(f"- Distribution binaire label1 : {df['label1'].astype(str).value_counts().to_dict()}")
    lines.append(f"- Top features a forte variance : {top_features}")
    lines.append("")
    lines.append("Fichiers generes :")
    lines.append("- label1_distribution.csv / .png")
    lines.append("- label2_distribution.csv / .png")
    lines.append("- label3_distribution.csv / .png")
    lines.append("- label4_distribution.csv / .png")
    lines.append("- numeric_summary.csv")
    lines.append("- top_numeric_histograms.png")
    lines.append("- top_feature_correlation_heatmap.png")
    lines.append("- label1_binary_distribution.png")

    (REPORT_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(INPUT_PATH, compression="gzip")
    print(f"Loaded dataset: {df.shape}")

    for column in ["label1", "label2", "label3", "label4"]:
        save_class_distribution(df, column)
        print(f"Saved distribution for {column}")

    save_binary_target_plot(df)
    top_features = save_numeric_summary(df)
    save_report(df, top_features)

    print(f"EDA outputs saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
