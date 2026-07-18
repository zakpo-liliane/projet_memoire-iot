from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "cic_iiot_2025_merged.pkl.gz"
OUTPUT_PATH = ROOT / "outputs" / "cic_iiot_2025_cleaned.pkl.gz"

TEXT_COLUMNS = [
    "device_name",
    "device_mac",
    "label_full",
    "label1",
    "label2",
    "label3",
    "label4",
]

DATETIME_COLUMNS = [
    "timestamp_start",
    "timestamp_end",
]


def main() -> None:
    df = pd.read_pickle(INPUT_PATH, compression="gzip")
    print(f"Initial shape: {df.shape}")

    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows found: {duplicate_count}")
    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)

    missing_count = int(df.isna().sum().sum())
    print(f"Total missing values found: {missing_count}")

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("category")

    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="raise")

    print("Dtypes after cleaning:")
    print(df.dtypes.astype(str).value_counts())

    df.to_pickle(OUTPUT_PATH, compression="gzip")
    print(f"Cleaned shape: {df.shape}")
    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
