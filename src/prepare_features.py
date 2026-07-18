from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "cic_iiot_2025_cleaned.pkl.gz"
OUTPUT_PATH = ROOT / "outputs" / "cic_iiot_2025_prepared.pkl.gz"
MAPPINGS_PATH = ROOT / "outputs" / "label_mappings.csv"

LABEL_COLUMNS = ["label1", "label2", "label3", "label4"]
DROP_STRING_COLUMNS = [
    "device_name",
    "device_mac",
    "label_full",
    "timestamp",
    "log_data-types",
    "network_ips_all",
    "network_ips_dst",
    "network_ips_src",
    "network_macs_all",
    "network_macs_dst",
    "network_macs_src",
    "network_ports_all",
    "network_ports_dst",
    "network_ports_src",
    "network_protocols_all",
    "network_protocols_dst",
    "network_protocols_src",
]


def main() -> None:
    df = pd.read_pickle(INPUT_PATH, compression="gzip")
    print(f"Input shape: {df.shape}")

    # Derive a usable numeric duration feature from the cleaned timestamps.
    df["window_duration_seconds"] = (
        df["timestamp_end"] - df["timestamp_start"]
    ).dt.total_seconds()

    mapping_rows: list[dict[str, object]] = []
    for col in LABEL_COLUMNS:
        encoder = LabelEncoder()
        df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
        for idx, cls in enumerate(encoder.classes_):
            mapping_rows.append(
                {"label_column": col, "encoded_value": idx, "original_value": cls}
            )
        print(f"{col}: {len(encoder.classes_)} classes")

    # Keep the raw labels for interpretation, but drop high-cardinality string features.
    model_df = df.drop(columns=DROP_STRING_COLUMNS)

    # Absolute timestamps are dropped from modeling to avoid learning capture-session timing.
    model_df = model_df.drop(columns=["timestamp_start", "timestamp_end"])

    print(f"Prepared shape: {model_df.shape}")
    print(f"Numeric columns: {len(model_df.select_dtypes(include=['number']).columns)}")

    model_df.to_pickle(OUTPUT_PATH, compression="gzip")
    pd.DataFrame(mapping_rows).to_csv(MAPPINGS_PATH, index=False)

    print(f"Saved prepared dataset to: {OUTPUT_PATH}")
    print(f"Saved label mappings to: {MAPPINGS_PATH}")
    print("Note: scaling and SMOTE will be fit on the training split to avoid data leakage.")


if __name__ == "__main__":
    main()
