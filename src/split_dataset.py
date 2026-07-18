from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "cic_iiot_2025_prepared.pkl.gz"
OUTPUT_DIR = ROOT / "outputs" / "splits"

TARGET_COLUMN = "label1_encoded"
LABEL_COLUMNS = [
    "label1",
    "label2",
    "label3",
    "label4",
    "label1_encoded",
    "label2_encoded",
    "label3_encoded",
    "label4_encoded",
]


def main() -> None:
    df = pd.read_pickle(INPUT_PATH, compression="gzip")

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=LABEL_COLUMNS)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_pickle(OUTPUT_DIR / "X_train_raw.pkl.gz", compression="gzip")
    X_val.to_pickle(OUTPUT_DIR / "X_val_raw.pkl.gz", compression="gzip")
    X_test.to_pickle(OUTPUT_DIR / "X_test_raw.pkl.gz", compression="gzip")

    X_train_scaled.to_pickle(OUTPUT_DIR / "X_train_scaled.pkl.gz", compression="gzip")
    X_val_scaled.to_pickle(OUTPUT_DIR / "X_val_scaled.pkl.gz", compression="gzip")
    X_test_scaled.to_pickle(OUTPUT_DIR / "X_test_scaled.pkl.gz", compression="gzip")

    pd.Series(y_train, name=TARGET_COLUMN).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    pd.Series(y_val, name=TARGET_COLUMN).to_csv(OUTPUT_DIR / "y_val.csv", index=False)
    pd.Series(y_test, name=TARGET_COLUMN).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    pd.DataFrame(X_train_balanced, columns=X_train.columns).to_pickle(
        OUTPUT_DIR / "X_train_scaled_smote.pkl.gz", compression="gzip"
    )
    pd.Series(y_train_balanced, name=TARGET_COLUMN).to_csv(
        OUTPUT_DIR / "y_train_smote.csv", index=False
    )

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(X_train),
                "attack_count": int((y_train == 0).sum()),
                "benign_count": int((y_train == 1).sum()),
            },
            {
                "split": "validation",
                "rows": len(X_val),
                "attack_count": int((y_val == 0).sum()),
                "benign_count": int((y_val == 1).sum()),
            },
            {
                "split": "test",
                "rows": len(X_test),
                "attack_count": int((y_test == 0).sum()),
                "benign_count": int((y_test == 1).sum()),
            },
            {
                "split": "train_smote",
                "rows": len(X_train_balanced),
                "attack_count": int((y_train_balanced == 0).sum()),
                "benign_count": int((y_train_balanced == 1).sum()),
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved split artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
