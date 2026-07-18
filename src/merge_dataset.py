from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Datasense-IIoT-2025" / "data" / "all_attack_benign_samples"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "cic_iiot_2025_merged.pkl.gz"


def collect_files() -> list[Path]:
    attack_files = sorted((DATA_ROOT / "attack_data").glob("*.csv"))
    benign_files = sorted((DATA_ROOT / "benign_data").glob("*.csv"))
    return attack_files + benign_files


def main() -> None:
    files = collect_files()
    frames = []

    for csv_file in files:
        df = pd.read_csv(csv_file)
        frames.append(df)
        print(f"Loaded {csv_file.name}: {df.shape}")

    merged_df = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_df.to_pickle(OUTPUT_PATH, compression="gzip")

    print(f"Merged shape: {merged_df.shape}")
    print(f"Saved merged dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
