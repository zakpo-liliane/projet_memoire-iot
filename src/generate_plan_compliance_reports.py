from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
PREPARED_PATH = ROOT / "outputs" / "cic_iiot_2025_prepared.pkl.gz"
SPLIT_DIR = ROOT / "outputs" / "splits"
PREDICTION_DIR = ROOT / "reports" / "evaluation" / "predictions"
BEST_MODEL_PATH = ROOT / "models" / "final" / "best_model_bundle.joblib"
OUTPUT_DIR = ROOT / "reports" / "plan_compliance"


def load_test_frame() -> pd.DataFrame:
    prepared = pd.read_pickle(PREPARED_PATH, compression="gzip")
    x_test = pd.read_pickle(SPLIT_DIR / "X_test_raw.pkl.gz", compression="gzip")
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv").iloc[:, 0].astype("int32")

    test_labels = prepared.loc[
        x_test.index,
        [
            "label1",
            "label2",
            "label3",
            "label4",
            "label1_encoded",
            "label2_encoded",
            "label3_encoded",
            "label4_encoded",
        ],
    ].reset_index(drop=False)
    test_labels = test_labels.rename(columns={"index": "source_index"})
    test_labels.insert(0, "row_id", range(len(test_labels)))
    test_labels["y_true"] = y_test.to_numpy()
    return test_labels


def load_best_predictions() -> pd.DataFrame:
    metrics_path = ROOT / "reports" / "evaluation" / "figures" / "best_model_test.csv"
    if metrics_path.exists():
        best_model = pd.read_csv(metrics_path).iloc[0]["model"]
    else:
        best_model = "decision_tree"

    pred_path = PREDICTION_DIR / f"{best_model}_predictions_test.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    return pd.read_csv(pred_path)


def binary_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["y_true"].astype("int32")
    y_pred = frame["y_pred"].astype("int32")
    return {
        "support": int(len(frame)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_benign": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_benign": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_benign": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_attack": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_attack": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def write_group_metrics(test_frame: pd.DataFrame, predictions: pd.DataFrame) -> None:
    joined = test_frame.merge(
        predictions[["row_id", "y_pred", "y_score"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    joined.to_csv(OUTPUT_DIR / "test_predictions_with_attack_labels.csv", index=False)

    for column, filename, title in [
        ("label2", "attack_family_performance_test.csv", "Attack-family recall on test set"),
        ("label3", "attack_type_performance_test.csv", "Attack-type recall on test set"),
        ("label4", "attack_scenario_performance_test.csv", "Attack-scenario recall on test set"),
    ]:
        rows = []
        for label, group in joined.groupby(column, dropna=False):
            metrics = binary_metrics(group)
            attack_group = group[group["y_true"] == 0]
            metrics.update(
                {
                    column: label,
                    "attack_support": int(len(attack_group)),
                    "attack_detected": int((attack_group["y_pred"] == 0).sum()),
                    "attack_missed_as_benign": int((attack_group["y_pred"] == 1).sum()),
                    "attack_detection_rate": float((attack_group["y_pred"] == 0).mean())
                    if len(attack_group)
                    else 0.0,
                }
            )
            rows.append(metrics)

        result = pd.DataFrame(rows)
        ordered_cols = [
            column,
            "support",
            "attack_support",
            "attack_detected",
            "attack_missed_as_benign",
            "attack_detection_rate",
            "accuracy",
            "precision_attack",
            "recall_attack",
            "f1_attack",
            "macro_f1",
        ]
        result = result[ordered_cols].sort_values(
            ["attack_support", "attack_detection_rate"],
            ascending=[False, True],
        )
        result.to_csv(OUTPUT_DIR / filename, index=False)

        plot_df = result[result["attack_support"] > 0].head(15)
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=plot_df, x="attack_detection_rate", y=column, ax=ax, color="#3B82F6")
            ax.set_title(title)
            ax.set_xlabel("Detection rate")
            ax.set_ylabel(column)
            ax.set_xlim(0, 1.05)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / filename.replace(".csv", ".png"), dpi=160)
            plt.close(fig)


def write_inference_timing() -> None:
    bundle = joblib.load(BEST_MODEL_PATH)
    x_test = pd.read_pickle(SPLIT_DIR / "X_test_raw.pkl.gz", compression="gzip")
    feature_names = list(bundle["feature_names"])
    scaler = bundle["scaler"]
    classifier = bundle["classifier"]
    x_aligned = x_test[feature_names].copy()
    x_scaled = scaler.transform(x_aligned)

    classifier.predict(x_scaled[:512])
    repeats = 5
    rows = []
    for repeat in range(1, repeats + 1):
        start = time.perf_counter()
        classifier.predict(x_scaled)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "repeat": repeat,
                "samples": len(x_scaled),
                "total_seconds": elapsed,
                "milliseconds_per_sample": (elapsed / len(x_scaled)) * 1000,
                "samples_per_second": len(x_scaled) / elapsed if elapsed > 0 else 0.0,
            }
        )

    timing = pd.DataFrame(rows)
    summary = timing.mean(numeric_only=True).to_frame().T
    summary.insert(0, "model", bundle.get("model_name", "decision_tree"))
    timing.to_csv(OUTPUT_DIR / "best_model_inference_timing_repeats.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "best_model_inference_timing_summary.csv", index=False)


def write_state_of_art_table() -> None:
    rows = [
        ["Chawla et al. (2002)", "SMOTE", "Desequilibre de classes", "Sur-echantillonnage synthetique", "Reference methodologique pour l'equilibrage"],
        ["Breiman (2001)", "Random Forest", "Classification supervisee", "Ensembles d'arbres", "Base comparative classique robuste"],
        ["Cortes & Vapnik (1995)", "SVM", "Classification supervisee", "Hyperplan separateur", "Baseline lineaire interpretable"],
        ["Hochreiter & Schmidhuber (1997)", "LSTM", "Series temporelles", "Memoire long terme", "Reference pour dependances sequentielles"],
        ["LeCun et al. (1998)", "CNN", "Extraction de motifs", "Convolutions", "Reference pour motifs locaux"],
        ["Kingma & Ba (2014)", "Adam", "Optimisation", "Optimiseur adaptatif", "Optimiseur des reseaux profonds"],
        ["Vaswani et al. (2017)", "Attention", "Modeles sequence", "Attention multi-tete", "Base du mecanisme d'attention"],
        ["CIC-IoT-2023", "Benchmark IoT", "IDS IoT", "Trafic IoT moderne", "Dataset de comparaison"],
        ["Edge-IIoTset", "Benchmark IIoT", "IDS edge/IIoT", "Trafic industriel et IoT", "Dataset de comparaison"],
        ["CIC-IIoT-2025 / DataSense", "Benchmark IIoT", "IDS industriel", "Scenarios attaque/benign", "Dataset principal du memoire"],
        ["Travaux IDS a signatures", "Regles statiques", "IDS traditionnel", "Detection d'attaques connues", "Limite sur zero-day"],
        ["Travaux ML tabulaire IDS", "Arbres/SVM/ensembles", "Classification IDS", "Features reseau structurees", "Bon compromis performance/interpretabilite"],
        ["Travaux DL IDS", "CNN/LSTM", "Detection comportementale", "Apprentissage de representations", "Couteux et sensible aux reglages"],
        ["Travaux autoencoder IDS", "Autoencoder", "Detection d'anomalies", "Apprentissage du trafic normal", "Utile sans labels exhaustifs"],
        ["Travaux XAI cybersecurite", "SHAP/LIME", "Explicabilite", "Contribution locale/globale", "Perspective pour renforcer l'interpretation"],
    ]
    df = pd.DataFrame(
        rows,
        columns=["reference", "approche", "domaine", "methodologie", "apport_ou_limite"],
    )
    df.to_csv(OUTPUT_DIR / "state_of_art_15_references.csv", index=False)


def write_compliance_summary() -> None:
    timing = pd.read_csv(OUTPUT_DIR / "best_model_inference_timing_summary.csv").iloc[0]
    family = pd.read_csv(OUTPUT_DIR / "attack_family_performance_test.csv")
    lines = [
        "# Plan Compliance Report",
        "",
        "Ce rapport complete les elements demandes dans le plan du memoire.",
        "",
        "## Analyses ajoutees",
        "",
        "- Analyse par famille d'attaque: `attack_family_performance_test.csv`.",
        "- Analyse par type d'attaque detaille: `attack_type_performance_test.csv`.",
        "- Analyse par scenario: `attack_scenario_performance_test.csv`.",
        "- Temps d'inference du meilleur modele: `best_model_inference_timing_summary.csv`.",
        "- Tableau de synthese critique de 15 references: `state_of_art_15_references.csv`.",
        "",
        "## Inference",
        "",
        f"- Modele mesure: `{timing['model']}`.",
        f"- Echantillons test: `{int(timing['samples'])}`.",
        f"- Temps moyen total: `{timing['total_seconds']:.6f}` seconde(s).",
        f"- Temps moyen par echantillon: `{timing['milliseconds_per_sample']:.6f}` ms.",
        f"- Debit moyen: `{timing['samples_per_second']:.2f}` echantillons/seconde.",
        "",
        "## Familles d'attaques principales",
        "",
        "```csv",
        family.head(10).round(4).to_csv(index=False).strip(),
        "```",
        "",
        "## Statut multiclasse",
        "",
        "Les labels multiclasse `label2`, `label3` et `label4` sont conserves et analyses par groupe. "
        "Le modele final reste volontairement binaire (`attack` contre `benign`) car l'objectif experimental principal est la detection d'intrusion. "
        "Un entrainement multiclasse complet peut etre ajoute comme extension, mais il ne doit pas etre presente comme deja realise.",
        "",
        "## Statut SHAP/LIME",
        "",
        "Les bibliotheques SHAP et LIME ne sont pas installees dans cet environnement. "
        "L'explicabilite realisee repose donc sur l'importance des variables et la permutation importance. "
        "SHAP/LIME sont conserves comme perspective ou limite methodologique.",
    ]
    (OUTPUT_DIR / "plan_compliance_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_frame = load_test_frame()
    predictions = load_best_predictions()
    write_group_metrics(test_frame, predictions)
    write_inference_timing()
    write_state_of_art_table()
    write_compliance_summary()
    print(f"Plan compliance reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
