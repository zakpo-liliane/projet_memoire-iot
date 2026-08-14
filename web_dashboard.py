from __future__ import annotations

import argparse
import io
import json
import mimetypes
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import oracle_store


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "final" / "best_model_bundle.joblib"

MODEL_ORDER = [
    {
        "id": "decision_tree",
        "name": "Decision Tree",
        "type": "Classique",
        "role": "Modele interpretable retenu pour la detection finale attack/benign.",
    },
    {
        "id": "random_forest",
        "name": "Random Forest",
        "type": "Classique",
        "role": "Ensemble d'arbres utilise comme reference robuste.",
    },
    {
        "id": "linear_svm",
        "name": "Linear SVM",
        "type": "Classique",
        "role": "Separateur lineaire pour comparer une approche simple sur donnees tabulaires.",
    },
    {
        "id": "cnn",
        "name": "CNN",
        "type": "Deep learning",
        "role": "Extraction de motifs locaux sur le vecteur de caracteristiques.",
    },
    {
        "id": "lstm",
        "name": "LSTM",
        "type": "Deep learning",
        "role": "Apprentissage de dependances sequentielles entre caracteristiques.",
    },
    {
        "id": "autoencoder",
        "name": "Autoencoder",
        "type": "Deep learning",
        "role": "Detection d'anomalies a partir de l'erreur de reconstruction.",
    },
    {
        "id": "cnn_lstm",
        "name": "CNN + LSTM",
        "type": "Deep learning",
        "role": "Modele hybride combinant convolution et memoire sequentielle.",
    },
    {
        "id": "cnn_lstm_attention",
        "name": "CNN + LSTM + Attention",
        "type": "Deep learning",
        "role": "Extension hybride avec mecanisme d'attention multi-tete.",
    },
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def image_url(path: str) -> str:
    return f"/image?path={path}"


def model_asset_paths(model_id: str) -> dict[str, str | None]:
    baseline = model_id in {"decision_tree", "random_forest", "linear_svm"}
    eval_group = "baselines" if baseline else "deep_learning"
    validation_png = REPORTS / "figures" / f"{model_id}_confusion_matrix_validation.png"
    test_cm_png = REPORTS / "evaluation" / "figures" / f"{model_id}_confusion_matrix_test.png"
    test_roc_png = REPORTS / "evaluation" / "figures" / f"{model_id}_roc_curve_test.png"
    files = {
        "validation_confusion": f"reports/figures/{model_id}_confusion_matrix_validation.png"
        if validation_png.exists()
        else None,
        "test_confusion": f"reports/evaluation/figures/{model_id}_confusion_matrix_test.png"
        if test_cm_png.exists()
        else None,
        "test_roc": f"reports/evaluation/figures/{model_id}_roc_curve_test.png" if test_roc_png.exists() else None,
        "test_confusion_csv": f"reports/evaluation/{eval_group}/{model_id}_confusion_matrix_test.csv",
        "test_roc_csv": f"reports/evaluation/{eval_group}/{model_id}_roc_curve_test.csv",
        "predictions_csv": f"reports/evaluation/predictions/{model_id}_predictions_test.csv",
    }
    return files


def expected_feature_names() -> list[str]:
    if not MODEL_PATH.exists():
        return []
    try:
        bundle = joblib.load(MODEL_PATH)
        return list(bundle.get("feature_names", []))
    except Exception:
        return []


def validate_prediction_columns(columns: list[str], feature_names: list[str]) -> dict[str, object]:
    ignored = {
        "label1",
        "label2",
        "label3",
        "label4",
        "label1_encoded",
        "label2_encoded",
        "label3_encoded",
        "label4_encoded",
        "label_full",
        "timestamp",
        "timestamp_start",
        "timestamp_end",
        "target",
        "class",
        "label",
        "device_name",
        "device_mac",
    }
    provided = set(columns)
    expected = set(feature_names)
    missing = sorted(expected - provided)
    extra = sorted(col for col in provided - expected if col not in ignored)
    present = len(expected & provided)
    compatible = present >= max(1, int(len(expected) * 0.75))
    return {
        "compatible": compatible,
        "expected_count": len(feature_names),
        "present_count": present,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_preview": missing[:8],
        "extra_preview": extra[:8],
    }


def load_dashboard_data() -> dict[str, object]:
    metrics_df = read_csv(REPORTS / "final" / "final_model_comparison.csv")
    split_df = read_csv(OUTPUTS / "splits" / "split_summary.csv")
    label1_df = read_csv(REPORTS / "eda" / "label1_distribution.csv")
    label2_df = read_csv(REPORTS / "eda" / "label2_distribution.csv")
    importance_df = read_csv(REPORTS / "explainability" / "best_model_feature_importance.csv")
    family_df = read_csv(REPORTS / "plan_compliance" / "attack_family_performance_test.csv")
    timing_df = read_csv(REPORTS / "plan_compliance" / "best_model_inference_timing_summary.csv")

    metrics = {row["model"]: row.to_dict() for _, row in metrics_df.iterrows()} if not metrics_df.empty else {}
    ordered_models = []
    for index, item in enumerate(MODEL_ORDER, start=1):
        row = metrics.get(item["id"], {})
        ordered_models.append(
            {
                **item,
                "order": index,
                "accuracy": safe_float(row.get("accuracy")),
                "precision": safe_float(row.get("precision")),
                "recall": safe_float(row.get("recall")),
                "f1_score": safe_float(row.get("f1_score")),
                "attack_f1_score": safe_float(row.get("attack_f1_score")),
                "roc_auc": safe_float(row.get("roc_auc")),
                "assets": model_asset_paths(item["id"]),
            }
        )

    best = next((m for m in ordered_models if m["id"] == "decision_tree"), ordered_models[0])
    best_deep = max([m for m in ordered_models if m["type"] == "Deep learning"], key=lambda m: m["attack_f1_score"])
    split_total = 0
    if not split_df.empty and "rows" in split_df.columns:
        source_splits = split_df
        if "split" in split_df.columns:
            source_splits = split_df[~split_df["split"].astype(str).str.lower().eq("train_smote")]
        split_total = int(source_splits["rows"].sum())
    test_row = split_df.loc[split_df["split"].astype(str).str.lower().eq("test")].iloc[0].to_dict() if not split_df.empty and "split" in split_df.columns and (split_df["split"].astype(str).str.lower() == "test").any() else {}
    timing_row = timing_df.iloc[0].to_dict() if not timing_df.empty else {}

    return {
        "summary": {
            "project": "Detection d'intrusion dans les reseaux IIoT",
            "dataset": "CIC-IIoT-2025",
            "models_count": len(ordered_models),
            "best_model": best["name"],
            "accuracy": best["accuracy"],
            "attack_f1_score": best["attack_f1_score"],
            "status": "Modele pret pour la prediction",
            "best_deep_model": best_deep["name"],
            "total_rows": split_total,
            "test_rows": int(safe_float(test_row.get("rows"), 0)),
            "test_attacks": int(safe_float(test_row.get("attack_count"), 0)),
            "test_benign": int(safe_float(test_row.get("benign_count"), 0)),
            "latency_ms": safe_float(timing_row.get("milliseconds_per_sample"), 0),
            "samples_per_second": safe_float(timing_row.get("samples_per_second"), 0),
        },
        "models": ordered_models,
        "ranking": sorted(ordered_models, key=lambda row: row["attack_f1_score"], reverse=True),
        "split": split_df.to_dict(orient="records"),
        "label1": label1_df.to_dict(orient="records"),
        "label2": label2_df.to_dict(orient="records")[:12],
        "importance": importance_df.head(12).to_dict(orient="records"),
        "attack_family": family_df.head(12).to_dict(orient="records"),
        "images": {
            "label1": image_url("reports/eda/label1_binary_distribution.png"),
            "correlation": image_url("reports/eda/top_feature_correlation_heatmap.png"),
            "ranking": image_url("reports/final/final_model_ranking_test.png"),
            "best_confusion": image_url("reports/evaluation/figures/decision_tree_confusion_matrix_test.png"),
            "best_roc": image_url("reports/evaluation/figures/decision_tree_roc_curve_test.png"),
            "feature_importance": image_url("reports/explainability/best_model_feature_importance.png"),
            "permutation": image_url("reports/explainability/best_model_permutation_importance.png"),
        },
        "oracle": oracle_store.status(),
    }


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in [
        "label1",
        "label2",
        "label3",
        "label4",
        "label1_encoded",
        "label2_encoded",
        "label3_encoded",
        "label4_encoded",
        "label_full",
        "timestamp",
        "timestamp_start",
        "timestamp_end",
        "target",
        "class",
        "label",
    ]:
        if col in aligned.columns:
            aligned = aligned.drop(columns=[col])

    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0.0

    extra_cols = [col for col in aligned.columns if col not in feature_names]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)

    aligned = aligned[feature_names]
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def run_prediction(raw_df: pd.DataFrame) -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele final introuvable: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    feature_names = list(bundle["feature_names"])
    features = align_features(raw_df, feature_names)
    x_scaled = bundle["scaler"].transform(features)
    classifier = bundle["classifier"]
    y_pred = classifier.predict(x_scaled).astype("int32")

    if hasattr(classifier, "predict_proba"):
        benign_probability = classifier.predict_proba(x_scaled)[:, 1].astype(float)
    else:
        benign_probability = y_pred.astype(float)

    attack_probability = 1.0 - benign_probability
    attack_count = int((y_pred == 0).sum())
    benign_count = int((y_pred == 1).sum())
    alert_rate = float(attack_count / max(len(y_pred), 1))
    predictions = pd.DataFrame(
        {
            "ligne": np.arange(1, len(y_pred) + 1),
            "classe_predite": np.where(y_pred == 0, "Attaque", "Benin"),
            "probabilite_attaque": attack_probability,
            "probabilite_benin": benign_probability,
        }
    )

    label_col = None
    for candidate in ["label1_encoded", "label1", "target", "class", "label"]:
        if candidate in raw_df.columns:
            label_col = candidate
            break

    metrics = None
    if label_col:
        y_true = raw_df[label_col]
        if label_col == "label1" or y_true.dtype == object:
            y_true = y_true.astype(str).str.lower().str.strip().map({"attack": 0, "attacking": 0, "benign": 1, "normal": 1})
        y_true = pd.to_numeric(y_true, errors="coerce")
        valid = y_true.notna()
        if valid.any():
            yt = y_true.loc[valid].astype("int32").to_numpy()
            yp = y_pred[valid.to_numpy()]
            ap = attack_probability[valid.to_numpy()]
            metrics = {
                "accuracy": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1_score": f1_score(yt, yp, zero_division=0),
                "attack_precision": precision_score(yt, yp, pos_label=0, zero_division=0),
                "attack_recall": recall_score(yt, yp, pos_label=0, zero_division=0),
                "attack_f1_score": f1_score(yt, yp, pos_label=0, zero_division=0),
            }
            if len(set(yt.tolist())) == 2:
                metrics["attack_roc_auc"] = roc_auc_score((yt == 0).astype("int32"), ap)

    return {
        "upload_id": str(uuid.uuid4()),
        "model": bundle.get("model_name", "decision_tree"),
        "samples": int(len(raw_df)),
        "attack_count": attack_count,
        "benign_count": benign_count,
        "alert_rate": alert_rate,
        "status": "danger" if alert_rate >= 0.20 else "warning" if alert_rate > 0 else "safe",
        "metrics": metrics,
        "predictions": predictions.head(100).to_dict(orient="records"),
        "_predictions_df": predictions,
    }


def save_prediction_to_oracle(result: dict[str, object], source: str, filename: str) -> dict[str, object]:
    predictions = result.get("_predictions_df")
    if not isinstance(predictions, pd.DataFrame):
        return {"saved": False, "message": "Aucune prediction a sauvegarder."}
    try:
        return oracle_store.save_analysis(
            upload_id=str(result["upload_id"]),
            source=source,
            filename=filename,
            result=result,
            predictions=predictions,
        )
    except Exception as exc:
        return {"saved": False, "message": f"Prediction faite, mais sauvegarde Oracle impossible: {exc}"}


def public_prediction_result(result: dict[str, object], oracle_result: dict[str, object]) -> dict[str, object]:
    public = {key: value for key, value in result.items() if not key.startswith("_")}
    public["oracle"] = oracle_result
    public["validation"] = {
        "compatible": True,
        "expected_count": len(expected_feature_names()),
        "present_count": len(expected_feature_names()),
        "missing_count": 0,
        "extra_count": 0,
        "missing_preview": [],
        "extra_preview": [],
    }
    public["report"] = {
        "id": public["upload_id"],
        "source": "demo",
        "model": public["model"],
        "samples": public["samples"],
        "attack_count": public["attack_count"],
        "benign_count": public["benign_count"],
        "alert_rate": public["alert_rate"],
        "status": public["status"],
    }
    return public


def parse_multipart(body: bytes, content_type: str) -> dict[str, list[dict[str, object]]]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Requete multipart invalide.")
    boundary = ("--" + content_type.split(marker, 1)[1].split(";", 1)[0].strip()).encode()
    fields: dict[str, list[dict[str, object]]] = {}
    for part in body.split(boundary):
        part = part.strip()
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        header, payload = part.split(b"\r\n\r\n", 1)
        payload = payload.removesuffix(b"\r\n--").removesuffix(b"\r\n")
        header_text = header.decode("utf-8", errors="ignore")
        if 'name="' not in header_text:
            continue
        name = header_text.split('name="', 1)[1].split('"', 1)[0]
        filename = ""
        if 'filename="' in header_text:
            filename = header_text.split('filename="', 1)[1].split('"', 1)[0]
        fields.setdefault(name, []).append({"filename": filename, "content": payload})
    return fields


def predict_uploaded_files(files: list[dict[str, object]]) -> dict[str, object]:
    csv_files = [
        item for item in files
        if str(item.get("filename", "")).lower().endswith(".csv")
    ]
    if not csv_files:
        raise ValueError("Aucun fichier CSV trouve. Charge un CSV ou un dossier contenant des CSV.")

    summaries = []
    preview_rows = []
    total_samples = 0
    total_attacks = 0
    total_benign = 0
    oracle_saved = 0
    oracle_errors = []
    validations = []
    feature_names = expected_feature_names()

    for item in csv_files:
        filename = str(item.get("filename") or "uploaded_network_data.csv")
        content = item["content"]
        if not isinstance(content, bytes):
            continue
        raw_df = pd.read_csv(io.BytesIO(content))
        validation = validate_prediction_columns(list(raw_df.columns), feature_names)
        validations.append({"filename": filename, **validation})
        result = run_prediction(raw_df)
        oracle_result = save_prediction_to_oracle(result, source="upload", filename=filename)
        if oracle_result.get("saved"):
            oracle_saved += 1
        elif oracle_result.get("message"):
            oracle_errors.append(str(oracle_result["message"]))

        total_samples += int(result["samples"])
        total_attacks += int(result["attack_count"])
        total_benign += int(result["benign_count"])
        summaries.append(
            {
                "filename": filename,
                "samples": int(result["samples"]),
                "attack_count": int(result["attack_count"]),
                "benign_count": int(result["benign_count"]),
                "alert_rate": float(result["alert_rate"]),
                "status": str(result["status"]),
                "oracle_saved": bool(oracle_result.get("saved")),
                "compatible": bool(validation["compatible"]),
                "features_present": int(validation["present_count"]),
                "features_expected": int(validation["expected_count"]),
                "missing_columns": int(validation["missing_count"]),
            }
        )
        for row in result["predictions"][: max(0, 100 - len(preview_rows))]:
            row = dict(row)
            row["fichier"] = filename
            preview_rows.append(row)

    alert_rate = total_attacks / max(total_samples, 1)
    return {
        "upload_id": str(uuid.uuid4()),
        "model": "decision_tree",
        "samples": total_samples,
        "attack_count": total_attacks,
        "benign_count": total_benign,
        "alert_rate": alert_rate,
        "status": "danger" if alert_rate >= 0.20 else "warning" if alert_rate > 0 else "safe",
        "metrics": None,
        "predictions": preview_rows,
        "files": summaries,
        "validation": {
            "compatible": all(item["compatible"] for item in validations) if validations else False,
            "files": validations,
        },
        "report": {
            "id": str(uuid.uuid4()),
            "source": "upload",
            "model": "decision_tree",
            "files": len(csv_files),
            "samples": total_samples,
            "attack_count": total_attacks,
            "benign_count": total_benign,
            "alert_rate": alert_rate,
            "status": "danger" if alert_rate >= 0.20 else "warning" if alert_rate > 0 else "safe",
            "oracle_saved_files": oracle_saved,
        },
        "oracle": {
            "saved": oracle_saved == len(csv_files),
            "saved_files": oracle_saved,
            "total_files": len(csv_files),
            "message": f"{oracle_saved}/{len(csv_files)} fichier(s) sauvegarde(s) dans Oracle."
            if not oracle_errors else oracle_errors[0],
        },
    }


def html_page() -> str:
    return r"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard IDS IIoT</title>
  <style>
    :root {
      --bg: #0b111c;
      --panel: #121a29;
      --panel-2: rgba(23, 33, 51, 0.78);
      --ink: #e7ecf4;
      --muted: #7f8ca6;
      --line: #25324a;
      --blue: #45d2c4;
      --teal: #45d2c4;
      --accent-rgb: 69, 210, 196;
      --violet: #9c8cf5;
      --red: #f2555c;
      --amber: #f2a93b;
      --green: #45d2c4;
      --shadow: 0 14px 36px rgba(0, 0, 0, 0.24);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(rgba(var(--accent-rgb), 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(var(--accent-rgb), 0.035) 1px, transparent 1px),
        var(--bg);
      background-size: 32px 32px;
    }
    body.mode-light {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-2: rgba(245, 248, 252, 0.92);
      --ink: #132033;
      --muted: #59677d;
      --line: #d7e0ee;
      --shadow: 0 14px 36px rgba(22, 34, 51, 0.10);
    }
    .app { min-height: 100vh; display: grid; grid-template-columns: 260px 1fr; }
    aside {
      background: linear-gradient(180deg, #0d1420 0%, rgba(13,20,32,0.96) 100%);
      color: #fff;
      padding: 24px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
    }
    .brand { font-size: 20px; font-weight: 750; line-height: 1.2; margin-bottom: 6px; color: var(--blue); }
    .brand-sub { color: #b7c0d1; font-size: 13px; margin-bottom: 24px; }
    nav button {
      width: 100%;
      border: 0;
      background: transparent;
      color: #d5dbea;
      text-align: left;
      padding: 11px 12px;
      border-radius: 6px;
      font-size: 14px;
      cursor: pointer;
      margin: 2px 0;
    }
    nav button.active, nav button:hover { background: rgba(var(--accent-rgb),0.10); color: #fff; border: 1px solid rgba(var(--accent-rgb),0.25); }
    .session-panel { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--line); }
    .session-user { color: var(--muted); font-size: 12px; font-family: "Consolas", monospace; margin-bottom: 10px; }
    .sidebar-dropdown { display: none; margin: 4px 0 10px 12px; padding-left: 10px; border-left: 1px solid var(--line); }
    .sidebar-dropdown.open { display: grid; gap: 2px; }
    .sidebar-dropdown button {
      padding: 9px 10px;
      font-size: 13px;
      color: var(--muted);
      border: 0;
      background: transparent;
    }
    .sidebar-dropdown button:hover { color: var(--ink); background: rgba(var(--accent-rgb),0.08); }
    .settings-toggle::after { content: "v"; float: right; font-family: "Consolas", monospace; color: var(--muted); }
    .settings-toggle.open::after { content: "^"; }
    .settings-actions { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; }
    .settings-actions select {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 12px;
      min-height: 42px;
      font-size: 14px;
    }
    .demo-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .demo-step {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 14px;
      min-height: 118px;
    }
    .demo-step span {
      display: inline-grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: rgba(var(--accent-rgb), .12);
      color: var(--blue);
      font-weight: 800;
      margin-bottom: 10px;
    }
    .demo-step strong { display: block; margin-bottom: 6px; }
    .scope-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .scope-card {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 16px;
      min-height: 210px;
    }
    .scope-card ul {
      margin: 12px 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.75;
    }
    .scope-badge {
      display: inline-flex;
      align-items: center;
      border: 1px solid rgba(var(--accent-rgb), .35);
      color: var(--blue);
      background: rgba(var(--accent-rgb), .08);
      border-radius: 999px;
      padding: 6px 10px;
      font-family: "Consolas", monospace;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 12px;
    }
    .scope-badge.warning { color: var(--amber); border-color: rgba(242,169,59,.38); background: rgba(242,169,59,.10); }
    .scope-badge.future { color: var(--green); border-color: rgba(87,213,122,.35); background: rgba(87,213,122,.10); }
    .arch-flow {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }
    .arch-node {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 16px;
      min-height: 138px;
      position: relative;
    }
    .arch-node:not(:last-child)::after {
      content: ">";
      position: absolute;
      right: -11px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--blue);
      font-family: "Consolas", monospace;
      font-weight: 800;
    }
    .arch-node .metric-label { text-transform: uppercase; font-family: "Consolas", monospace; font-size: 11px; }
    main { padding: 28px; max-width: 1440px; width: 100%; }
    section { display: none; }
    section.active { display: block; }
    h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: 0; }
    h2 { font-size: 20px; margin: 0 0 14px; letter-spacing: 0; }
    h3 { font-size: 16px; margin: 0 0 10px; letter-spacing: 0; }
    p { color: var(--muted); line-height: 1.5; }
    .lead { margin: 0 0 22px; max-width: 920px; }
    .grid { display: grid; gap: 16px; }
    .cards { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .card, .panel {
      background: var(--panel);
      border: 1px solid rgba(37, 50, 74, 0.85);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    body.mode-light .card, body.mode-light .panel { border-color: var(--line); }
    body.mode-light aside { background: #ffffff; color: var(--ink); border-right: 1px solid var(--line); }
    body.mode-light nav button { color: var(--ink); }
    body.mode-light .brand-sub { color: var(--muted); }
    body.mode-light .alert-main strong,
    body.mode-light .pipeline-stat strong,
    body.mode-light .model-title { color: var(--ink); }
    body.mode-light .pipeline-node rect { fill: #f8fbff; stroke: var(--line); }
    body.mode-light .pipeline-node text { fill: var(--ink); }
    body.mode-light .bar-track,
    body.mode-light .rank-track { background: #e6edf7; }
    body.mode-light th { background: #eef3fa; color: var(--ink); }
    .metric-label { color: var(--muted); font-size: 13px; margin-bottom: 7px; }
    .metric-value { font-size: 26px; font-weight: 760; }
    .kpi-card { min-height: 112px; position: relative; overflow: hidden; }
    .kpi-card::before {
      content: "";
      position: absolute;
      inset: 0;
      border-top: 1px solid rgba(var(--accent-rgb),0.22);
      pointer-events: none;
    }
    .kpi-card .metric-label {
      font-family: "Consolas", monospace;
      font-size: 11px;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .kpi-card .metric-value { font-size: 32px; color: #f8fbff; }
    .kpi-trend { color: var(--blue); font-family: "Consolas", monospace; font-size: 12px; margin-top: 8px; }
    .dash-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid rgba(37,50,74,.8);
      margin-bottom: 22px;
      padding-bottom: 18px;
    }
    .dash-kicker {
      color: var(--muted);
      font-family: "Consolas", monospace;
      font-size: 11px;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }
    .system-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(var(--accent-rgb),.35);
      background: rgba(var(--accent-rgb),.08);
      color: var(--blue);
      border-radius: 999px;
      padding: 8px 14px;
      font-family: "Consolas", monospace;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .system-pill::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--blue);
      box-shadow: 0 0 12px rgba(var(--accent-rgb),.85);
    }
    .sentinel-grid { grid-template-columns: 1.25fr .82fr; margin-top: 16px; }
    .panel-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel h2 { font-size: 16px; }
    .mono-sub {
      color: var(--muted);
      font-family: "Consolas", monospace;
      font-size: 11px;
      margin-top: -7px;
    }
    .traffic-svg { width: 100%; height: 220px; display: block; }
    .traffic-line, .traffic-area, .attack-line, .attack-point { transition: all .45s ease; }
    .traffic-pulse { animation: trafficPulse 1.8s ease-in-out infinite; }
    @keyframes trafficPulse { 0%, 100% { opacity: .95; transform: scale(1); } 50% { opacity: .55; transform: scale(1.35); } }
    .alert-feed { display: grid; gap: 10px; margin-top: 12px; }
    .alert-item {
      display: grid;
      grid-template-columns: 82px 1fr auto;
      gap: 12px;
      align-items: center;
      border-left: 2px solid var(--red);
      background: rgba(11,17,28,.35);
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
    }
    .alert-item.medium { border-left-color: var(--amber); }
    .alert-badge {
      border: 1px solid currentColor;
      border-radius: 5px;
      color: var(--red);
      font-family: "Consolas", monospace;
      font-weight: 800;
      text-align: center;
      padding: 5px 6px;
      font-size: 11px;
    }
    .alert-item.medium .alert-badge { color: var(--amber); }
    .alert-main strong { display: block; color: #fff; margin-bottom: 2px; }
    .alert-time { color: var(--blue); font-family: "Consolas", monospace; white-space: nowrap; }
    .active-model {
      border: 1px solid rgba(156,140,245,.65);
      background: rgba(156,140,245,.12);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .active-dot { width: 11px; height: 11px; border-radius: 50%; background: var(--violet); box-shadow: 0 0 14px rgba(156,140,245,.9); }
    .rank-row {
      display: grid;
      grid-template-columns: 130px 1fr 58px;
      gap: 10px;
      align-items: center;
      font-family: "Consolas", monospace;
      font-size: 12px;
      margin: 10px 0;
    }
    .rank-track { height: 7px; border-radius: 999px; background: #1d273a; overflow: hidden; }
    .rank-fill { height: 100%; background: linear-gradient(90deg, var(--blue), var(--violet)); border-radius: 999px; }
    .pipeline-card { overflow: hidden; }
    .pipeline-card svg { width: 100%; min-height: 320px; display: block; }
    .pipeline-node rect { fill: #101827; stroke: #25324a; }
    .pipeline-node text { font-family: "Consolas", monospace; font-size: 12px; fill: #d7e0f2; text-anchor: middle; }
    .pipeline-node small { color: var(--muted); }
    .pipeline-link { stroke: #25324a; stroke-width: 2; fill: none; }
    .pipeline-flow {
      fill: var(--blue);
      filter: drop-shadow(0 0 8px var(--blue));
      offset-path: path("M72 145 H210 H350 H490 H628");
      animation: flow-main 4.5s linear infinite;
    }
    .pipeline-flow.attack {
      fill: var(--red);
      filter: drop-shadow(0 0 8px var(--red));
      offset-path: path("M350 145 V235");
      animation: flow-attack 2.8s linear infinite;
      animation-delay: 1.8s;
    }
    .pipeline-stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .pipeline-stat {
      background: rgba(11,17,28,.4);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .pipeline-stat span { color: var(--muted); font-size: 11px; font-family: "Consolas", monospace; display: block; }
    .pipeline-stat strong { display: block; margin-top: 4px; color: #fff; }
    @keyframes flow-main {
      from { offset-distance: 0%; opacity: 0; }
      8% { opacity: 1; }
      92% { opacity: 1; }
      to { offset-distance: 100%; opacity: 0; }
    }
    @keyframes flow-attack {
      from { offset-distance: 0%; opacity: 0; }
      15% { opacity: 1; }
      90% { opacity: 1; }
      to { offset-distance: 100%; opacity: 0; }
    }
    .attack-list { display: grid; gap: 10px; }
    .attack-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
      font-size: 13px;
    }
    .attack-name { display: flex; align-items: center; gap: 8px; font-weight: 700; }
    .swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
    .topology { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 18px; }
    .node {
      border: 1px solid var(--line);
      background: rgba(23,33,51,.65);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
      min-height: 88px;
    }
    .node-icon {
      width: 26px;
      height: 26px;
      border-radius: 7px;
      border: 1px solid rgba(var(--accent-rgb),.25);
      display: inline-grid;
      place-items: center;
      color: var(--blue);
      margin-bottom: 8px;
    }
    .node strong { display: block; font-size: 12px; }
    .node span { color: var(--muted); font-family: "Consolas", monospace; font-size: 11px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 9px;
      background: rgba(var(--accent-rgb),0.08);
      color: var(--blue);
      font-size: 12px;
      font-weight: 700;
    }
    .pill.green { background: rgba(var(--accent-rgb),0.10); color: var(--green); }
    .pill.amber { background: rgba(242,169,59,0.12); color: var(--amber); }
    .pill.red { background: rgba(242,85,92,0.12); color: var(--red); }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }
    .btn {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 12px;
      cursor: pointer;
      font-weight: 650;
    }
    .btn.primary { background: var(--blue); color: #07111d; border-color: var(--blue); }
    .btn.ghost { background: transparent; color: var(--blue); border-color: rgba(var(--accent-rgb), .35); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; }
    th { color: var(--ink); background: rgba(23,33,51,0.95); font-weight: 750; }
    .table-wrap { overflow-x: auto; }
    .bar-list { display: grid; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 170px 1fr 70px; gap: 10px; align-items: center; }
    .bar-track { height: 12px; border-radius: 999px; background: #1b2437; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; background: var(--blue); }
    .bar-fill.teal { background: var(--teal); }
    .model-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .model-card { display: grid; gap: 12px; }
    .model-head { display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }
    .model-title { font-size: 17px; font-weight: 760; }
    .mini-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .mini { background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 9px; }
    .mini span { display: block; color: var(--muted); font-size: 12px; }
    .mini strong { display: block; margin-top: 4px; }
    .media-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .media-box img, .plot img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    .file-list { display: flex; flex-wrap: wrap; gap: 6px; }
    .file-list a { color: var(--blue); text-decoration: none; border: 1px solid var(--line); padding: 6px 8px; border-radius: 6px; font-size: 12px; background: var(--panel-2); }
    .upload {
      border: 1px dashed #98a2b3;
      background: var(--panel);
      border-radius: 8px;
      padding: 18px;
      display: grid;
      gap: 12px;
    }
    input[type=file] { padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel-2); color: var(--ink); }
    .status-box { border-radius: 8px; padding: 14px; border: 1px solid var(--line); background: var(--panel-2); }
    .status-box.safe { background: rgba(var(--accent-rgb),0.10); color: var(--green); border-color: rgba(var(--accent-rgb),0.35); }
    .status-box.warning { background: rgba(242,169,59,0.12); color: var(--amber); border-color: rgba(242,169,59,0.35); }
    .status-box.danger { background: rgba(242,85,92,0.12); color: var(--red); border-color: rgba(242,85,92,0.35); }
    .note { font-size: 13px; color: var(--muted); }
    .brand-mark {
      width: 34px;
      height: 34px;
      border: 1.5px solid var(--blue);
      border-radius: 8px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-right: 10px;
      position: relative;
      vertical-align: middle;
    }
    .brand-mark::after {
      content: "";
      width: 8px;
      height: 8px;
      background: var(--blue);
      border-radius: 50%;
      box-shadow: 0 0 12px rgba(var(--accent-rgb),0.8);
      animation: pulse 2.2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(.78); } }
    .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .upload-option { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: var(--panel-2); }
    .upload-option h3 { color: var(--ink); }
    .oracle-chip { font-family: "Consolas", monospace; color: var(--blue); font-size: 12px; }
    .login-screen {
      min-height: 100vh;
      display: grid;
      place-items: center;
      background-image: url('/image?path=ufhb.jpeg');
      background-size: cover;
      background-position: center;
      padding: 22px;
    }
    .login-card {
      width: min(420px, 100%);
      background: rgba(255,255,255,.88);
      color: #142033;
      border: 1px solid rgba(255,255,255,.72);
      border-radius: 8px;
      box-shadow: 0 24px 70px rgba(0,0,0,.25);
      padding: 28px;
    }
    .login-card h1 { color: #142033; font-size: 26px; }
    .login-card p { color: #4d5b70; margin-bottom: 22px; }
    .login-card label { display: block; color: #314056; font-weight: 700; margin-bottom: 7px; }
    .login-card input {
      width: 100%;
      border: 1px solid #c8d3e2;
      background: #fff;
      color: #142033;
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 14px;
      font-size: 14px;
    }
    .login-error { color: var(--red); min-height: 18px; font-size: 13px; margin-top: 10px; }
    body.authenticated .login-screen { display: none; }
    body:not(.authenticated) .app { display: none; }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .cards, .two, .three, .model-grid, .media-row, .sentinel-grid, .demo-steps, .scope-grid, .arch-flow { grid-template-columns: 1fr; }
      .arch-node::after { display: none; }
      main { padding: 18px; }
      .bar-row { grid-template-columns: 120px 1fr 58px; }
      .dash-title { display: block; }
      .topology { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .pipeline-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .settings-actions { grid-template-columns: 1fr; }
      .alert-item { grid-template-columns: 74px 1fr; }
      .alert-time { grid-column: 2; }
    }
  </style>
</head>
<body>
<div class="login-screen" id="loginScreen">
  <form class="login-card" id="loginForm">
    <div class="brand" style="margin-bottom:14px"><span class="brand-mark"></span>SENTINEL-IIoT</div>
    <h1>Connexion IDS IIoT</h1>
    <p>Acces au tableau de bord de detection d'intrusion pour reseaux industriels.</p>
    <label for="loginUser">Utilisateur</label>
    <input id="loginUser" autocomplete="username" placeholder="admin">
    <label for="loginPassword">Mot de passe</label>
    <input id="loginPassword" type="password" autocomplete="current-password" placeholder="admin1">
    <button class="btn primary" type="submit" style="width:100%">Se connecter</button>
    <div class="login-error" id="loginError"></div>
  </form>
</div>
<div class="app">
  <aside>
    <div class="brand"><span class="brand-mark"></span>SENTINEL-IIoT</div>
    <div class="brand-sub">Demonstration CIC-IIoT-2025</div>
    <div class="status-box" style="margin-bottom:16px">
      <div class="oracle-chip">SYSTEME IDS</div>
      <div id="sideClock" style="margin-top:6px;font-weight:700">--:--:--</div>
    </div>
    <nav id="nav">
      <button data-page="dashboard" class="active">Tableau de bord</button>
      <button class="btn ghost" id="refreshAppBtn" type="button">Actualiser</button>
      <button class="btn ghost settings-toggle" id="settingsMenuBtn" type="button" aria-expanded="false">Parametres</button>
      <div class="sidebar-dropdown" id="settingsDropdown">
        <button data-page="dataset" type="button">Dataset</button>
        <button data-page="models" type="button">Modeles traites</button>
        <button data-page="comparison" type="button">Comparaison</button>
        <button data-page="best" type="button">Meilleur modele</button>
        <button data-page="prediction" type="button">Prediction</button>
        <button data-page="demo" type="button">Mode demonstration</button>
        <button data-page="architecture" type="button">Architecture</button>
        <button data-page="limits" type="button">Limites & perspectives</button>
        <button data-page="history" type="button">Historique Oracle</button>
        <button data-page="explainability" type="button">Explicabilite</button>
      </div>
      <button class="btn ghost" id="themeToggleBtn" type="button">Passer en mode clair</button>
      <button class="btn ghost" id="logoutBtn" type="button">Deconnexion</button>
    </nav>
  </aside>
  <main>
    <section id="dashboard" class="active">
      <div class="dash-title">
        <div>
          <h1>Detection d'intrusion dans les reseaux IIoT</h1>
          <p class="lead">Tableau de bord de soutenance pour presenter le pipeline IDS, les resultats experimentaux et la prediction avec le modele final.</p>
        </div>
        <div>
          <div class="system-pill">SYSTEME OPERATIONNEL</div>
          <div class="mono-sub" style="margin-top:8px;text-align:right">Abidjan, CI Â· <span id="topClock">--:--:--</span></div>
        </div>
      </div>
      <div class="dash-kicker">Vue d'ensemble Â· reseau industriel Â· CIC-IIoT-2025</div>
      <div class="grid cards" id="summaryCards"></div>
      <div class="grid sentinel-grid">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2>Trafic reseau en temps reel</h2>
              <div class="mono-sub">Flux classifies Â· fenetre glissante 60 min</div>
            </div>
            <div class="mono-sub"><span style="color:var(--blue)">â– </span> Normal &nbsp; <span style="color:var(--red)">â– </span> Attaque</div>
          </div>
          <svg class="traffic-svg" viewBox="0 0 760 250" role="img" aria-label="Graphique trafic reseau">
            <defs>
              <linearGradient id="trafficFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="#45d2c4" stop-opacity=".36"/>
                <stop offset="100%" stop-color="#45d2c4" stop-opacity="0"/>
              </linearGradient>
              <filter id="alertGlow">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            <g stroke="#1d2a40" stroke-width="1">
              <line x1="20" y1="35" x2="735" y2="35"/><line x1="20" y1="95" x2="735" y2="95"/>
              <line x1="20" y1="155" x2="735" y2="155"/><line x1="20" y1="215" x2="735" y2="215"/>
            </g>
            <path id="trafficArea" class="traffic-area" fill="url(#trafficFill)"/>
            <polyline id="normalTrafficLine" class="traffic-line" fill="none" stroke="var(--blue)" stroke-width="3"/>
            <polyline id="attackTrafficLine" class="attack-line" fill="none" stroke="var(--red)" stroke-width="2"/>
            <g id="attackTrafficPoints"></g>
          </svg>
          <div id="alertFeed" class="alert-feed"></div>
        </div>
        <div class="panel plot">
          <div class="panel-header">
            <div>
              <h2>Modele de detection</h2>
              <div class="mono-sub">Comparaison des 8 modeles evalues</div>
            </div>
          </div>
          <div id="activeModel" class="active-model"></div>
          <div id="rankingBars" class="rank-list"></div>
        </div>
      </div>
      <div class="grid sentinel-grid" style="margin-top:16px">
        <div class="panel pipeline-card">
          <h2>Pipeline de detection</h2>
          <div class="mono-sub" id="pipelineSubtitle">Du dataset prepare a la classification finale</div>
          <svg viewBox="0 0 720 330" role="img" aria-label="Pipeline IDS dynamique">
            <defs>
              <marker id="arrowHead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 Z" fill="#25324a"/>
              </marker>
            </defs>
            <path class="pipeline-link" marker-end="url(#arrowHead)" d="M72 145 H210"/>
            <path class="pipeline-link" marker-end="url(#arrowHead)" d="M210 145 H350"/>
            <path class="pipeline-link" marker-end="url(#arrowHead)" d="M350 145 H490"/>
            <path class="pipeline-link" marker-end="url(#arrowHead)" d="M490 145 H628"/>
            <path class="pipeline-link" marker-end="url(#arrowHead)" d="M350 145 V235"/>
            <g class="pipeline-node">
              <rect x="18" y="108" width="108" height="74" rx="8"/>
              <text x="72" y="139">Dataset</text>
              <text x="72" y="158" fill="#7f8ca6">CIC-IIoT-2025</text>
            </g>
            <g class="pipeline-node">
              <rect x="154" y="108" width="112" height="74" rx="8"/>
              <text x="210" y="135">Preparation</text>
              <text x="210" y="154">normalisation</text>
              <text x="210" y="173" fill="#45d2c4">SMOTE train</text>
            </g>
            <g class="pipeline-node">
              <rect x="296" y="96" width="108" height="98" rx="10" fill="rgba(156,140,245,.16)" stroke="#9c8cf5"/>
              <text x="350" y="130" fill="#fff" font-weight="700" id="pipelineModelA">Decision</text>
              <text x="350" y="151" fill="#fff" font-weight="700" id="pipelineModelB">Tree</text>
              <text x="350" y="172" fill="#9c8cf5">modele final</text>
            </g>
            <g class="pipeline-node">
              <rect x="436" y="108" width="108" height="74" rx="8"/>
              <text x="490" y="139">Prediction</text>
              <text x="490" y="158" fill="#7f8ca6">CSV / dossier</text>
            </g>
            <g class="pipeline-node">
              <rect x="592" y="108" width="104" height="74" rx="8" fill="rgba(var(--accent-rgb),.08)" stroke="var(--blue)"/>
              <text x="644" y="140" fill="var(--blue)">Benin</text>
              <text x="644" y="160" fill="#7f8ca6" id="pipelineBenign">classe 1</text>
            </g>
            <g class="pipeline-node">
              <rect x="296" y="235" width="108" height="62" rx="8" fill="rgba(242,85,92,.08)" stroke="#f2555c"/>
              <text x="350" y="262" fill="#f2555c">Attaque</text>
              <text x="350" y="281" fill="#7f8ca6" id="pipelineAttack">classe 0</text>
            </g>
            <circle class="pipeline-flow" r="5"/>
            <circle class="pipeline-flow" r="4" style="animation-delay:1.2s"/>
            <circle class="pipeline-flow" r="4" style="animation-delay:2.4s"/>
            <circle class="pipeline-flow attack" r="5"/>
          </svg>
          <div class="pipeline-stats" id="pipelineStats"></div>
        </div>
        <div class="panel">
          <h2>Repartition des attaques</h2>
          <div class="mono-sub">Donnees test - CIC-IIoT-2025</div>
          <div id="attackBreakdown" class="attack-list" style="margin-top:18px"></div>
          <div class="dash-kicker" style="margin-top:20px;margin-bottom:10px">Topologie du reseau surveille</div>
          <div class="topology">
            <div class="node"><div class="node-icon">S1</div><strong>Capteur-Temp-07</strong><span>normal</span></div>
            <div class="node"><div class="node-icon">PLC</div><strong>PLC-Ligne-A</strong><span>normal</span></div>
            <div class="node"><div class="node-icon" style="color:var(--red)">AL</div><strong>Vanne-Actionneur-02</strong><span>alerte</span></div>
            <div class="node"><div class="node-icon">GW</div><strong>Passerelle-03</strong><span>normal</span></div>
          </div>
        </div>
      </div>
    </section>

    <section id="dataset">
      <h1>Dataset CIC-IIoT-2025</h1>
      <p class="lead">Les donnees reseau sont nettoyees, encodees, normalisees puis separees en train, validation et test. SMOTE est applique sur l'entrainement pour reduire le desequilibre des classes.</p>
      <div class="grid two">
        <div class="panel">
          <h2>Separation des donnees</h2>
          <div class="table-wrap"><table id="splitTable"></table></div>
        </div>
        <div class="panel">
          <h2>Labels principaux</h2>
          <div id="labelBars" class="bar-list"></div>
        </div>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div class="panel plot">
          <h2>Correlation des variables</h2>
          <img id="correlationImage" alt="Correlation des features">
        </div>
        <div class="panel">
          <h2>Familles d'attaques</h2>
          <div class="table-wrap"><table id="familyTable"></table></div>
        </div>
      </div>
    </section>

    <section id="models">
      <h1>Modeles traites</h1>
      <p class="lead">Les modeles sont presentes dans l'ordre de traitement utilise pour le projet.</p>
      <div id="modelCards" class="grid model-grid"></div>
    </section>

    <section id="comparison">
      <h1>Comparaison des modeles</h1>
      <p class="lead">Le tableau et le graphique montrent que les modeles classiques dominent sur le test, avec Decision Tree comme meilleur modele global.</p>
      <div class="grid two">
        <div class="panel">
          <h2>Attack F1-score</h2>
          <div id="comparisonBars" class="bar-list"></div>
        </div>
        <div class="panel plot">
          <h2>Figure de classement</h2>
          <img id="rankingImage" alt="Classement final">
        </div>
      </div>
      <div class="panel" style="margin-top:16px">
        <h2>Tableau comparatif</h2>
        <div class="table-wrap"><table id="metricsTable"></table></div>
      </div>
    </section>

    <section id="best">
      <h1>Meilleur modele : Decision Tree</h1>
      <p class="lead">Le Decision Tree est retenu comme modele final car il offre le meilleur compromis entre performance, simplicite et interpretabilite.</p>
      <div class="grid cards" id="bestCards"></div>
      <div class="grid two" style="margin-top:16px">
        <div class="panel plot">
          <h2>Matrice de confusion</h2>
          <img id="bestConfusionImage" alt="Matrice de confusion Decision Tree">
        </div>
        <div class="panel plot">
          <h2>Courbe ROC</h2>
          <img id="bestRocImage" alt="Courbe ROC Decision Tree">
        </div>
      </div>
    </section>

    <section id="prediction">
      <h1>Prediction IDS</h1>
      <p class="lead">Importez un CSV de donnees reseau ou un dossier contenant plusieurs CSV. Le modele final Decision Tree classe chaque ligne en Benin ou Attaque, puis sauvegarde l'analyse dans Oracle.</p>
      <div class="grid two" style="margin-bottom:16px">
        <div class="panel">
          <h2>Quel fichier charger ?</h2>
          <p>Charge un fichier CSV contenant les caracteristiques reseau deja preparees, comme les colonnes de `X_test_raw`.</p>
          <div class="file-list">
            <a>Exemples: log_data-ranges_avg</a>
            <a>network_packets_dst_count</a>
            <a>network_tcp-flags-fin_count</a>
            <a>network_window-size_avg</a>
          </div>
          <p class="note">Les colonnes `label1`, `label2`, `label3`, `label4` peuvent etre presentes. Elles seront ignorees pour la prediction, sauf `label1` qui peut servir au calcul des metriques.</p>
        </div>
        <div class="panel">
          <h2>Dossier dataset</h2>
          <p>Oui, tu peux charger un dossier du dataset pour la prediction si ce dossier contient des fichiers `.csv` compatibles.</p>
          <p class="note">Le navigateur enverra tous les CSV du dossier. Les autres fichiers seront ignores. Pour une demonstration fluide, utilise un sous-dossier ou un echantillon reduit plutot que tout le dataset brut complet.</p>
        </div>
      </div>
      <div class="upload">
        <div class="upload-grid">
          <div class="upload-option">
            <h3>CSV unique</h3>
            <input id="csvFile" type="file" accept=".csv">
          </div>
          <div class="upload-option">
            <h3>Dossier de CSV</h3>
            <input id="folderInput" type="file" accept=".csv" webkitdirectory directory multiple>
          </div>
        </div>
        <div id="csvPreview" class="status-box">Aucun fichier selectionne.</div>
        <div class="toolbar">
          <button class="btn primary" id="predictBtn">Lancer la detection</button>
          <button class="btn" id="demoBtn">Utiliser un echantillon demo</button>
          <button class="btn" id="downloadReportBtn" type="button" disabled>Telecharger le rapport</button>
        </div>
        <div id="predictStatus" class="status-box">Aucune prediction lancee.</div>
      </div>
      <div id="predictionResult" style="margin-top:16px"></div>
    </section>

    <section id="history">
      <h1>Historique Oracle</h1>
      <p class="lead">Cette page affiche l'etat de connexion Oracle, les analyses sauvegardees et les alertes generees apres prediction.</p>
      <div class="grid three">
        <div class="card">
          <div class="metric-label">Etat Oracle</div>
          <div class="metric-value" id="oracleState">-</div>
        </div>
        <div class="card">
          <div class="metric-label">Dernieres analyses</div>
          <div class="metric-value" id="historyCount">0</div>
        </div>
        <div class="card">
          <div class="metric-label">Alertes recentes</div>
          <div class="metric-value" id="alertCount">0</div>
        </div>
      </div>
      <div id="oracleMessage" class="status-box" style="margin-top:16px"></div>
      <div class="toolbar">
        <button class="btn primary" id="refreshHistoryBtn">Actualiser</button>
        <button class="btn" id="syncModelsBtn">Synchroniser les resultats modeles</button>
      </div>
      <div class="grid two">
        <div class="panel">
          <h2>Analyses sauvegardees</h2>
          <div class="table-wrap"><table id="historyTable"></table></div>
        </div>
        <div class="panel">
          <h2>Alertes</h2>
          <div class="table-wrap"><table id="alertsTable"></table></div>
        </div>
      </div>
    </section>

    <section id="explainability">
      <h1>Explicabilite</h1>
      <p class="lead">L'explication repose sur l'importance des variables du Decision Tree et la permutation importance. Elle montre les facteurs les plus influents dans la detection.</p>
      <div class="grid two">
        <div class="panel">
          <h2>Variables importantes</h2>
          <div id="importanceBars" class="bar-list"></div>
        </div>
        <div class="panel plot">
          <h2>Graphique d'importance</h2>
          <img id="importanceImage" alt="Importance des variables">
        </div>
      </div>
    </section>

    <section id="demo">
      <h1>Mode demonstration</h1>
      <p class="lead">Parcours court pour presenter l'application au jury sans chercher les pages pendant la soutenance.</p>
      <div class="demo-steps">
        <div class="demo-step"><span>1</span><strong>Dashboard</strong><p>Presenter le projet, le dataset, le modele final et l'etat Oracle.</p><button class="btn" data-page="dashboard" type="button">Ouvrir</button></div>
        <div class="demo-step"><span>2</span><strong>Dataset</strong><p>Montrer la preparation, le split train/validation/test et SMOTE.</p><button class="btn" data-page="dataset" type="button">Ouvrir</button></div>
        <div class="demo-step"><span>3</span><strong>Comparaison</strong><p>Expliquer pourquoi Decision Tree est retenu face aux 8 modeles.</p><button class="btn" data-page="comparison" type="button">Ouvrir</button></div>
        <div class="demo-step"><span>4</span><strong>Prediction</strong><p>Importer un CSV, lancer la detection et telecharger le rapport.</p><button class="btn primary" data-page="prediction" type="button">Ouvrir</button></div>
      </div>
      <div class="panel" style="margin-top:16px">
        <h2>Phrase de demonstration</h2>
        <p>Cette application transforme les resultats du memoire en outil IDS: elle presente les donnees, compare les modeles, justifie le choix du Decision Tree, predit sur de nouveaux CSV et sauvegarde les analyses dans Oracle.</p>
      </div>
    </section>

    <section id="architecture">
      <h1>Architecture du systeme</h1>
      <p class="lead">Vue technique de bout en bout entre le dataset, le modele final, l'application web et Oracle.</p>
      <div class="arch-flow">
        <div class="arch-node"><div class="metric-label">Source</div><h2>Dataset</h2><p>CIC-IIoT-2025 avec trafic benin et attaques IIoT.</p></div>
        <div class="arch-node"><div class="metric-label">Preparation</div><h2>Pipeline ML</h2><p>Nettoyage, encodage, normalisation, split et SMOTE sur l'entrainement.</p></div>
        <div class="arch-node"><div class="metric-label">Modele</div><h2>Decision Tree</h2><p>Modele final choisi pour performance, simplicite et interpretabilite.</p></div>
        <div class="arch-node"><div class="metric-label">Interface</div><h2>Dashboard IDS</h2><p>Visualisation, comparaison, explicabilite et prediction CSV.</p></div>
        <div class="arch-node"><div class="metric-label">Persistance</div><h2>Oracle</h2><p>Stockage des analyses, predictions et alertes detectees.</p></div>
      </div>
      <div class="grid three" style="margin-top:16px">
        <div class="card"><div class="metric-label">Utilisateur</div><div class="metric-value">admin</div></div>
        <div class="card"><div class="metric-label">Modele deploye</div><div class="metric-value">Decision Tree</div></div>
        <div class="card"><div class="metric-label">Prediction</div><div class="metric-value">CSV</div></div>
      </div>
    </section>

    <section id="limits">
      <h1>Limites & perspectives</h1>
      <p class="lead">Cette page clarifie le perimetre exact du prototype: ce qui est deja fonctionnel, ce qui reste une limite actuelle, et les evolutions possibles vers un IDS temps reel.</p>
      <div class="scope-grid">
        <div class="scope-card">
          <div class="scope-badge">REALISE</div>
          <h2>Prototype fonctionnel</h2>
          <ul>
            <li>Comparaison de 8 modeles sur CIC-IIoT-2025.</li>
            <li>Decision Tree retenu comme modele final.</li>
            <li>Prediction reelle a partir de fichiers CSV compatibles.</li>
            <li>Sauvegarde des analyses et alertes dans Oracle.</li>
            <li>Tableau de bord dynamique pour la soutenance.</li>
          </ul>
        </div>
        <div class="scope-card">
          <div class="scope-badge warning">LIMITES</div>
          <h2>Perimetre actuel</h2>
          <ul>
            <li>La prediction depend du format des fichiers CSV.</li>
            <li>L'application n'est pas connectee a un capteur reseau physique.</li>
            <li>Le graphique temps reel utilise les predictions et l'historique Oracle.</li>
            <li>Il ne s'agit pas encore d'une capture reseau industrielle en direct.</li>
          </ul>
        </div>
        <div class="scope-card">
          <div class="scope-badge future">PERSPECTIVES</div>
          <h2>Evolution professionnelle</h2>
          <ul>
            <li>Connecter un flux reseau reel ou un collecteur de paquets.</li>
            <li>Ajouter une API de surveillance continue.</li>
            <li>Creer des roles utilisateurs: admin, analyste, lecteur.</li>
            <li>Ajouter des notifications d'alerte.</li>
            <li>Deployer l'application sur un serveur.</li>
          </ul>
        </div>
      </div>
      <div class="panel" style="margin-top:16px">
        <h2>Formulation correcte pour la soutenance</h2>
        <p>L'application est un prototype IDS fonctionnel: elle applique le modele Decision Tree sur des donnees reseau au format CSV, enregistre les resultats dans Oracle et affiche les alertes dans un tableau de bord dynamique. La connexion a un flux reseau reel constitue une perspective d'amelioration.</p>
      </div>
    </section>

    <section id="settings">
      <h1>Parametres</h1>
      <p class="lead">Reglages de l'interface et de la session.</p>
      <div class="grid two">
        <div class="panel">
          <h2>Apparence</h2>
          <p>Le changement sombre/clair est disponible directement dans le menu deroulant Parametres de la barre verticale.</p>
        </div>
        <div class="panel">
          <h2>Session</h2>
          <p>Page de connexion active avec arriere-plan UFHB. Cette connexion sert a presenter l'application comme un outil web complet pendant la soutenance.</p>
          <div class="status-box safe">Utilisateur connecte : <strong id="settingsUser">admin</strong></div>
        </div>
      </div>
    </section>
  </main>
</div>

<script>
const state = { data: null, history: [], alerts: [], lastPrediction: null };
const fmt = (v, digits = 4) => Number(v || 0).toFixed(digits);
const pct = (v) => (Number(v || 0) * 100).toFixed(1) + "%";

function setPage(page) {
  document.querySelectorAll("section").forEach(s => s.classList.toggle("active", s.id === page));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.page === page));
}

function applyMode(mode) {
  const light = mode === "light";
  document.body.classList.toggle("mode-light", light);
  const btn = document.getElementById("themeToggleBtn");
  if (btn) btn.textContent = light ? "Passer en mode sombre" : "Passer en mode clair";
  localStorage.setItem("idsMode", light ? "light" : "dark");
}

function authenticate(user) {
  const username = user || "admin";
  localStorage.setItem("idsSessionUser", username);
  document.body.classList.add("authenticated");
  const sessionUser = document.getElementById("sessionUser");
  const settingsUser = document.getElementById("settingsUser");
  if (sessionUser) sessionUser.textContent = username;
  if (settingsUser) settingsUser.textContent = username;
}

function logout() {
  localStorage.removeItem("idsSessionUser");
  document.body.classList.remove("authenticated");
  setPage("dashboard");
}

function table(el, rows, columns) {
  if (!rows || rows.length === 0) {
    el.innerHTML = "<tbody><tr><td>Aucune donnee disponible</td></tr></tbody>";
    return;
  }
  const cols = columns || Object.keys(rows[0]);
  el.innerHTML = "<thead><tr>" + cols.map(c => `<th>${c.label || c}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map(row => "<tr>" + cols.map(c => {
      const key = c.key || c;
      const val = row[key];
      return `<td>${typeof val === "number" ? fmt(val) : (val ?? "")}</td>`;
    }).join("") + "</tr>").join("") + "</tbody>";
}

function bars(el, rows, labelKey, valueKey, colorClass = "") {
  if (!el) return;
  const max = Math.max(...rows.map(r => Number(r[valueKey] || 0)), 1);
  el.innerHTML = rows.map(r => {
    const value = Number(r[valueKey] || 0);
    return `<div class="bar-row">
      <div>${r[labelKey]}</div>
      <div class="bar-track"><div class="bar-fill ${colorClass}" style="width:${(value / max) * 100}%"></div></div>
      <strong>${value <= 1 ? fmt(value, 3) : value}</strong>
    </div>`;
  }).join("");
}

function rankingBars(el, rows) {
  if (!el) return;
  const max = Math.max(...rows.map(r => Number(r.attack_f1_score || 0)), 1);
  el.innerHTML = rows.map(r => {
    const value = Number(r.attack_f1_score || 0);
    return `<div class="rank-row">
      <div>${r.name}</div>
      <div class="rank-track"><div class="rank-fill" style="width:${(value / max) * 100}%"></div></div>
      <strong>${fmt(value, 3)}</strong>
    </div>`;
  }).join("");
}

function renderAlertFeed(rows) {
  const el = document.getElementById("alertFeed");
  if (!el) return;
  const fallback = [
    { severity: "CRITIQUE", message: "DDoS volumetrique detecte", detail: "Passerelle-03 Â· controleur PLC", source_ip: "192.168.4.12" },
    { severity: "MOYEN", message: "Scan de ports anormal", detail: "Capteur-Temp-07", source_ip: "192.168.4.44" },
    { severity: "CRITIQUE", message: "Injection de commandes Modbus", detail: "Actionneur-Vanne-02", source_ip: "192.168.4.09" },
  ];
  const list = (rows && rows.length ? rows : fallback).slice(0, 3);
  el.innerHTML = list.map((row, index) => {
    const sev = String(row.severity || (index === 1 ? "MOYEN" : "CRITIQUE")).toUpperCase();
    const medium = sev.includes("MOY") || sev.includes("WARN") || sev.includes("LOW");
    return `<div class="alert-item ${medium ? "medium" : ""}">
      <div class="alert-badge">${medium ? "MOYEN" : "CRITIQUE"}</div>
      <div class="alert-main"><strong>${row.message || "Attaque detectee"}</strong><span class="mono-sub">${row.detail || "Decision Tree Â· paquet suspect"}</span></div>
      <div class="alert-time">${row.source_ip || "Oracle"} Â· ${index === 0 ? "il y a 41 s" : "il y a " + (index + 2) + " min"}</div>
    </div>`;
  }).join("");
}

function fileSize(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 * 1024) return (value / (1024 * 1024)).toFixed(1) + " Mo";
  if (value >= 1024) return (value / 1024).toFixed(1) + " Ko";
  return value + " o";
}

async function updateCsvPreview() {
  const input = document.getElementById("csvFile");
  const folderInput = document.getElementById("folderInput");
  const preview = document.getElementById("csvPreview");
  const files = folderInput.files.length ? Array.from(folderInput.files) : Array.from(input.files);
  const csvFiles = files.filter(file => file.name.toLowerCase().endsWith(".csv"));
  if (!csvFiles.length) {
    preview.className = "status-box";
    preview.textContent = "Aucun fichier CSV selectionne.";
    return;
  }
  const first = csvFiles[0];
  let columns = [];
  try {
    const head = await first.slice(0, 65536).text();
    columns = (head.split(/\r?\n/)[0] || "").split(",").filter(Boolean);
  } catch {
    columns = [];
  }
  preview.className = "status-box safe";
  preview.innerHTML = `<strong>${csvFiles.length} fichier(s) CSV selectionne(s)</strong><br>
    Premier fichier : ${first.webkitRelativePath || first.name} · ${fileSize(first.size)}<br>
    Colonnes detectees : ${columns.length || "non determine"}`;
}

function buildPredictionReport(data) {
  const esc = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const lines = [
    "section;champ;valeur",
    `rapport;date;${esc(new Date().toLocaleString("fr-FR"))}`,
    `rapport;modele;${esc(data.model || "decision_tree")}`,
    `rapport;echantillons;${esc(data.samples)}`,
    `rapport;attaques;${esc(data.attack_count)}`,
    `rapport;benins;${esc(data.benign_count)}`,
    `rapport;taux_alerte;${esc(pct(data.alert_rate))}`,
    `rapport;statut;${esc(data.status)}`,
    `rapport;oracle;${esc(data.oracle && data.oracle.message ? data.oracle.message : "non renseignee")}`,
  ];
  (data.files || []).forEach(file => {
    lines.push(`fichier;${esc(file.filename)};${esc(`lignes=${file.samples}; attaques=${file.attack_count}; benins=${file.benign_count}; compatible=${file.compatible}`)}`);
  });
  if (data.validation && data.validation.files) {
    data.validation.files.forEach(file => {
      lines.push(`validation;${esc(file.filename)};${esc(`features=${file.present_count}/${file.expected_count}; manquantes=${file.missing_count}; supplementaires=${file.extra_count}`)}`);
    });
  }
  return lines.join("\n");
}

function downloadPredictionReport() {
  if (!state.lastPrediction) return;
  const blob = new Blob([buildPredictionReport(state.lastPrediction)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `rapport_prediction_${state.lastPrediction.upload_id || "ids"}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function numberFromText(value) {
  const match = String(value || "").replace(/\s/g, "").match(/(\d+(?:[.,]\d+)?)/);
  return match ? Number(match[1].replace(",", ".")) : 0;
}

function historyToTraffic(historyRows, alertRows) {
  const slots = 16;
  const normal = Array.from({ length: slots }, (_, i) => 45 + i * 3 + Math.round(Math.sin(i * 1.1) * 8));
  const attacks = Array.from({ length: slots }, () => 3);
  const rows = historyRows && historyRows.length ? historyRows.slice(0, slots).reverse() : [];
  rows.forEach((row, index) => {
    const slot = Math.max(0, slots - rows.length + index);
    const attack = Number(row.attack_count || 0);
    const samples = Number(row.sample_count || row.samples || 0);
    const benign = Math.max(Number(row.benign_count || 0), samples - attack, 0);
    normal[slot] = Math.max(8, Math.round(benign / Math.max(samples, 1) * 100));
    attacks[slot] = Math.max(2, Math.round(attack / Math.max(samples, 1) * 100));
  });
  const alerts = alertRows && alertRows.length ? alertRows.slice(0, 4) : [];
  alerts.forEach((row, index) => {
    const slot = Math.max(1, slots - 1 - index * 3);
    const severityBoost = String(row.severity || "").toUpperCase().includes("CRIT") ? 38 : 22;
    const detected = numberFromText(row.message);
    attacks[slot] = Math.min(98, Math.max(attacks[slot], severityBoost + Math.min(38, Math.round(detected / 600))));
  });
  return { normal, attacks };
}

function renderTrafficChart(historyRows = [], alertRows = []) {
  const normalLine = document.getElementById("normalTrafficLine");
  const attackLine = document.getElementById("attackTrafficLine");
  const area = document.getElementById("trafficArea");
  const points = document.getElementById("attackTrafficPoints");
  if (!normalLine || !attackLine || !area || !points) return;
  const { normal, attacks } = historyToTraffic(historyRows, alertRows);
  const left = 20, right = 735, top = 35, bottom = 225;
  const xStep = (right - left) / Math.max(normal.length - 1, 1);
  const maxValue = Math.max(...normal, ...attacks, 100);
  const toPoint = (value, index) => {
    const x = left + xStep * index;
    const y = bottom - (Math.min(value, maxValue) / maxValue) * (bottom - top);
    return [Math.round(x), Math.round(y)];
  };
  const normalPoints = normal.map(toPoint);
  const attackPoints = attacks.map(toPoint);
  normalLine.setAttribute("points", normalPoints.map(p => p.join(",")).join(" "));
  attackLine.setAttribute("points", attackPoints.map(p => p.join(",")).join(" "));
  area.setAttribute("d", `M${normalPoints.map(p => p.join(" ")).join(" L")} L${right} ${bottom} L${left} ${bottom} Z`);
  points.innerHTML = attackPoints
    .filter((point, index) => attacks[index] >= 20)
    .map(point => `<circle class="attack-point traffic-pulse" cx="${point[0]}" cy="${point[1]}" r="5" fill="var(--red)" filter="url(#alertGlow)"/>`)
    .join("");
}

function renderAttackBreakdown(rows) {
  const el = document.getElementById("attackBreakdown");
  if (!el) return;
  const colors = ["#f2555c", "#f2a93b", "#9c8cf5", "#45d2c4", "#7f8ca6"];
  const prepared = (rows || []).slice(0, 5).map((row, index) => ({
    name: row.label2 || row.attack_family || `Famille ${index + 1}`,
    count: Number(row.attack_support || row.support || row.count || 0),
    color: colors[index % colors.length],
  }));
  const list = prepared.length ? prepared : [
    { name: "DDoS", count: 1204, color: colors[0] },
    { name: "Reconnaissance / scan", count: 861, color: colors[1] },
    { name: "Injection Modbus", count: 373, color: colors[2] },
    { name: "Usurpation / MITM", count: 258, color: colors[3] },
  ];
  const total = Math.max(list.reduce((sum, row) => sum + row.count, 0), 1);
  el.innerHTML = list.map(row => `<div class="attack-row">
    <div class="attack-name"><span class="swatch" style="background:${row.color}"></span>${row.name}</div>
    <div>${Math.round(row.count).toLocaleString("fr-FR")}</div>
    <strong>${Math.round((row.count / total) * 100)}%</strong>
  </div>`).join("");
}

function renderPipeline(data, best) {
  const totalRows = (data.split || []).reduce((sum, row) => {
    const value = Number(row.n_rows || row.rows || row.count || row.samples || 0);
    return sum + value;
  }, 0);
  const labelRows = data.label1 || [];
  const attackRow = labelRows.find(r => String(r.label1 || r.label || "").toLowerCase().includes("attack"));
  const benignRow = labelRows.find(r => String(r.label1 || r.label || "").toLowerCase().includes("benign"));
  const attackCount = Number((attackRow && (attackRow.count || attackRow.support)) || 0);
  const benignCount = Number((benignRow && (benignRow.count || benignRow.support)) || 0);
  const modelParts = String(best.name || "Decision Tree").split(" ");
  const modelA = document.getElementById("pipelineModelA");
  const modelB = document.getElementById("pipelineModelB");
  if (modelA) modelA.textContent = modelParts[0] || "Decision";
  if (modelB) modelB.textContent = modelParts.slice(1).join(" ") || "Tree";
  const benignEl = document.getElementById("pipelineBenign");
  const attackEl = document.getElementById("pipelineAttack");
  if (benignEl) benignEl.textContent = benignCount ? `${benignCount.toLocaleString("fr-FR")} exemples` : "classe 1";
  if (attackEl) attackEl.textContent = attackCount ? `${attackCount.toLocaleString("fr-FR")} exemples` : "classe 0";
  const subtitle = document.getElementById("pipelineSubtitle");
  if (subtitle) subtitle.textContent = `${data.summary.dataset} - ${data.summary.models_count} modeles testes - ${best.name} retenu`;
  const stats = document.getElementById("pipelineStats");
  if (!stats) return;
  const rows = [
    ["Dataset", data.summary.dataset],
    ["Total lignes", totalRows ? totalRows.toLocaleString("fr-FR") : "train/val/test"],
    ["Modele final", best.name],
    ["Attack F1", fmt(best.attack_f1_score)],
  ];
  stats.innerHTML = rows.map(([label, value]) => `<div class="pipeline-stat"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function fileLinks(assets) {
  return Object.entries(assets).filter(([, path]) => path).map(([name, path]) => {
    const href = path.endsWith(".png") ? `/image?path=${encodeURIComponent(path)}` : `/file?path=${encodeURIComponent(path)}`;
    return `<a href="${href}" target="_blank">${name}</a>`;
  }).join("");
}

function render(data) {
  state.data = data;
  const s = data.summary;
  document.getElementById("summaryCards").innerHTML = [
    ["Lignes test", Number(s.test_rows || 0).toLocaleString("fr-FR"), "jeu de test reel"],
    ["Attaques test", Number(s.test_attacks || 0).toLocaleString("fr-FR"), "classes label1"],
    ["Taux de detection F1", pct(s.attack_f1_score), s.best_model],
    ["Latence inference", fmt(s.latency_ms, 4), "ms / ligne"],
  ].map(([label, value, trend]) => `<div class="card kpi-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="kpi-trend">${trend}</div></div>`).join("");

  document.getElementById("activeModel").innerHTML = `<div class="active-dot"></div><div><strong>${s.best_model}</strong><div class="mono-sub">Modele final deploye Â· pret pour la prediction</div></div>`;
  rankingBars(document.getElementById("rankingBars"), data.ranking);
  bars(document.getElementById("comparisonBars"), data.ranking, "name", "attack_f1_score", "teal");
  if (document.getElementById("label1Image")) document.getElementById("label1Image").src = data.images.label1;
  document.getElementById("correlationImage").src = data.images.correlation;
  document.getElementById("rankingImage").src = data.images.ranking;
  document.getElementById("bestConfusionImage").src = data.images.best_confusion;
  document.getElementById("bestRocImage").src = data.images.best_roc;
  document.getElementById("importanceImage").src = data.images.feature_importance;
  renderOracleStatus(data.oracle);
  renderAttackBreakdown(data.attack_family);
  renderAlertFeed([]);
  renderTrafficChart([], []);

  table(document.getElementById("splitTable"), data.split);
  bars(document.getElementById("labelBars"), data.label1, "label1", "count", "teal");
  table(document.getElementById("familyTable"), data.attack_family, [
    {key: "label2", label: "famille"},
    {key: "attack_support", label: "attaques"},
    {key: "attack_detection_rate", label: "taux detection"},
    {key: "f1_attack", label: "F1 attaque"},
  ]);

  document.getElementById("modelCards").innerHTML = data.models.map(m => `
    <div class="card model-card">
      <div class="model-head">
        <div><div class="model-title">${m.order}. ${m.name}</div><p>${m.role}</p></div>
        <span class="pill">${m.type}</span>
      </div>
      <div class="mini-metrics">
        <div class="mini"><span>Accuracy</span><strong>${fmt(m.accuracy)}</strong></div>
        <div class="mini"><span>Attack F1</span><strong>${fmt(m.attack_f1_score)}</strong></div>
        <div class="mini"><span>ROC-AUC</span><strong>${fmt(m.roc_auc)}</strong></div>
      </div>
      <div class="media-row">
        <div class="media-box">${m.assets.test_confusion ? `<img src="/image?path=${encodeURIComponent(m.assets.test_confusion)}" alt="Matrice ${m.name}">` : ""}</div>
        <div class="media-box">${m.assets.test_roc ? `<img src="/image?path=${encodeURIComponent(m.assets.test_roc)}" alt="ROC ${m.name}">` : ""}</div>
      </div>
      <div class="file-list">${fileLinks(m.assets)}</div>
    </div>`).join("");

  table(document.getElementById("metricsTable"), data.ranking, [
    {key: "name", label: "Modele"},
    {key: "type", label: "Type"},
    {key: "accuracy", label: "Accuracy"},
    {key: "precision", label: "Precision"},
    {key: "recall", label: "Recall"},
    {key: "f1_score", label: "F1-score"},
    {key: "attack_f1_score", label: "Attack F1"},
    {key: "roc_auc", label: "ROC-AUC"},
  ]);

  const best = data.models.find(m => m.id === "decision_tree");
  renderPipeline(data, best);
  document.getElementById("bestCards").innerHTML = [
    ["Accuracy", fmt(best.accuracy)],
    ["Precision", fmt(best.precision)],
    ["Recall", fmt(best.recall)],
    ["Attack F1-score", fmt(best.attack_f1_score)],
  ].map(([label, value]) => `<div class="card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`).join("");

  const importanceRows = data.importance.map(r => ({
    feature: r.feature,
    value: Number(r.permutation_importance_mean || r.tree_importance || 0)
  }));
  bars(document.getElementById("importanceBars"), importanceRows, "feature", "value", "teal");
}

function renderOracleStatus(oracle) {
  const stateEl = document.getElementById("oracleState");
  const msgEl = document.getElementById("oracleMessage");
  if (!stateEl || !msgEl) return;
  stateEl.textContent = oracle.connected ? "Connecte" : (oracle.enabled ? "Non connecte" : "Desactive");
  msgEl.className = "status-box " + (oracle.connected ? "safe" : (oracle.enabled ? "warning" : ""));
  msgEl.textContent = oracle.message;
}

async function loadData() {
  const res = await fetch("/api/summary");
  render(await res.json());
  loadHistory();
}

async function loadHistory() {
  const res = await fetch("/api/history");
  const data = await res.json();
  if (!res.ok) return;
  document.getElementById("historyCount").textContent = data.history.length;
  document.getElementById("alertCount").textContent = data.alerts.length;
  state.history = data.history || [];
  state.alerts = data.alerts || [];
  renderAlertFeed(data.alerts);
  renderTrafficChart(state.history, state.alerts);
  renderOracleStatus(data.oracle);
  table(document.getElementById("historyTable"), data.history, [
    {key: "created_at", label: "date"},
    {key: "source_type", label: "source"},
    {key: "filename", label: "fichier"},
    {key: "sample_count", label: "lignes"},
    {key: "attack_count", label: "attaques"},
    {key: "alert_rate", label: "taux"},
    {key: "status", label: "statut"},
  ]);
  table(document.getElementById("alertsTable"), data.alerts, [
    {key: "created_at", label: "date"},
    {key: "severity", label: "niveau"},
    {key: "message", label: "message"},
  ]);
}

async function syncModels() {
  const msgEl = document.getElementById("oracleMessage");
  msgEl.className = "status-box";
  msgEl.textContent = "Synchronisation en cours...";
  const res = await fetch("/api/oracle/sync-models", { method: "POST" });
  const data = await res.json();
  msgEl.className = "status-box " + (res.ok && data.saved ? "safe" : "warning");
  msgEl.textContent = data.message || data.error || "Synchronisation terminee.";
  loadHistory();
}

async function predictWithFile() {
  const input = document.getElementById("csvFile");
  const folderInput = document.getElementById("folderInput");
  const status = document.getElementById("predictStatus");
  const files = folderInput.files.length ? Array.from(folderInput.files) : Array.from(input.files);
  if (!files.length) {
    status.className = "status-box warning";
    status.textContent = "Choisissez un fichier CSV ou un dossier contenant des CSV avant de lancer la detection.";
    return;
  }
  const form = new FormData();
  files.forEach(file => {
    if (file.name.toLowerCase().endsWith(".csv")) {
      form.append("file", file, file.webkitRelativePath || file.name);
    }
  });
  status.className = "status-box";
  status.textContent = `Prediction en cours sur ${files.length} fichier(s)...`;
  const res = await fetch("/api/predict", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    status.className = "status-box danger";
    status.textContent = data.error || "Erreur pendant la prediction.";
    return;
  }
  renderPrediction(data);
  loadHistory();
}

async function predictDemo() {
  const status = document.getElementById("predictStatus");
  status.className = "status-box";
  status.textContent = "Prediction demo en cours...";
  const res = await fetch("/api/predict-demo?limit=80", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    status.className = "status-box danger";
    status.textContent = data.error || "Erreur pendant la prediction demo.";
    return;
  }
  renderPrediction(data);
  loadHistory();
}

function renderPrediction(data) {
  state.lastPrediction = data;
  document.getElementById("downloadReportBtn").disabled = false;
  const liveRow = {
    sample_count: data.samples,
    attack_count: data.attack_count,
    benign_count: data.benign_count,
  };
  const liveAlert = {
    severity: data.attack_count > 0 ? "CRITIQUE" : "NORMAL",
    message: data.attack_count > 0
      ? `Alerte critique: ${data.attack_count} attaque(s) detectee(s), taux ${pct(data.alert_rate)}.`
      : "Trafic majoritairement benin.",
    detail: "Decision Tree - prediction instantanee",
    source_ip: "Live",
  };
  state.history = [...(state.history || []), liveRow].slice(-16);
  state.alerts = [liveAlert, ...(state.alerts || [])].slice(0, 8);
  renderAlertFeed(state.alerts);
  renderTrafficChart(state.history, state.alerts);
  const status = document.getElementById("predictStatus");
  status.className = `status-box ${data.status}`;
  status.innerHTML = `<strong>${data.attack_count} attaque(s) detectee(s)</strong> sur ${data.samples} ligne(s). Taux d'alerte : ${pct(data.alert_rate)}.`;
  const result = document.getElementById("predictionResult");
  const metrics = data.metrics ? `<div class="panel" style="margin-bottom:16px"><h2>Metriques sur ce fichier</h2><div class="grid cards">
    ${Object.entries(data.metrics).slice(0, 8).map(([k, v]) => `<div class="card"><div class="metric-label">${k}</div><div class="metric-value">${fmt(v)}</div></div>`).join("")}
  </div></div>` : "";
  const validation = data.validation ? `<div class="panel" style="margin-bottom:16px">
    <h2>Verification du CSV</h2>
    <div class="status-box ${data.validation.compatible ? "safe" : "warning"}">
      ${data.validation.compatible ? "Format compatible avec le modele." : "Format partiellement compatible: certaines colonnes attendues sont absentes."}
    </div>
    ${data.validation.files ? `<div class="table-wrap" style="margin-top:12px"><table id="validationTable"></table></div>` : ""}
  </div>` : "";
  const fileSummary = data.files ? `<div class="panel" style="margin-bottom:16px"><h2>Resume par fichier</h2><div class="table-wrap"><table id="fileTable"></table></div></div>` : "";
  result.innerHTML = metrics + validation + fileSummary + `<div class="panel"><h2>Predictions</h2><div class="table-wrap"><table id="predTable"></table></div><p class="note">Affichage limite aux 100 premieres lignes.</p></div>`;
  if (data.oracle && data.oracle.message) {
    result.innerHTML = `<div class="status-box ${data.oracle.saved ? "safe" : ""}" style="margin-bottom:16px">${data.oracle.message}</div>` + result.innerHTML;
  }
  if (data.files) {
    table(document.getElementById("fileTable"), data.files, [
      {key: "filename", label: "fichier"},
      {key: "samples", label: "lignes"},
      {key: "attack_count", label: "attaques"},
      {key: "benign_count", label: "benins"},
      {key: "alert_rate", label: "taux alerte"},
      {key: "compatible", label: "compatible"},
      {key: "features_present", label: "features"},
      {key: "oracle_saved", label: "oracle"},
    ]);
  }
  if (data.validation && data.validation.files) {
    table(document.getElementById("validationTable"), data.validation.files, [
      {key: "filename", label: "fichier"},
      {key: "compatible", label: "compatible"},
      {key: "present_count", label: "features presentes"},
      {key: "expected_count", label: "features attendues"},
      {key: "missing_count", label: "manquantes"},
      {key: "extra_count", label: "supplementaires"},
    ]);
  }
  table(document.getElementById("predTable"), data.predictions, [
    {key: "fichier", label: "fichier"},
    {key: "ligne", label: "ligne"},
    {key: "classe_predite", label: "classe predite"},
    {key: "probabilite_attaque", label: "probabilite attaque"},
    {key: "probabilite_benin", label: "probabilite benin"},
  ]);
}

document.addEventListener("click", (event) => {
  const pageBtn = event.target.closest("[data-page]");
  if (pageBtn) setPage(pageBtn.dataset.page);
});
document.getElementById("loginForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const user = document.getElementById("loginUser").value.trim();
  const password = document.getElementById("loginPassword").value.trim();
  if (!user || !password) {
    document.getElementById("loginError").textContent = "Saisis un utilisateur et un mot de passe.";
    return;
  }
  if (user !== "admin" || password !== "admin1") {
    document.getElementById("loginError").textContent = "Identifiants incorrects. Utilise admin / admin1 pour la demonstration.";
    return;
  }
  authenticate(user);
});
document.getElementById("logoutBtn").addEventListener("click", logout);
document.getElementById("refreshAppBtn").addEventListener("click", () => {
  loadData();
});
document.getElementById("themeToggleBtn").addEventListener("click", () => {
  applyMode(document.body.classList.contains("mode-light") ? "dark" : "light");
});
document.getElementById("settingsMenuBtn").addEventListener("click", () => {
  const dropdown = document.getElementById("settingsDropdown");
  const button = document.getElementById("settingsMenuBtn");
  const open = !dropdown.classList.contains("open");
  dropdown.classList.toggle("open", open);
  button.classList.toggle("open", open);
  button.setAttribute("aria-expanded", open ? "true" : "false");
});
document.getElementById("predictBtn").addEventListener("click", predictWithFile);
document.getElementById("demoBtn").addEventListener("click", predictDemo);
document.getElementById("csvFile").addEventListener("change", updateCsvPreview);
document.getElementById("folderInput").addEventListener("change", updateCsvPreview);
document.getElementById("downloadReportBtn").addEventListener("click", downloadPredictionReport);
document.getElementById("refreshHistoryBtn").addEventListener("click", loadHistory);
document.getElementById("syncModelsBtn").addEventListener("click", syncModels);
loadData().catch(err => {
  document.body.innerHTML = `<pre style="padding:24px">${err.stack || err}</pre>`;
});
applyMode(localStorage.getItem("idsMode") || "dark");
const savedUser = localStorage.getItem("idsSessionUser");
if (savedUser) authenticate(savedUser);
function tickClock() {
  const el = document.getElementById("sideClock");
  if (el) el.textContent = new Date().toLocaleTimeString("fr-FR");
  const topEl = document.getElementById("topClock");
  if (topEl) topEl.textContent = new Date().toLocaleTimeString("fr-FR");
}
tickClock();
setInterval(tickClock, 1000);
setInterval(() => {
  if (document.body.classList.contains("authenticated")) loadHistory();
}, 15000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "IDSWebDashboard/1.0"

    def send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_text(html_page())
            elif parsed.path == "/api/health":
                self.send_json({"status": "ok", "app": "IDS IIoT dashboard"})
            elif parsed.path == "/api/summary":
                self.send_json(load_dashboard_data())
            elif parsed.path == "/api/oracle/status":
                self.send_json(oracle_store.status())
            elif parsed.path == "/api/history":
                self.handle_history()
            elif parsed.path == "/image":
                self.serve_image(parsed.query)
            elif parsed.path == "/file":
                self.serve_file(parsed.query)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - keeps demo server informative
            self.send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/predict":
                self.handle_predict()
            elif parsed.path == "/api/predict-demo":
                limit = int(parse_qs(parsed.query).get("limit", ["80"])[0])
                self.handle_predict_demo(limit)
            elif parsed.path == "/api/oracle/sync-models":
                self.handle_sync_models()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover
            self.send_json({"error": str(exc)}, status=500)

    def serve_image(self, query: str) -> None:
        self.serve_file(query)

    def serve_file(self, query: str) -> None:
        params = parse_qs(query)
        requested = params.get("path", [""])[0]
        path = (ROOT / requested).resolve()
        allowed_roots = [(REPORTS).resolve(), (ROOT / "models").resolve()]
        allowed_files = {(ROOT / "ufhb.jpeg").resolve()}
        if (not any(path.is_relative_to(root) for root in allowed_roots) and path not in allowed_files) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_predict(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        fields = parse_multipart(body, content_type)
        if "file" not in fields:
            self.send_json({"error": "Aucun fichier CSV recu."}, status=400)
            return
        self.send_json(predict_uploaded_files(fields["file"]))

    def handle_predict_demo(self, limit: int) -> None:
        x_path = OUTPUTS / "splits" / "X_test_raw.pkl.gz"
        y_path = OUTPUTS / "splits" / "y_test.csv"
        if not x_path.exists():
            self.send_json({"error": "Echantillon demo introuvable."}, status=404)
            return
        raw_df = pd.read_pickle(x_path, compression="gzip").head(limit).copy()
        if y_path.exists():
            raw_df["label1_encoded"] = pd.read_csv(y_path).iloc[: len(raw_df), 0].to_numpy()
        result = run_prediction(raw_df)
        oracle_result = save_prediction_to_oracle(result, source="demo", filename="X_test_raw_demo")
        self.send_json(public_prediction_result(result, oracle_result))

    def handle_history(self) -> None:
        oracle_status = oracle_store.status()
        if not oracle_status.get("connected"):
            self.send_json({"oracle": oracle_status, "history": [], "alerts": []})
            return
        try:
            self.send_json(
                {
                    "oracle": oracle_status,
                    "history": oracle_store.list_history(limit=25),
                    "alerts": oracle_store.list_alerts(limit=25),
                }
            )
        except Exception as exc:
            self.send_json({"oracle": {**oracle_status, "connected": False, "message": str(exc)}, "history": [], "alerts": []})

    def handle_sync_models(self) -> None:
        data = load_dashboard_data()
        oracle_status = data["oracle"]
        if not oracle_status.get("connected"):
            self.send_json({"saved": False, "message": oracle_status.get("message", "Oracle non connecte.")}, status=400)
            return
        try:
            self.send_json(oracle_store.sync_model_results(data["models"]))
        except Exception as exc:
            self.send_json({"saved": False, "message": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        return


def run(host: str = "127.0.0.1", port: int = 8060) -> None:
    with ThreadingHTTPServer((host, port), DashboardHandler) as server:
        print(f"IDS IIoT dashboard running at http://{host}:{port}")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IDS IIoT web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
