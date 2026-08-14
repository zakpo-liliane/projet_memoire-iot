from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
REQUIRED_TABLES = {
    "IDS_UPLOADS",
    "IDS_PREDICTIONS",
    "IDS_ALERTS",
    "IDS_MODEL_RESULTS",
}


def load_local_env() -> None:
    env_path = ROOT / "oracle_config.env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class OracleConfig:
    enabled: bool
    user: str
    password: str
    dsn: str

    @classmethod
    def from_env(cls) -> "OracleConfig":
        load_local_env()
        user = os.getenv("IDS_ORACLE_USER", "")
        password = os.getenv("IDS_ORACLE_PASSWORD", "")
        dsn = os.getenv("IDS_ORACLE_DSN", "")
        enabled = os.getenv("IDS_ORACLE_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        return cls(enabled=enabled, user=user, password=password, dsn=dsn)


def _load_driver():
    try:
        import oracledb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Le module Python 'oracledb' n'est pas installe. "
            "Installez-le avec: pip install oracledb"
        ) from exc
    return oracledb


def status() -> dict[str, Any]:
    config = OracleConfig.from_env()
    if not config.enabled:
        return {
            "enabled": False,
            "connected": False,
            "message": "Oracle desactive. Configurez IDS_ORACLE_ENABLED=1 pour activer la base.",
        }
    if not (config.user and config.password and config.dsn):
        return {
            "enabled": True,
            "connected": False,
            "message": "Configuration Oracle incomplete: IDS_ORACLE_USER, IDS_ORACLE_PASSWORD, IDS_ORACLE_DSN.",
        }
    try:
        oracledb = _load_driver()
        with oracledb.connect(user=config.user, password=config.password, dsn=config.dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute("select 1 from dual")
                cursor.fetchone()
                cursor.execute(
                    """
                    select table_name
                    from user_tables
                    where table_name in ('IDS_UPLOADS', 'IDS_PREDICTIONS', 'IDS_ALERTS', 'IDS_MODEL_RESULTS')
                    """
                )
                existing = {row[0] for row in cursor.fetchall()}
        missing = sorted(REQUIRED_TABLES - existing)
        if missing:
            return {
                "enabled": True,
                "connected": True,
                "ready": False,
                "message": "Connexion Oracle active, mais tables manquantes: " + ", ".join(missing),
            }
        return {"enabled": True, "connected": True, "ready": True, "message": "Connexion Oracle active et schema IDS pret."}
    except Exception as exc:  # pragma: no cover - depends on local Oracle
        return {"enabled": True, "connected": False, "ready": False, "message": str(exc)}


def _connect():
    config = OracleConfig.from_env()
    if not config.enabled:
        return None
    if not (config.user and config.password and config.dsn):
        raise RuntimeError("Configuration Oracle incomplete.")
    oracledb = _load_driver()
    return oracledb.connect(user=config.user, password=config.password, dsn=config.dsn)


def save_analysis(
    *,
    upload_id: str,
    source: str,
    filename: str,
    result: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        return {"saved": False, "message": "Oracle desactive."}

    metrics_json = json.dumps(result.get("metrics"), ensure_ascii=False) if result.get("metrics") else None
    severity = result.get("status", "safe")
    alert_rate = float(result.get("alert_rate", 0.0))
    message = _alert_message(severity, int(result.get("attack_count", 0)), alert_rate)

    with conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into ids_uploads (
                    upload_id, source_type, filename, model_name, sample_count,
                    attack_count, benign_count, alert_rate, status, metrics_json
                ) values (
                    :upload_id, :source_type, :filename, :model_name, :sample_count,
                    :attack_count, :benign_count, :alert_rate, :status, :metrics_json
                )
                """,
                {
                    "upload_id": upload_id,
                    "source_type": source,
                    "filename": filename,
                    "model_name": str(result.get("model", "decision_tree")),
                    "sample_count": int(result.get("samples", 0)),
                    "attack_count": int(result.get("attack_count", 0)),
                    "benign_count": int(result.get("benign_count", 0)),
                    "alert_rate": alert_rate,
                    "status": severity,
                    "metrics_json": metrics_json,
                },
            )

            rows = [
                {
                    "upload_id": upload_id,
                    "row_number": int(row.ligne),
                    "predicted_class": 0 if str(row.classe_predite).lower().startswith("attaque") else 1,
                    "predicted_label": str(row.classe_predite),
                    "attack_probability": float(row.probabilite_attaque),
                    "benign_probability": float(row.probabilite_benin),
                }
                for row in predictions.itertuples(index=False)
            ]
            if rows:
                cursor.executemany(
                    """
                    insert into ids_predictions (
                        upload_id, row_number, predicted_class, predicted_label,
                        attack_probability, benign_probability
                    ) values (
                        :upload_id, :row_number, :predicted_class, :predicted_label,
                        :attack_probability, :benign_probability
                    )
                    """,
                    rows,
                )

            if severity in {"warning", "danger"}:
                cursor.execute(
                    """
                    insert into ids_alerts (upload_id, severity, message)
                    values (:upload_id, :severity, :message)
                    """,
                    {"upload_id": upload_id, "severity": severity, "message": message},
                )
        conn.commit()

    return {"saved": True, "upload_id": upload_id, "message": "Analyse sauvegardee dans Oracle."}


def list_history(limit: int = 25) -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        return []
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select upload_id, source_type, filename, model_name, sample_count,
                       attack_count, benign_count, alert_rate, status, created_at
                from (
                    select upload_id, source_type, filename, model_name, sample_count,
                           attack_count, benign_count, alert_rate, status, created_at
                    from ids_uploads
                    order by created_at desc
                )
                where rownum <= :limit
                """,
                {"limit": int(limit)},
            )
            return [_row_to_dict(cursor, row) for row in cursor.fetchall()]


def list_alerts(limit: int = 25) -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        return []
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select alert_id, upload_id, severity, message, created_at
                from (
                    select alert_id, upload_id, severity, message, created_at
                    from ids_alerts
                    order by created_at desc
                )
                where rownum <= :limit
                """,
                {"limit": int(limit)},
            )
            return [_row_to_dict(cursor, row) for row in cursor.fetchall()]


def sync_model_results(models: list[dict[str, Any]]) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        return {"saved": False, "message": "Oracle desactive."}
    rows = [
        {
            "model_id": str(model["id"]),
            "model_name": str(model["name"]),
            "model_type": str(model["type"]),
            "accuracy": float(model["accuracy"]),
            "precision_value": float(model["precision"]),
            "recall_value": float(model["recall"]),
            "f1_score": float(model["f1_score"]),
            "attack_f1_score": float(model["attack_f1_score"]),
            "roc_auc": float(model["roc_auc"]),
        }
        for model in models
    ]
    with conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                merge into ids_model_results dst
                using (
                    select :model_id model_id, :model_name model_name, :model_type model_type,
                           :accuracy accuracy, :precision_value precision_value,
                           :recall_value recall_value, :f1_score f1_score,
                           :attack_f1_score attack_f1_score, :roc_auc roc_auc
                    from dual
                ) src
                on (dst.model_id = src.model_id)
                when matched then update set
                    dst.model_name = src.model_name,
                    dst.model_type = src.model_type,
                    dst.accuracy = src.accuracy,
                    dst.precision_value = src.precision_value,
                    dst.recall_value = src.recall_value,
                    dst.f1_score = src.f1_score,
                    dst.attack_f1_score = src.attack_f1_score,
                    dst.roc_auc = src.roc_auc,
                    dst.updated_at = systimestamp
                when not matched then insert (
                    model_id, model_name, model_type, accuracy, precision_value,
                    recall_value, f1_score, attack_f1_score, roc_auc
                ) values (
                    src.model_id, src.model_name, src.model_type, src.accuracy,
                    src.precision_value, src.recall_value, src.f1_score,
                    src.attack_f1_score, src.roc_auc
                )
                """,
                rows,
            )
        conn.commit()
    return {"saved": True, "rows": len(rows), "message": "Resultats modeles synchronises dans Oracle."}


def _row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
    keys = [col[0].lower() for col in cursor.description]
    result: dict[str, Any] = {}
    for key, value in zip(keys, row):
        if isinstance(value, datetime):
            result[key] = value.isoformat(sep=" ", timespec="seconds")
        else:
            result[key] = value
    return result


def _alert_message(severity: str, attack_count: int, alert_rate: float) -> str:
    if severity == "danger":
        return f"Alerte critique: {attack_count} attaque(s) detectee(s), taux {alert_rate:.1%}."
    if severity == "warning":
        return f"Surveillance: {attack_count} attaque(s) detectee(s), taux {alert_rate:.1%}."
    return "Trafic majoritairement benin."
