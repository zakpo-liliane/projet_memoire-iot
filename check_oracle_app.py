from __future__ import annotations

import oracle_store


def main() -> None:
    status = oracle_store.status()
    print("Statut Oracle:")
    for key, value in status.items():
        print(f"- {key}: {value}")

    if not status.get("connected"):
        raise SystemExit(1)
    if not status.get("ready"):
        raise SystemExit(2)

    history = oracle_store.list_history(limit=5)
    alerts = oracle_store.list_alerts(limit=5)
    print(f"Historique lisible: {len(history)} ligne(s)")
    print(f"Alertes lisibles: {len(alerts)} ligne(s)")
    print("Verification Oracle application OK.")


if __name__ == "__main__":
    main()
