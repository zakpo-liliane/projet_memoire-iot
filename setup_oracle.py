from __future__ import annotations

import argparse
import getpass
import re
from pathlib import Path

import oracledb


ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "sql" / "oracle_schema.sql"


def split_sql_script(text: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    in_plsql = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if stripped.lower().startswith("set "):
            continue
        if stripped == "/":
            statement = "\n".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            in_plsql = False
            continue

        buffer.append(line)
        lowered = stripped.lower()
        if re.match(r"^(declare|begin)\b", lowered):
            in_plsql = True
        if not in_plsql and stripped.endswith(";"):
            statement = "\n".join(buffer).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            buffer = []

    statement = "\n".join(buffer).strip().rstrip(";").strip()
    if statement:
        statements.append(statement)
    return statements


def run_setup(user: str, password: str, dsn: str, sysdba: bool) -> None:
    mode = oracledb.AUTH_MODE_SYSDBA if sysdba else oracledb.AUTH_MODE_DEFAULT
    script = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = split_sql_script(script)

    with oracledb.connect(user=user, password=password, dsn=dsn, mode=mode) as conn:
        with conn.cursor() as cursor:
            for statement in statements:
                preview = " ".join(statement.split())[:90]
                try:
                    cursor.execute(statement)
                    print(f"OK: {preview}")
                except oracledb.DatabaseError as exc:
                    print(f"ERREUR: {preview}")
                    raise exc
        conn.commit()

    print("Installation Oracle terminee.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Installer le schema Oracle de l'application IDS IIoT.")
    parser.add_argument("--dsn", default="192.168.56.1:1521/XE")
    parser.add_argument("--user", default="SYS")
    parser.add_argument("--no-sysdba", action="store_true")
    args = parser.parse_args()

    password = getpass.getpass(f"Mot de passe Oracle pour {args.user}: ")
    run_setup(args.user, password, args.dsn, sysdba=not args.no_sysdba)


if __name__ == "__main__":
    main()
