"""CLI для запуска загрузки и построения витрины."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.db.session import SessionLocal
from src.db.setup import ensure_schema
from src.ingestion import load_moloco_file, load_tw_file
from src.services import rebuild_statistics


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="ETL для TW и Moloco")
    parser.add_argument("--tw", dest="tw_path", help="Путь к файлу TW", required=False)
    parser.add_argument("--moloco", dest="moloco_path", help="Путь к файлу Moloco", required=False)
    args = parser.parse_args()

    ensure_schema()
    session = SessionLocal()
    try:
        if args.tw_path:
            load_tw_file(args.tw_path, session)
        if args.moloco_path:
            load_moloco_file(args.moloco_path, session)

        rebuild_statistics(session)
    finally:
        session.close()


if __name__ == "__main__":
    run_cli()


