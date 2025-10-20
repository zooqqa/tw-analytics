"""Загрузка данных Moloco в базу."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.db.models import DataLoad, MolocoEvent


REQUIRED_MOLOCO_COLUMNS = {
    "Date",
    "Campaign",
    "Impression",
    "Click",
    "Install",
    "Spend",
    "CPI",
}


def _validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_MOLOCO_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"В Moloco файле отсутствуют колонки: {', '.join(sorted(missing))}")


def _calculate_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_moloco_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    _validate_columns(df.columns)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df[df["Date"].notna()]

    for column in ["Impression", "Click", "Install"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    for column in ["Spend", "CPI"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .apply(lambda x: float(x) if x not in ("", "nan", None) else 0.0)
        )

    numeric_columns = [
        "Impression",
        "Click",
        "Install",
        "Spend",
        "CPI",
        "CPA",
        "Cost per Conversion",
        "first_purchase",
        "registration",
        "purchase",
    ]
    present_numeric = [col for col in numeric_columns if col in df.columns]
    for column in present_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    agg_dict = {col: "sum" for col in present_numeric if col != "CPI" and col != "CPA" and col != "Cost per Conversion"}
    agg_dict.update({"CPI": "sum"}) if "CPI" in present_numeric else None
    agg_dict.update({"CPA": "sum"}) if "CPA" in present_numeric else None
    agg_dict.update({"Cost per Conversion": "sum"}) if "Cost per Conversion" in present_numeric else None
    if "Currency" in df.columns:
        agg_dict["Currency"] = "first"

    grouped = df.groupby(["Date", "Campaign"], as_index=False).agg(agg_dict)

    if "Install" in grouped.columns:
        grouped["Install"] = grouped["Install"].astype(float)
    if "Spend" in grouped.columns:
        grouped["Spend"] = grouped["Spend"].astype(float)

    if "CPI" in grouped.columns:
        grouped["CPI"] = grouped.apply(
            lambda row: row["Spend"] / row["Install"] if row.get("Install", 0) else 0.0,
            axis=1,
        )
    if "CPA" in grouped.columns and "first_purchase" in grouped.columns:
        grouped["CPA"] = grouped.apply(
            lambda row: row["Spend"] / row["first_purchase"] if row.get("first_purchase", 0) else 0.0,
            axis=1,
        )
    if "Cost per Conversion" in grouped.columns and "registration" in grouped.columns:
        grouped["Cost per Conversion"] = grouped.apply(
            lambda row: row["Spend"] / row["registration"] if row.get("registration", 0) else 0.0,
            axis=1,
        )

    return grouped


def load_moloco_file(file_path: str | Path, session: Session) -> str:
    """Загружает Moloco CSV, заменяя существующие данные."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден")

    file_hash = _calculate_hash(path)
    data_load = DataLoad(
        source="MOLOCO",
        file_name=path.name,
        file_hash=file_hash,
        status="processing",
        started_at=datetime.utcnow(),
    )
    session.add(data_load)
    session.flush()

    try:
        df = _read_moloco_csv(path)
        records = df.to_dict(orient="records")

        session.execute(delete(MolocoEvent))

        payload: List[dict] = []
        for row in records:
            payload.append(
                {
                    "load_id": data_load.id,
                    "source_file": path.name,
                    "m_date": row["Date"],
                    "m_campaign": row["Campaign"].strip(),
                    "m_impression": row.get("Impression", 0),
                    "m_click": row.get("Click", 0),
                    "m_install": row.get("Install", 0),
                    "m_spend": float(row.get("Spend", 0) or 0),
                    "m_cpi": float(row.get("CPI", 0) or 0),
                    "m_currency": (row.get("Currency") or "USD").strip() or "USD",
                }
            )

        if payload:
            session.bulk_insert_mappings(MolocoEvent, payload)

        data_load.records_total = len(df)
        data_load.records_valid = len(payload)
        data_load.status = "completed"
        data_load.finished_at = datetime.utcnow()
        session.commit()
    except Exception as exc:
        session.rollback()
        data_load.status = "failed"
        data_load.error_log = str(exc)
        data_load.finished_at = datetime.utcnow()
        session.add(data_load)
        session.commit()
        raise

    return data_load.id


