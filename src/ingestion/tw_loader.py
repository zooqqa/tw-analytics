"""Загрузка данных TrafficWave в базу."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.db.models import DataLoad, TwEvent


REQUIRED_TW_COLUMNS = {
    "date",
    "link_title",
    "ad_campaign_name",
    "geo_country_code",
    "carrot_id",
    "first_purchases",
    "registrations",
    "installations",
    "install2reg",
    "reg2dep",
    "income_usd",
    "epc",
}

# Опциональные колонки (могут присутствовать в старом или новом формате)
OPTIONAL_TW_COLUMNS = {
    "average_bill",
    "purchases_sum",
    "d14_aov",
    "d14_purchases_sum",
}


def _validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_TW_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"В TW файле отсутствуют колонки: {', '.join(sorted(missing))}")


def _calculate_file_hash(file_path: Path) -> str:
    hash_sha = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()


def _read_tw_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";", dtype=str)
    _validate_columns(df.columns)

    # Очищаем данные
    df = df[df["link_title"].notna()]
    df["link_title"] = df["link_title"].astype(str).str.strip()
    df = df[df["link_title"] != ""]
    df = df[df["link_title"] != "0"]

    if "geo_country_code" in df.columns:
        df["geo_country_code"] = df["geo_country_code"].fillna("").astype(str).str.strip().str.upper()
    else:
        df["geo_country_code"] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].notna()]

    # Заполняем пропущенные названия кампаний по предыдущим записям с тем же оффером
    if "ad_campaign_name" in df.columns:
        df["ad_campaign_name"] = df.groupby(["link_title", "geo_country_code"])["ad_campaign_name"].transform(
            lambda s: s.ffill().bfill()
        )
        df["ad_campaign_name"] = df["ad_campaign_name"].fillna("")

    numeric_fields = {
        "first_purchases": int,
        "registrations": int,
        "installations": int,
        "install2reg": float,
        "reg2dep": float,
        "income_usd": float,
        "average_bill": float,
        "purchases_sum": float,
        "d14_aov": float,
        "d14_purchases_sum": float,
        "epc": float,
    }

    for column, cast_type in numeric_fields.items():
        if column not in df.columns:
            continue  # Пропускаем если колонки нет
        df[column] = (
            pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            .fillna(0)
        )
        if cast_type is int:
            df[column] = df[column].round().astype(int)

    # Удаляем дубликаты по ключу кампания+оффер+дата (берем последнюю)
    return df


def load_tw_file(file_path: str | Path, session: Session) -> str:
    """Загружает TW CSV, заменяя существующие данные."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден")

    file_hash = _calculate_file_hash(path)
    data_load = DataLoad(
        source="TW",
        file_name=path.name,
        file_hash=file_hash,
        status="processing",
        started_at=datetime.utcnow(),
    )
    session.add(data_load)
    session.flush()

    try:
        df = _read_tw_csv(path)
        records = df.to_dict(orient="records")

        session.execute(delete(TwEvent))

        payload: List[dict] = []
        for row in records:
            campaign = str(row.get("ad_campaign_name", "")).strip()
            if not campaign:
                # пропускаем строки без кампании: не смогли восстановить
                continue

            payload.append(
                {
                    "load_id": data_load.id,
                    "source_file": path.name,
                    "tw_date": row["date"],
                    "tw_link_title": str(row.get("link_title", "")).strip(),
                    "tw_ad_campaign_name": campaign,
                    "tw_geo_country_code": (str(row.get("geo_country_code", "")).strip() or None),
                    "tw_carrot_id": (str(row.get("carrot_id", "")).strip() or None),
                    "tw_first_purchases": int(float(row["first_purchases"] or 0)),
                    "tw_registrations": int(float(row["registrations"] or 0)),
                    "tw_installations": int(float(row["installations"] or 0)),
                    "tw_install2reg": float(row.get("install2reg", 0) or 0),
                    "tw_reg2dep": float(row.get("reg2dep", 0) or 0),
                    "tw_income_usd": float(row.get("income_usd", 0) or 0),
                    "tw_average_bill": float(row.get("average_bill", 0) or 0),
                    "tw_purchases_sum": float(row.get("purchases_sum", 0) or 0),
                    "tw_d14_aov": float(row.get("d14_aov", 0) or 0),
                    "tw_d14_purchases_sum": float(row.get("d14_purchases_sum", 0) or 0),
                    "tw_epc": float(row.get("epc", 0) or 0),
                }
            )

        if payload:
            session.bulk_insert_mappings(TwEvent, payload)

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


