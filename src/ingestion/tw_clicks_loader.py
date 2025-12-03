"""Загрузка данных data_clicks.csv в базу."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.db.models import DataLoad, TwClicksEvent


def _calculate_file_hash(file_path: Path) -> str:
    hash_sha = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()


def _read_tw_clicks_csv(file_path: Path) -> pd.DataFrame:
    """Читает data_clicks.csv файл."""
    df = pd.read_csv(file_path, sep=";", dtype=str)

    # Проверяем наличие обязательных колонок
    required_columns = {"date", "installations", "income_usd", "first_purchases", "registrations", "clicks"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"В файле data_clicks.csv отсутствуют колонки: {', '.join(sorted(missing))}")

    # Обрабатываем дату
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].notna()]
    # Исключаем строки с некорректной датой 1970-01-01 (обычно означает ошибку парсинга)
    from datetime import date as date_type
    df = df[df["date"] != date_type(1970, 1, 1)]

    # Обрабатываем ad_campaign_name (может быть пустым для non-attribution)
    if "ad_campaign_name" in df.columns:
        df["ad_campaign_name"] = df["ad_campaign_name"].fillna("non-attribution").astype(str).str.strip()
    else:
        df["ad_campaign_name"] = "non-attribution"

    # Обрабатываем geo_country_code
    if "geo_country_code" in df.columns:
        df["geo_country_code"] = df["geo_country_code"].fillna("").astype(str).str.strip().str.upper()
    else:
        df["geo_country_code"] = ""

    # Обрабатываем carrot_id
    if "carrot_id" in df.columns:
        df["carrot_id"] = df["carrot_id"].fillna("").astype(str).str.strip()
    else:
        df["carrot_id"] = ""

    # Обрабатываем числовые поля
    numeric_fields = {
        "clicks": int,
        "installations": int,
        "registrations": int,
        "first_purchases": int,
        "income_usd": float,
    }

    for column, cast_type in numeric_fields.items():
        if column not in df.columns:
            df[column] = 0
        df[column] = (
            pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            .fillna(0)
        )
        if cast_type is int:
            df[column] = df[column].round().astype(int)

    # Переименовываем колонки для соответствия модели БД
    df = df.rename(columns={
        "date": "tw_date",
        "ad_campaign_name": "tw_ad_campaign_name",
        "geo_country_code": "tw_geo_country_code",
        "carrot_id": "tw_carrot_id",
        "installations": "tw_installations",
        "income_usd": "tw_revenue",
        "first_purchases": "tw_deposits",
        "registrations": "tw_registrations",
        "clicks": "tw_clicks",
    })

    return df


def load_tw_clicks_file(file_path: str | Path, session: Session) -> str:
    """Загружает data_clicks.csv, заменяя существующие данные."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден")

    file_hash = _calculate_file_hash(path)
    data_load = DataLoad(
        source="TW_CLICKS",
        file_name=path.name,
        file_hash=file_hash,
        status="processing",
        started_at=datetime.utcnow(),
    )
    session.add(data_load)
    session.flush()

    try:
        df = _read_tw_clicks_csv(path)
        records = df.to_dict(orient="records")

        session.execute(delete(TwClicksEvent))

        payload: List[dict] = []
        for row in records:
            campaign = str(row.get("tw_ad_campaign_name", "")).strip()
            # Для non-attribution сохраняем как строку "non-attribution", не NULL
            if not campaign or campaign == "":
                campaign = "non-attribution"

            payload.append(
                {
                    "load_id": data_load.id,
                    "source_file": path.name,
                    "tw_date": row["tw_date"],
                    "tw_ad_campaign_name": campaign,
                    "tw_geo_country_code": (str(row.get("tw_geo_country_code", "")).strip() or None),
                    "tw_carrot_id": (str(row.get("tw_carrot_id", "")).strip() or None),
                    "tw_clicks": int(float(row.get("tw_clicks", 0) or 0)),
                    "tw_installations": int(float(row.get("tw_installations", 0) or 0)),
                    "tw_registrations": int(float(row.get("tw_registrations", 0) or 0)),
                    "tw_deposits": int(float(row.get("tw_deposits", 0) or 0)),
                    "tw_revenue": float(row.get("tw_revenue", 0) or 0),
                }
            )

        if payload:
            session.bulk_insert_mappings(TwClicksEvent, payload)

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
