"""Построение витрины campaign_statistics."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.db.models import CampaignStatistics, MolocoEvent, TwEvent

logger = logging.getLogger(__name__)


def safe_str(value):
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed or trimmed.lower() == "nan":
            return None
        return trimmed
    if pd.isna(value):
        return None
    return str(value)


def rebuild_statistics(session: Session) -> int:
    """Пересобирает витрину: строки TW + CPI из Moloco."""

    tw_query = select(TwEvent)
    tw_df = pd.read_sql(tw_query, session.bind)

    if tw_df.empty:
        logger.info("TW данные отсутствуют, очищаю витрину")
        session.execute(delete(CampaignStatistics))
        session.commit()
        return 0

    logger.info("Загружено %s строк TW", len(tw_df))

    moloco_query = select(MolocoEvent)
    moloco_df = pd.read_sql(moloco_query, session.bind)
    if moloco_df.empty:
        logger.info("Moloco данные отсутствуют, CPI будут равны 0")
        moloco_df = pd.DataFrame(columns=["m_date", "m_campaign"])
    else:
        logger.info("Загружено %s строк Moloco", len(moloco_df))
        moloco_df["m_date"] = pd.to_datetime(moloco_df["m_date"]).dt.date
        moloco_df = moloco_df.sort_values("m_date")

    tw_df["tw_date"] = pd.to_datetime(tw_df["tw_date"]).dt.date

    numeric_columns = [
        "tw_first_purchases",
        "tw_registrations",
        "tw_installations",
        "tw_install2reg",
        "tw_reg2dep",
        "tw_income_usd",
        "tw_average_bill",
        "tw_purchases_sum",
        "tw_epc",
    ]
    for col in numeric_columns:
        if col not in tw_df.columns:
            tw_df[col] = 0

    # Сначала создаем словарь Moloco
    moloco_map: dict[tuple, dict] = {}
    if not moloco_df.empty:
        moloco_df["m_campaign"] = moloco_df["m_campaign"].astype(str).str.strip()
        moloco_grouped = moloco_df.groupby(["m_date", "m_campaign"], dropna=False).agg(
            {
                "m_impression": "sum",
                "m_click": "sum",
                "m_install": "sum",
                "m_spend": "sum",
                "m_cpi": "mean",
                "m_currency": "last",
            }
        )
        for (m_date, m_campaign), row in moloco_grouped.iterrows():
            moloco_map[(m_date, m_campaign)] = {
                "m_impression": float(row.get("m_impression", 0) or 0),
                "m_click": float(row.get("m_click", 0) or 0),
                "m_install": float(row.get("m_install", 0) or 0),
                "m_spend": float(row.get("m_spend", 0.0) or 0.0),
                "m_cpi": float(row.get("m_cpi", 0.0) or 0.0),
                "m_currency": row.get("m_currency", "USD") or "USD",
            }
    
    # Предварительно считаем статистики по группам для корректного распределения Moloco
    tw_df["tw_ad_campaign_name"] = tw_df["tw_ad_campaign_name"].astype(str).str.strip()
    tw_df["_group_key"] = list(zip(tw_df["tw_date"], tw_df["tw_ad_campaign_name"]))
    installs_per_group = tw_df.groupby("_group_key")["tw_installations"].sum().to_dict()
    rows_per_group = tw_df.groupby("_group_key").size().to_dict()

    # Добавляем данные Moloco к каждой строке TW
    def add_moloco_data(row):
        key = (row["tw_date"], row["tw_ad_campaign_name"])
        moloco_data = moloco_map.get(key)
        if not moloco_data:
            return pd.Series({
                "m_impression": 0.0,
                "m_click": 0.0,
                "m_install": 0.0,
                "m_spend": 0.0,
                "m_cpi": 0.0,
                "m_currency": "USD",
            })

        total_installs = installs_per_group.get(key, 0) or 0
        group_rows = rows_per_group.get(key, 1)
        if total_installs > 0:
            share = (row.get("tw_installations", 0) or 0) / total_installs
        else:
            share = 1 / group_rows if group_rows else 0

        share = float(share)

        allocated_spend = moloco_data["m_spend"] * share
        allocated_impression = moloco_data["m_impression"] * share
        allocated_click = moloco_data["m_click"] * share
        allocated_install = moloco_data["m_install"] * share

        return pd.Series({
            "m_impression": allocated_impression,
            "m_click": allocated_click,
            "m_install": allocated_install,
            "m_spend": allocated_spend,
            "m_cpi": moloco_data.get("m_cpi", 0.0),
            "m_currency": moloco_data.get("m_currency", "USD"),
        })
    
    tw_df["tw_installations"] = tw_df["tw_installations"].fillna(0).astype(int)
    tw_df["tw_registrations"] = tw_df["tw_registrations"].fillna(0).astype(int)
    tw_df["tw_first_purchases"] = tw_df["tw_first_purchases"].fillna(0).astype(int)
    for col in ["tw_install2reg", "tw_reg2dep", "tw_income_usd", "tw_average_bill", "tw_purchases_sum", "tw_epc"]:
        tw_df[col] = tw_df[col].fillna(0).astype(float)
    
    # Применяем функцию добавления данных Moloco
    moloco_cols = tw_df.apply(add_moloco_data, axis=1)
    tw_df = pd.concat([tw_df, moloco_cols], axis=1)

    tw_df["calculated_spend"] = tw_df[
        "m_spend"
    ]
    tw_df["calculated_cpi"] = tw_df.apply(
        lambda r: r["m_spend"] / r["tw_installations"]
        if r["tw_installations"]
        else float(r.get("m_cpi", 0.0)),
        axis=1,
    )
    tw_df["profit"] = tw_df["tw_income_usd"] - tw_df["calculated_spend"]
    tw_df["roi_pct"] = tw_df.apply(
        lambda r: (r["profit"] / r["calculated_spend"] * 100) if r["calculated_spend"] else 0.0,
        axis=1,
    )

    tw_df = tw_df.drop(columns=["_group_key"], errors="ignore")

    session.execute(delete(CampaignStatistics))

    records = []
    for row in tw_df.to_dict(orient="records"):
        campaign = row.get("tw_ad_campaign_name")
        if not campaign:
            # пробуем найти прошлую кампанию по офферу и гео
            campaign = (
                session.query(CampaignStatistics.tw_ad_campaign_name)
                .filter(
                    CampaignStatistics.tw_link_title == row.get("tw_link_title"),
                    CampaignStatistics.tw_geo_country_code == row.get("tw_geo_country_code"),
                )
                .order_by(CampaignStatistics.stat_date.desc())
                .limit(1)
                .scalar()
            )
        if not campaign:
            # если не нашли – пропускаем строку
            continue

        records.append(
            {
                "stat_date": row.get("tw_date"),
                "tw_ad_campaign_name": campaign,
                "tw_link_title": row.get("tw_link_title"),
                "tw_geo_country_code": row.get("tw_geo_country_code"),
                "tw_carrot_id": row.get("tw_carrot_id"),
                "tw_first_purchases": int(row.get("tw_first_purchases", 0) or 0),
                "tw_registrations": int(row.get("tw_registrations", 0) or 0),
                "tw_installations": int(row.get("tw_installations", 0) or 0),
                "tw_install2reg": float(row.get("tw_install2reg", 0) or 0),
                "tw_reg2dep": float(row.get("tw_reg2dep", 0) or 0),
                "tw_income_usd": float(row.get("tw_income_usd", 0) or 0),
                "tw_average_bill": float(row.get("tw_average_bill", 0) or 0),
                "tw_purchases_sum": float(row.get("tw_purchases_sum", 0) or 0),
                "tw_epc": float(row.get("tw_epc", 0) or 0),
                "m_campaign": campaign,
                "m_impression": int(row.get("m_impression", 0) or 0),
                "m_click": int(row.get("m_click", 0) or 0),
                "m_install": int(row.get("m_install", 0) or 0),
                "m_spend": float(row.get("m_spend", 0) or 0),
                "m_cpi": float(row.get("m_cpi", 0) or 0),
                "m_currency": row.get("m_currency", "USD"),
                "calculated_spend": float(row.get("calculated_spend", 0) or 0),
                "calculated_cpi": float(row.get("calculated_cpi", 0) or 0),
                "profit": float(row.get("profit", 0) or 0),
                "roi_pct": float(row.get("roi_pct", 0) or 0),
                "created_at": datetime.utcnow(),
            }
        )

    session.bulk_insert_mappings(CampaignStatistics, records)
    session.commit()

    logger.info("Записано %s строк в campaign_statistics", len(records))
    return len(records)


