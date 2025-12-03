"""FastAPI приложение для загрузки и выдачи статистики."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from src.db.models import CampaignStatistics, DataLoad, MolocoEvent, TwClicksEvent
from src.db.session import get_session
from src.db.setup import ensure_schema
from src.ingestion import load_moloco_file, load_tw_file, load_tw_clicks_file
from src.services import rebuild_statistics

# Мета информация о колонках
METADATA = [
    {"key": "stat_date", "dimension": True},
    {"key": "tw_ad_campaign_name", "dimension": True},
    {"key": "tw_link_title", "dimension": True},
    {"key": "tw_carrot_id", "dimension": True},
    {"key": "tw_geo_country_code", "dimension": True},
    {"key": "m_campaign", "dimension": True},
    {"key": "m_creative", "dimension": True},
    {"key": "m_countries", "dimension": True},
    {"key": "tw_installations", "dimension": False},
    {"key": "tw_install2reg", "dimension": False},
    {"key": "tw_registrations", "dimension": False},
    {"key": "tw_reg2dep", "dimension": False},
    {"key": "tw_first_purchases", "dimension": False},
    {"key": "tw_average_bill", "dimension": False},
    {"key": "tw_purchases_sum", "dimension": False},
    {"key": "tw_epc", "dimension": False},
    {"key": "tw_income_usd", "dimension": False},
    {"key": "m_impression", "dimension": False},
    {"key": "m_click", "dimension": False},
    {"key": "m_install", "dimension": False},
    {"key": "m_spend", "dimension": False},
    {"key": "m_cpi", "dimension": False},
    {"key": "m_ctr", "dimension": False},
    {"key": "m_conversion", "dimension": False},
    {"key": "m_cost_per_conversion", "dimension": False},
    {"key": "m_registration", "dimension": False},
    {"key": "m_first_purchase", "dimension": False},
    {"key": "m_purchase", "dimension": False},
    {"key": "calculated_spend", "dimension": False},
    {"key": "calculated_cpi", "dimension": False},
    {"key": "profit", "dimension": False},
    {"key": "roi_pct", "dimension": False},
    {"key": "kpi_pct", "dimension": False},
    {"key": "click2inst", "dimension": False},
    {"key": "inst2dep", "dimension": False},
    {"key": "inst2reg", "dimension": False},
    {"key": "revenue", "dimension": False},
]
ALL_COLUMNS = [item["key"] for item in METADATA]
COLUMN_SET = set(ALL_COLUMNS)
DEFAULT_COLUMNS = [
    item["key"]
    for item in METADATA
    if item["key"] not in {"tw_ad_campaign_name", "tw_link_title", "tw_carrot_id", "tw_geo_country_code"}
]
FLOAT_FIELDS = {
    "tw_install2reg",
    "tw_reg2dep",
    "tw_average_bill",
    "tw_epc",
    "tw_income_usd",
    "tw_purchases_sum",
    "tw_d14_aov",
    "tw_d14_purchases_sum",
    "m_cpi",
    "m_spend",
    "m_ctr",
    "m_cost_per_conversion",
    "calculated_spend",
    "calculated_cpi",
    "profit",
    "roi_pct",
    "kpi_pct",
    "click2inst",
    "inst2dep",
    "inst2reg",
    "revenue",
    "tw_revenue",
    "calc_click2inst",
    "calc_reg2dep",
    "calc_inst2dep",
    "calc_profit",
    "calc_$inst",
    "calc_$reg",
    "calc_$dep",
    "calc_roi",
}
INT_FIELDS = {
    "tw_installations",
    "tw_registrations",
    "tw_first_purchases",
    "m_impression",
    "m_click",
    "m_install",
    "m_conversion",
    "m_registration",
    "m_first_purchase",
    "m_purchase",
    "tw_click",
    "tw_install",
    "tw_registration",
    "tw_deposits",
}
STRING_FIELDS = {
    "stat_date",
    "tw_ad_campaign_name",
    "tw_link_title",
    "tw_carrot_id",
    "tw_geo_country_code",
    "m_campaign",
    "m_creative",
    "m_countries",
}
NUMERIC_FIELDS = FLOAT_FIELDS | INT_FIELDS
DIMENSION_COLUMNS = [item["key"] for item in METADATA if item["dimension"]]
AGG_NUMERIC_COLUMNS = {
    "tw_first_purchases",
    "tw_registrations",
    "tw_installations",
    "tw_income_usd",
    "tw_purchases_sum",
    "tw_d14_aov",
    "tw_d14_purchases_sum",
    "m_impression",
    "m_click",
    "m_install",
    "m_spend",
    "m_cpi",
    "m_conversion",
    "m_registration",
    "m_first_purchase",
    "m_purchase",
    "calculated_spend",
    "calculated_cpi",
    "profit",
    "roi_pct",
    "kpi_pct",
    "revenue",
}


def select_columns(raw: Optional[str]) -> list[str]:
    if raw and raw.strip():
        columns = [item.strip() for item in raw.split(",") if item.strip() in COLUMN_SET]
        if columns:
            return columns
    return DEFAULT_COLUMNS.copy()


DEFAULT_COLUMNS_MOLOCO_CAMP = [
    "m_campaign",
    "m_impression",
    "m_click",
    "m_install",
    "tw_click",
    "tw_install",
    "tw_registration",
    "tw_deposits",
    "calc_click2inst",
    "calc_reg2dep",
    "calc_inst2dep",
    "calc_$inst",
    "calc_$reg",
    "calc_$dep",
    "m_spend",
    "tw_revenue",
    "calc_profit",
    "calc_roi",
]


DEFAULT_COLUMNS_MOLOCO_CREO = [
    "m_creative",
    "m_impression",
    "m_click",
    "m_ctr",
    "m_conversion",
    "m_cost_per_conversion",
    "m_purchase",
    "tw_d14_purchases_sum",
    "click2inst",
    "inst2dep",
    "inst2reg",
    "revenue",
    "profit",
    "roi_pct",
]


DEFAULT_COLUMNS_TW_OFFER = [
    "tw_ad_campaign_name",
    "tw_geo_country_code",
    "tw_installations",
    "inst2reg",
    "tw_registrations",
    "tw_reg2dep",
    "tw_first_purchases",
    "tw_d14_aov",
    "tw_d14_purchases_sum",
    "revenue",
    "calculated_spend",
    "profit",
    "roi_pct",
]


def select_columns_for_tab(raw: Optional[str], tab_type: str) -> list[str]:
    """Выбирает колонки для конкретного таба с учетом дефолтных значений."""
    if raw and raw.strip():
        columns = [item.strip() for item in raw.split(",") if item.strip() in COLUMN_SET]
        if columns:
            return columns
    
    if tab_type == "moloco-camp":
        return DEFAULT_COLUMNS_MOLOCO_CAMP.copy()
    elif tab_type == "moloco-creo":
        return DEFAULT_COLUMNS_MOLOCO_CREO.copy()
    elif tab_type == "tw-offer":
        return DEFAULT_COLUMNS_TW_OFFER.copy()
    return DEFAULT_COLUMNS.copy()


def apply_filters(rows: list[CampaignStatistics], params) -> list[CampaignStatistics]:
    date_from = params.get("date_from")
    date_to = params.get("date_to")
    tw_campaign = params.get("filter_tw_campaign", "").lower()
    tw_campaign_exclude = params.get("filter_tw_campaign_exclude", "").lower()
    tw_offer = params.get("filter_tw_offer", "").lower()
    tw_offer_exclude = params.get("filter_tw_offer_exclude", "").lower()
    m_campaign = params.get("filter_m_campaign", "").lower()
    m_campaign_exclude = params.get("filter_m_campaign_exclude", "").lower()
    geo = params.get("filter_geo", "").strip().upper()
    geo_exclude = params.get("filter_geo_exclude", "").strip().upper()
    
    # Числовые фильтры
    profit_min = params.get("profit_min")
    profit_max = params.get("profit_max")
    roi_min = params.get("roi_min")
    roi_max = params.get("roi_max")
    spend_min = params.get("spend_min")
    spend_max = params.get("spend_max")
    installs_min = params.get("installs_min")
    installs_max = params.get("installs_max")

    filtered = []
    for row in rows:
        # Фильтр по датам
        if date_from and row.stat_date and row.stat_date < date.fromisoformat(date_from):
            continue
        if date_to and row.stat_date and row.stat_date > date.fromisoformat(date_to):
            continue
        
        # Фильтр по кампании (содержит)
        if tw_campaign and (row.tw_ad_campaign_name or "").lower().find(tw_campaign) == -1:
            continue
        # Фильтр по кампании (не содержит)
        if tw_campaign_exclude and (row.tw_ad_campaign_name or "").lower().find(tw_campaign_exclude) != -1:
            continue
        
        # Фильтр по офферу (содержит)
        if tw_offer and (row.tw_link_title or "").lower().find(tw_offer) == -1:
            continue
        # Фильтр по офферу (не содержит)
        if tw_offer_exclude and (row.tw_link_title or "").lower().find(tw_offer_exclude) != -1:
            continue
        
        # Фильтр по Moloco кампании (содержит)
        if m_campaign and (row.m_campaign or "").lower().find(m_campaign) == -1:
            continue
        # Фильтр по Moloco кампании (не содержит)
        if m_campaign_exclude and (row.m_campaign or "").lower().find(m_campaign_exclude) != -1:
            continue
        
        # Фильтр по гео (содержит)
        if geo and (row.tw_geo_country_code or "").upper().find(geo) == -1:
            continue
        # Фильтр по гео (не содержит)
        if geo_exclude and (row.tw_geo_country_code or "").upper().find(geo_exclude) != -1:
            continue
        
        # Числовые фильтры
        row_profit = float(row.profit or 0)
        row_roi = float(row.roi_pct or 0)
        row_spend = float(row.calculated_spend or 0)
        row_installs = int(row.tw_installations or 0)
        
        if profit_min is not None and row_profit < float(profit_min):
            continue
        if profit_max is not None and row_profit > float(profit_max):
            continue
        if roi_min is not None and row_roi < float(roi_min):
            continue
        if roi_max is not None and row_roi > float(roi_max):
            continue
        if spend_min is not None and row_spend < float(spend_min):
            continue
        if spend_max is not None and row_spend > float(spend_max):
            continue
        if installs_min is not None and row_installs < int(installs_min):
            continue
        if installs_max is not None and row_installs > int(installs_max):
            continue
        
        filtered.append(row)
    return filtered


def decimal_to_float(value: Optional[Decimal]) -> float:
    if value is None:
        return 0.0
    return float(value)


def row_to_dict(row: CampaignStatistics) -> dict:
    return {
        "stat_date": row.stat_date.isoformat() if row.stat_date else None,
        "tw_ad_campaign_name": row.tw_ad_campaign_name,
        "tw_link_title": row.tw_link_title,
        "tw_carrot_id": row.tw_carrot_id,
        "tw_geo_country_code": row.tw_geo_country_code,
        "m_campaign": row.m_campaign,
        "m_creative": row.m_creative,
        "m_countries": None,  # Будет заполняться из MolocoEvent при необходимости
        "tw_first_purchases": row.tw_first_purchases or 0,
        "tw_registrations": row.tw_registrations or 0,
        "tw_installations": row.tw_installations or 0,
        "tw_install2reg": float(row.tw_install2reg or 0),
        "tw_reg2dep": float(row.tw_reg2dep or 0),
        "tw_income_usd": decimal_to_float(row.tw_income_usd),
        "tw_average_bill": float(row.tw_average_bill or 0),
        "tw_purchases_sum": decimal_to_float(row.tw_purchases_sum),
        "tw_epc": float(row.tw_epc or 0),
        "m_impression": row.m_impression or 0,
        "m_click": row.m_click or 0,
        "m_install": row.m_install or 0,
        "m_spend": decimal_to_float(row.m_spend),
        "m_cpi": float(row.m_cpi or 0),
        "m_ctr": float(row.m_ctr or 0),
        "m_conversion": row.m_conversion or 0,
        "m_cost_per_conversion": float(row.m_cpa or 0),  # Используем m_cpa как cost_per_conversion
        "m_registration": row.m_registration or 0,
        "m_first_purchase": row.m_first_purchase or 0,
        "m_purchase": row.m_purchase or 0,
        "calculated_spend": decimal_to_float(row.calculated_spend),
        "calculated_cpi": float(row.calculated_cpi or 0),
        "profit": decimal_to_float(row.profit),
        "roi_pct": float(row.roi_pct or 0),
        "revenue": decimal_to_float(row.tw_income_usd),  # revenue = tw_income_usd
    }


def group_key(row: dict, dimensions: list[str]) -> tuple:
    return tuple(row.get(dim) for dim in dimensions)


def aggregate_dataset(rows: list[CampaignStatistics], selected_columns: list[str]) -> tuple[list[dict], list[str]]:
    dataset = [row_to_dict(r) for r in rows]
    dimensions = [col for col in selected_columns if col in DIMENSION_COLUMNS]
    if not dimensions:
        dimensions = ["tw_link_title", "tw_ad_campaign_name"]
        if "tw_link_title" not in selected_columns:
            selected_columns.append("tw_link_title")
        if "tw_ad_campaign_name" not in selected_columns:
            selected_columns.append("tw_ad_campaign_name")

    buckets: dict[tuple, dict] = {}
    sums: dict[tuple, dict] = {}
    actual_spend_totals: dict[tuple, float] = {}
    for row in dataset:
        key = group_key(row, dimensions)
        bucket = buckets.setdefault(key, {dim: row.get(dim) for dim in dimensions})
        acc = sums.setdefault(key, {col: 0.0 for col in AGG_NUMERIC_COLUMNS})

        for col in selected_columns:
            if col in STRING_FIELDS:
                bucket.setdefault(col, row.get(col))

        for col in AGG_NUMERIC_COLUMNS:
            acc[col] = acc.get(col, 0.0) + float(row.get(col) or 0.0)
            if col in selected_columns:
                bucket[col] = acc[col]

        row_m_spend = float(row.get("m_spend") or 0.0)
        row_calculated_spend = float(row.get("calculated_spend") or 0.0)
        actual_spend_value = row_m_spend if row_m_spend > 0 else row_calculated_spend
        actual_spend_totals[key] = actual_spend_totals.get(key, 0.0) + actual_spend_value

    for key, bucket in buckets.items():
        acc = sums[key]
        installs = acc.get("tw_installations", 0.0)
        regs = acc.get("tw_registrations", 0.0)
        ftd = acc.get("tw_first_purchases", 0.0)
        income = acc.get("tw_income_usd", 0.0)
        spend = actual_spend_totals.get(key, 0.0)
        m_spend = acc.get("m_spend", 0.0)
        purchases_sum = acc.get("tw_purchases_sum", 0.0)

        # Используем m_spend если есть, иначе calculated_spend
        actual_spend = spend
        
        bucket["tw_installations"] = installs
        bucket["tw_registrations"] = regs
        bucket["tw_first_purchases"] = ftd
        bucket["tw_income_usd"] = income
        bucket["tw_purchases_sum"] = purchases_sum
        bucket["m_spend"] = m_spend
        bucket["calculated_spend"] = actual_spend
        bucket["calculated_cpi"] = actual_spend / installs if installs else 0.0
        bucket["m_cpi"] = m_spend / installs if installs else 0.0
        bucket["tw_install2reg"] = (regs / installs * 100) if installs else 0.0
        bucket["tw_reg2dep"] = (ftd / regs * 100) if regs else 0.0
        bucket["tw_average_bill"] = purchases_sum / ftd if ftd else 0.0
        bucket["profit"] = income - actual_spend
        bucket["roi_pct"] = (bucket["profit"] / actual_spend * 100) if actual_spend else 0.0

    return list(buckets.values()), dimensions


def filter_zero_rows(rows: list[dict], columns: list[str]) -> list[dict]:
    numeric_columns = [col for col in columns if col in NUMERIC_FIELDS]
    metrics = numeric_columns or list(NUMERIC_FIELDS)
    return [row for row in rows if any((row.get(col) or 0) != 0 for col in metrics)]


def format_row(row: dict, columns: list[str]) -> dict:
    formatted = {}
    for column in columns:
        value = row.get(column)
        # Если значение None и это числовое поле, используем 0
        if value is None:
            if column in FLOAT_FIELDS:
                value = 0.0
            elif column in INT_FIELDS:
                value = 0
        if column in FLOAT_FIELDS:
            formatted[column] = float(value or 0)
        elif column in INT_FIELDS:
            formatted[column] = int(float(value or 0))
        else:
            formatted[column] = value
    return formatted


def calculate_total_row(data: list[dict], columns: list[str]) -> dict:
    """Рассчитывает строку Total с суммами для числовых колонок и пересчетом процентов."""
    if not data:
        return {}
    
    total = {}
    
    # Суммируем числовые колонки
    for col in columns:
        if col in INT_FIELDS or col in FLOAT_FIELDS:
            total[col] = sum(float(row.get(col) or 0) for row in data)
        else:
            total[col] = "Total"
    
    # Пересчитываем процентные метрики на основе сумм
    total_installs = total.get("tw_installations", 0.0) or total.get("m_install", 0.0) or 0.0
    total_regs = total.get("tw_registrations", 0.0) or total.get("m_registration", 0.0) or 0.0
    total_ftd = total.get("tw_first_purchases", 0.0) or total.get("m_first_purchase", 0.0) or 0.0
    total_clicks = total.get("m_click", 0.0) or 0.0
    total_revenue = total.get("revenue", 0.0) or total.get("tw_income_usd", 0.0) or 0.0
    total_spend = total.get("calculated_spend", 0.0) or total.get("m_spend", 0.0) or 0.0
    total_profit = total.get("profit", 0.0) or (total_revenue - total_spend)
    total_conversion = total.get("m_conversion", 0.0) or 0.0
    
    # Пересчет процентных метрик
    if "tw_install2reg" in columns or "inst2reg" in columns:
        total["tw_install2reg"] = (total_regs / total_installs * 100) if total_installs else 0.0
        total["inst2reg"] = (total_regs / total_installs * 100) if total_installs else 0.0
    if "tw_reg2dep" in columns or "inst2dep" in columns:
        total["tw_reg2dep"] = (total_ftd / total_regs * 100) if total_regs else 0.0
        total["inst2dep"] = (total_ftd / total_installs * 100) if total_installs else 0.0
    if "click2inst" in columns:
        total["click2inst"] = (total_installs / total_clicks * 100) if total_clicks else 0.0
    if "roi_pct" in columns:
        total["roi_pct"] = (total_profit / total_spend * 100) if total_spend else 0.0
    if "m_cost_per_conversion" in columns:
        total["m_cost_per_conversion"] = (total_spend / total_conversion) if total_conversion else 0.0
    if "m_ctr" in columns:
        total_impressions = total.get("m_impression", 0.0) or 0.0
        total["m_ctr"] = (total_clicks / total_impressions * 100) if total_impressions else 0.0
    if "calculated_cpi" in columns or "m_cpi" in columns:
        total["calculated_cpi"] = (total_spend / total_installs) if total_installs else 0.0
        total["m_cpi"] = (total_spend / total_installs) if total_installs else 0.0
    
    total["profit"] = total_profit
    total["revenue"] = total_revenue
    
    return total


def aggregate_moloco_camp(session: Session, selected_columns: list[str], date_from: date | None = None, date_to: date | None = None) -> list[dict]:
    """Агрегирует данные по кампаниям Moloco с привязкой данных из TW data_clicks.csv."""
    import pandas as pd
    from sqlalchemy import select
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Локальные определения для проверки типов полей
    float_fields = {
        "m_spend", "tw_revenue", "calc_click2inst", "calc_reg2dep", "calc_inst2dep",
        "calc_profit", "calc_$inst", "calc_$reg", "calc_$dep", "calc_roi"
    }
    int_fields = {
        "m_impression", "m_click", "m_install", "tw_click", "tw_install",
        "tw_registration", "tw_deposits"
    }
    
    # Загружаем данные из Moloco
    moloco_query = select(MolocoEvent)
    moloco_df = pd.read_sql(moloco_query, session.bind)
    
    if moloco_df.empty:
        logger.warning("Moloco данные отсутствуют")
        return []
    
    logger.info(f"Загружено {len(moloco_df)} строк из Moloco")
    
    # Проверяем наличие нужных столбцов
    required_moloco_cols = ["m_date", "m_campaign", "m_impression", "m_click", "m_install", "m_spend"]
    missing_moloco = [col for col in required_moloco_cols if col not in moloco_df.columns]
    if missing_moloco:
        logger.error(f"В Moloco данных отсутствуют столбцы: {missing_moloco}")
        return []
    
    moloco_df["m_date"] = pd.to_datetime(moloco_df["m_date"]).dt.date
    moloco_df["m_campaign"] = moloco_df["m_campaign"].astype(str).str.strip()
    
    # Фильтруем данные Moloco по датам если указаны
    if date_from:
        moloco_df = moloco_df[moloco_df["m_date"] >= pd.Timestamp(date_from).date()]
    if date_to:
        moloco_df = moloco_df[moloco_df["m_date"] <= pd.Timestamp(date_to).date()]
    
    # Загружаем данные из TW clicks
    tw_clicks_query = select(TwClicksEvent)
    tw_clicks_df = pd.read_sql(tw_clicks_query, session.bind)
    
    if tw_clicks_df.empty:
        logger.warning("TW clicks данные отсутствуют")
        tw_grouped = pd.DataFrame(columns=["tw_date", "tw_ad_campaign_name", "tw_clicks", "tw_installations", "tw_registrations", "tw_deposits", "tw_revenue"])
    else:
        logger.info(f"Загружено {len(tw_clicks_df)} строк из TW clicks")
        
        # Проверяем наличие нужных столбцов
        required_tw_cols = ["tw_date", "tw_ad_campaign_name", "tw_clicks", "tw_installations", "tw_registrations", "tw_deposits", "tw_revenue"]
        missing_tw = [col for col in required_tw_cols if col not in tw_clicks_df.columns]
        if missing_tw:
            logger.error(f"В TW clicks данных отсутствуют столбцы: {missing_tw}")
            tw_grouped = pd.DataFrame(columns=["tw_date", "tw_ad_campaign_name", "tw_clicks", "tw_installations", "tw_registrations", "tw_deposits", "tw_revenue"])
        else:
            tw_clicks_df["tw_date"] = pd.to_datetime(tw_clicks_df["tw_date"]).dt.date
            tw_clicks_df["tw_ad_campaign_name"] = tw_clicks_df["tw_ad_campaign_name"].fillna("non-attribution").astype(str).str.strip()
            
            # Фильтруем данные TW clicks по датам если указаны
            if date_from:
                tw_clicks_df = tw_clicks_df[tw_clicks_df["tw_date"] >= pd.Timestamp(date_from).date()]
            if date_to:
                tw_clicks_df = tw_clicks_df[tw_clicks_df["tw_date"] <= pd.Timestamp(date_to).date()]
            
            tw_grouped = tw_clicks_df.groupby(["tw_date", "tw_ad_campaign_name"]).agg({
                "tw_clicks": "sum",
                "tw_installations": "sum",
                "tw_registrations": "sum",
                "tw_deposits": "sum",
                "tw_revenue": "sum",
            }).reset_index()
            logger.info(f"Сгруппировано {len(tw_grouped)} уникальных комбинаций дата+кампания из TW")
    
    # Группируем Moloco по дате и кампании
    moloco_grouped = moloco_df.groupby(["m_date", "m_campaign"]).agg({
        "m_impression": "sum",
        "m_click": "sum",
        "m_install": "sum",
        "m_spend": "sum",
    }).reset_index()
    logger.info(f"Сгруппировано {len(moloco_grouped)} уникальных комбинаций дата+кампания из Moloco")
    
    # Объединяем по дате и кампании
    result = []
    matched_count = 0
    unmatched_count = 0
    
    for _, moloco_row in moloco_grouped.iterrows():
        m_date = moloco_row["m_date"]
        m_campaign = moloco_row["m_campaign"]
        
        # Ищем соответствующую запись в TW по дате и кампании
        # Используем точное совпадение с нормализацией пробелов
        tw_match = tw_grouped[
            (tw_grouped["tw_date"] == m_date) & 
            (tw_grouped["tw_ad_campaign_name"].str.strip() == m_campaign.strip())
        ]
        
        if not tw_match.empty:
            matched_count += 1
        else:
            unmatched_count += 1
            # Логируем первые несколько несовпадений для диагностики
            if unmatched_count <= 5:
                logger.debug(f"Не найдено соответствие для Moloco: дата={m_date}, кампания='{m_campaign}'")
                # Проверяем, есть ли похожие кампании в TW для этой даты
                same_date_tw = tw_grouped[tw_grouped["tw_date"] == m_date]
                if not same_date_tw.empty:
                    logger.debug(f"  Доступные кампании в TW для этой даты: {same_date_tw['tw_ad_campaign_name'].tolist()[:5]}")
        
        # Вычисляем значения для всех полей
        m_clicks = int(moloco_row["m_click"] or 0)
        m_installs = int(moloco_row["m_install"] or 0)
        m_spend_val = float(moloco_row["m_spend"] or 0)
        tw_click_val = int(tw_match["tw_clicks"].iloc[0] or 0) if not tw_match.empty else 0
        tw_install_val = int(tw_match["tw_installations"].iloc[0] or 0) if not tw_match.empty else 0
        tw_registration_val = int(tw_match["tw_registrations"].iloc[0] or 0) if not tw_match.empty else 0
        tw_deposits_val = int(tw_match["tw_deposits"].iloc[0] or 0) if not tw_match.empty else 0
        tw_revenue_val = float(tw_match["tw_revenue"].iloc[0] or 0) if not tw_match.empty else 0
        
        # Формируем строку результата в порядке selected_columns
        row = {}
        for col in selected_columns:
            if col == "m_campaign":
                row[col] = m_campaign
            elif col == "m_impression":
                row[col] = int(moloco_row["m_impression"] or 0)
            elif col == "m_click":
                row[col] = m_clicks
            elif col == "m_install":
                row[col] = m_installs
            elif col == "m_spend":
                row[col] = m_spend_val
            elif col == "tw_click":
                row[col] = tw_click_val
            elif col == "tw_install":
                row[col] = tw_install_val
            elif col == "tw_registration":
                row[col] = tw_registration_val
            elif col == "tw_deposits":
                row[col] = tw_deposits_val
            elif col == "tw_revenue":
                row[col] = tw_revenue_val
            elif col == "calc_click2inst":
                row[col] = (m_installs / m_clicks * 100) if m_clicks > 0 else 0.0
            elif col == "calc_reg2dep":
                row[col] = (tw_deposits_val / tw_registration_val * 100) if tw_registration_val > 0 else 0.0
            elif col == "calc_inst2dep":
                row[col] = (tw_deposits_val / tw_install_val * 100) if tw_install_val > 0 else 0.0
            elif col == "calc_profit":
                row[col] = tw_revenue_val - m_spend_val
            elif col == "calc_$inst":
                row[col] = (m_spend_val / m_installs) if m_installs > 0 else 0.0
            elif col == "calc_$reg":
                row[col] = (m_spend_val / tw_registration_val) if tw_registration_val > 0 else 0.0
            elif col == "calc_$dep":
                row[col] = (m_spend_val / tw_deposits_val) if tw_deposits_val > 0 else 0.0
            elif col == "calc_roi":
                profit_val = tw_revenue_val - m_spend_val
                row[col] = (profit_val / m_spend_val * 100) if m_spend_val > 0 else 0.0
        
        result.append(row)
    
    # Отмечаем, какие комбинации дата+кампания уже обработаны
    processed_keys = set(zip(moloco_grouped["m_date"], moloco_grouped["m_campaign"]))
    
    # Добавляем все строки из TW, которые не имеют соответствия в Moloco
    if not tw_grouped.empty:
        for _, tw_row in tw_grouped.iterrows():
            tw_date = tw_row["tw_date"]
            tw_campaign = tw_row["tw_ad_campaign_name"]
            
            # Проверяем, была ли эта комбинация дата+кампания уже обработана
            key = (tw_date, tw_campaign)
            if key in processed_keys:
                continue  # Уже обработано в основном цикле
            
            # Добавляем строку из TW без соответствия в Moloco
            if tw_campaign == "non-attribution":
                # Для non-attribution добавляем только если для этой даты нет данных в Moloco
                if tw_date not in moloco_grouped["m_date"].values:
                    tw_click_val = int(tw_row["tw_clicks"] or 0)
                    tw_install_val = int(tw_row["tw_installations"] or 0)
                    tw_registration_val = int(tw_row["tw_registrations"] or 0)
                    tw_deposits_val = int(tw_row["tw_deposits"] or 0)
                    tw_revenue_val = float(tw_row["tw_revenue"] or 0)
                    
                    # Формируем строку non-attribution в порядке selected_columns
                    row = {}
                    for col in selected_columns:
                        if col == "m_campaign":
                            row[col] = "non-attribution"
                        elif col == "m_impression":
                            row[col] = 0
                        elif col == "m_click":
                            row[col] = 0
                        elif col == "m_install":
                            row[col] = 0
                        elif col == "m_spend":
                            row[col] = 0
                        elif col == "tw_click":
                            row[col] = tw_click_val
                        elif col == "tw_install":
                            row[col] = tw_install_val
                        elif col == "tw_registration":
                            row[col] = tw_registration_val
                        elif col == "tw_deposits":
                            row[col] = tw_deposits_val
                        elif col == "tw_revenue":
                            row[col] = tw_revenue_val
                        elif col == "calc_click2inst":
                            row[col] = 0.0
                        elif col == "calc_reg2dep":
                            row[col] = (tw_deposits_val / tw_registration_val * 100) if tw_registration_val > 0 else 0.0
                        elif col == "calc_inst2dep":
                            row[col] = (tw_deposits_val / tw_install_val * 100) if tw_install_val > 0 else 0.0
                        elif col == "calc_profit":
                            row[col] = tw_revenue_val
                        elif col == "calc_$inst":
                            row[col] = 0.0
                        elif col == "calc_$reg":
                            row[col] = 0.0
                        elif col == "calc_$dep":
                            row[col] = 0.0
                        elif col == "calc_roi":
                            row[col] = 0.0
                    
                    result.append(row)
            else:
                # Для обычных кампаний добавляем строку с данными из TW, но без данных Moloco
                tw_click_val = int(tw_row["tw_clicks"] or 0)
                tw_install_val = int(tw_row["tw_installations"] or 0)
                tw_registration_val = int(tw_row["tw_registrations"] or 0)
                tw_deposits_val = int(tw_row["tw_deposits"] or 0)
                tw_revenue_val = float(tw_row["tw_revenue"] or 0)
                
                # Формируем строку TW без Moloco в порядке selected_columns
                row = {}
                for col in selected_columns:
                    if col == "m_campaign":
                        row[col] = tw_campaign
                    elif col == "m_impression":
                        row[col] = 0
                    elif col == "m_click":
                        row[col] = 0
                    elif col == "m_install":
                        row[col] = 0
                    elif col == "m_spend":
                        row[col] = 0
                    elif col == "tw_click":
                        row[col] = tw_click_val
                    elif col == "tw_install":
                        row[col] = tw_install_val
                    elif col == "tw_registration":
                        row[col] = tw_registration_val
                    elif col == "tw_deposits":
                        row[col] = tw_deposits_val
                    elif col == "tw_revenue":
                        row[col] = tw_revenue_val
                    elif col == "calc_click2inst":
                        row[col] = 0.0
                    elif col == "calc_reg2dep":
                        row[col] = (tw_deposits_val / tw_registration_val * 100) if tw_registration_val > 0 else 0.0
                    elif col == "calc_inst2dep":
                        row[col] = (tw_deposits_val / tw_install_val * 100) if tw_install_val > 0 else 0.0
                    elif col == "calc_profit":
                        row[col] = tw_revenue_val
                    elif col == "calc_$inst":
                        row[col] = 0.0
                    elif col == "calc_$reg":
                        row[col] = 0.0
                    elif col == "calc_$dep":
                        row[col] = 0.0
                    elif col == "calc_roi":
                        row[col] = 0.0
                
                result.append(row)
    
    # Фильтруем результат: оставляем ТОЛЬКО поля из selected_columns в правильном порядке
    filtered_result = []
    for row in result:
        filtered_row = {}
        # Добавляем поля ТОЛЬКО из selected_columns в правильном порядке
        for col in selected_columns:
            if col in row:
                filtered_row[col] = row[col]
            else:
                # Если поля нет, добавляем значение по умолчанию
                if col in float_fields:
                    filtered_row[col] = 0.0
                elif col in int_fields:
                    filtered_row[col] = 0
                else:
                    filtered_row[col] = ""
        filtered_result.append(filtered_row)
    
    # Финальная группировка по кампании для объединения одинаковых кампаний
    campaign_groups = {}
    campaign_name_map = {}  # Маппинг нормализованного названия на оригинальное
    for row in filtered_result:
        campaign = row.get("m_campaign", "")
        # Нормализуем название кампании (приводим к нижнему регистру и убираем пробелы)
        campaign_normalized = str(campaign).strip().lower() if campaign else ""
        
        # Сохраняем оригинальное название для первой встречи
        if campaign_normalized not in campaign_name_map:
            campaign_name_map[campaign_normalized] = campaign
        
        if campaign_normalized not in campaign_groups:
            campaign_groups[campaign_normalized] = {
                "m_campaign": campaign_name_map[campaign_normalized],  # Используем оригинальное название
                "m_impression": 0,
                "m_click": 0,
                "m_install": 0,
                "m_spend": 0.0,
                "tw_click": 0,
                "tw_install": 0,
                "tw_registration": 0,
                "tw_deposits": 0,
                "tw_revenue": 0.0,
            }
        
        # Суммируем числовые поля
        group = campaign_groups[campaign_normalized]
        group["m_impression"] += int(row.get("m_impression", 0) or 0)
        group["m_click"] += int(row.get("m_click", 0) or 0)
        group["m_install"] += int(row.get("m_install", 0) or 0)
        group["m_spend"] += float(row.get("m_spend", 0) or 0)
        group["tw_click"] += int(row.get("tw_click", 0) or 0)
        group["tw_install"] += int(row.get("tw_install", 0) or 0)
        group["tw_registration"] += int(row.get("tw_registration", 0) or 0)
        group["tw_deposits"] += int(row.get("tw_deposits", 0) or 0)
        group["tw_revenue"] += float(row.get("tw_revenue", 0) or 0)
    
    # Пересчитываем метрики и формируем финальный результат
    final_result = []
    for campaign_normalized, group in campaign_groups.items():
        campaign = group["m_campaign"]  # Используем оригинальное название
        m_clicks = group["m_click"]
        m_installs = group["m_install"]
        m_spend_val = group["m_spend"]
        tw_click_val = group["tw_click"]
        tw_install_val = group["tw_install"]
        tw_registration_val = group["tw_registration"]
        tw_deposits_val = group["tw_deposits"]
        tw_revenue_val = group["tw_revenue"]
        
        # Формируем строку результата в порядке selected_columns
        row = {}
        for col in selected_columns:
            if col == "m_campaign":
                row[col] = campaign
            elif col == "m_impression":
                row[col] = group["m_impression"]
            elif col == "m_click":
                row[col] = m_clicks
            elif col == "m_install":
                row[col] = m_installs
            elif col == "m_spend":
                row[col] = m_spend_val
            elif col == "tw_click":
                row[col] = tw_click_val
            elif col == "tw_install":
                row[col] = tw_install_val
            elif col == "tw_registration":
                row[col] = tw_registration_val
            elif col == "tw_deposits":
                row[col] = tw_deposits_val
            elif col == "tw_revenue":
                row[col] = tw_revenue_val
            elif col == "calc_click2inst":
                row[col] = (m_installs / m_clicks * 100) if m_clicks > 0 else 0.0
            elif col == "calc_reg2dep":
                row[col] = (tw_deposits_val / tw_registration_val * 100) if tw_registration_val > 0 else 0.0
            elif col == "calc_inst2dep":
                row[col] = (tw_deposits_val / tw_install_val * 100) if tw_install_val > 0 else 0.0
            elif col == "calc_profit":
                row[col] = tw_revenue_val - m_spend_val
            elif col == "calc_$inst":
                row[col] = (m_spend_val / m_installs) if m_installs > 0 else 0.0
            elif col == "calc_$reg":
                row[col] = (m_spend_val / tw_registration_val) if tw_registration_val > 0 else 0.0
            elif col == "calc_$dep":
                row[col] = (m_spend_val / tw_deposits_val) if tw_deposits_val > 0 else 0.0
            elif col == "calc_roi":
                profit_val = tw_revenue_val - m_spend_val
                row[col] = (profit_val / m_spend_val * 100) if m_spend_val > 0 else 0.0
            else:
                # Для остальных полей используем значение по умолчанию
                if col in float_fields:
                    row[col] = 0.0
                elif col in int_fields:
                    row[col] = 0
                else:
                    row[col] = ""
        
        final_result.append(row)
    
    logger.info(f"Объединено: совпадений={matched_count}, несовпадений={unmatched_count}, строк до группировки={len(filtered_result)}, строк после группировки={len(final_result)}")
    
    return final_result


def aggregate_moloco_creo(rows: list[CampaignStatistics], selected_columns: list[str]) -> list[dict]:
    """Агрегирует данные по креативам Moloco с привязкой revenue через кампанию."""
    dataset = [row_to_dict(r) for r in rows]
    
    # Определяем dimensions - по умолчанию m_creative, можно добавить stat_date
    dimensions = [col for col in selected_columns if col in {"m_creative", "stat_date"}]
    if not dimensions:
        dimensions = ["m_creative"]
        if "m_creative" not in selected_columns:
            selected_columns.append("m_creative")
    
    buckets: dict[tuple, dict] = {}
    sums: dict[tuple, dict] = {}
    campaign_revenue_map: dict[str, float] = {}  # Маппинг кампания -> revenue
    
    # Сначала собираем revenue по кампаниям из TW
    for row in dataset:
        campaign = row.get("m_campaign")
        if campaign:
            revenue = float(row.get("revenue", 0) or row.get("tw_income_usd", 0) or 0)
            if campaign not in campaign_revenue_map:
                campaign_revenue_map[campaign] = 0.0
            campaign_revenue_map[campaign] += revenue
    
    # Агрегируем по креативам
    for row in dataset:
        key = group_key(row, dimensions)
        bucket = buckets.setdefault(key, {dim: row.get(dim) for dim in dimensions})
        acc = sums.setdefault(key, {col: 0.0 for col in AGG_NUMERIC_COLUMNS})
        
        # Сохраняем кампанию для креатива
        if "m_campaign" not in bucket:
            bucket["m_campaign"] = row.get("m_campaign")
        
        # Агрегируем числовые поля
        for col in AGG_NUMERIC_COLUMNS:
            acc[col] = acc.get(col, 0.0) + float(row.get(col) or 0.0)
            if col in selected_columns:
                bucket[col] = acc[col]
    
    # Пересчитываем метрики и добавляем revenue из кампании
    for key, bucket in buckets.items():
        acc = sums[key]
        campaign = bucket.get("m_campaign")
        revenue = campaign_revenue_map.get(campaign, 0.0) if campaign else 0.0
        
        installs = acc.get("m_install", 0.0) or 0.0
        clicks = acc.get("m_click", 0.0) or 0.0
        regs = acc.get("m_registration", 0.0) or 0.0
        ftd = acc.get("m_first_purchase", 0.0) or 0.0
        spend = acc.get("m_spend", 0.0) or 0.0
        impressions = acc.get("m_impression", 0.0) or 0.0
        conversion = acc.get("m_conversion", 0.0) or 0.0
        
        bucket["m_install"] = installs
        bucket["m_click"] = clicks
        bucket["m_registration"] = regs
        bucket["m_first_purchase"] = ftd
        bucket["revenue"] = revenue
        bucket["m_spend"] = spend
        bucket["m_impression"] = impressions
        bucket["m_conversion"] = conversion
        
        # Расчетные метрики
        bucket["click2inst"] = (installs / clicks * 100) if clicks else 0.0
        bucket["inst2reg"] = (regs / installs * 100) if installs else 0.0
        bucket["inst2dep"] = (ftd / installs * 100) if installs else 0.0
        bucket["m_ctr"] = (clicks / impressions * 100) if impressions else acc.get("m_ctr", 0.0)
        bucket["m_cost_per_conversion"] = (spend / conversion) if conversion else 0.0
        bucket["m_cpi"] = (spend / installs) if installs else 0.0
        bucket["profit"] = revenue - spend
        bucket["roi_pct"] = ((revenue - spend) / spend * 100) if spend else 0.0
        bucket["tw_d14_purchases_sum"] = acc.get("tw_d14_purchases_sum", 0.0)
        bucket["m_purchase"] = acc.get("m_purchase", 0.0)
    
    return list(buckets.values())


def aggregate_tw_offer(rows: list[CampaignStatistics], selected_columns: list[str]) -> list[dict]:
    """Агрегирует данные по офферам TW с привязкой spend из Moloco."""
    dataset = [row_to_dict(r) for r in rows]
    
    # Определяем dimensions - по умолчанию tw_link_title, можно добавить stat_date
    dimensions = [col for col in selected_columns if col in {"tw_link_title", "stat_date", "tw_ad_campaign_name", "tw_geo_country_code"}]
    if not dimensions:
        dimensions = ["tw_link_title"]
        if "tw_link_title" not in selected_columns:
            selected_columns.append("tw_link_title")
    
    buckets: dict[tuple, dict] = {}
    sums: dict[tuple, dict] = {}
    
    for row in dataset:
        key = group_key(row, dimensions)
        bucket = buckets.setdefault(key, {dim: row.get(dim) for dim in dimensions})
        acc = sums.setdefault(key, {col: 0.0 for col in AGG_NUMERIC_COLUMNS})
        
        # Агрегируем числовые поля
        for col in AGG_NUMERIC_COLUMNS:
            acc[col] = acc.get(col, 0.0) + float(row.get(col) or 0.0)
            if col in selected_columns:
                bucket[col] = acc[col]
    
    # Пересчитываем метрики
    for key, bucket in buckets.items():
        acc = sums[key]
        installs = acc.get("tw_installations", 0.0) or 0.0
        regs = acc.get("tw_registrations", 0.0) or 0.0
        ftd = acc.get("tw_first_purchases", 0.0) or 0.0
        revenue = acc.get("tw_income_usd", 0.0) or 0.0
        spend = acc.get("calculated_spend", 0.0) or acc.get("m_spend", 0.0) or 0.0
        
        bucket["tw_installations"] = installs
        bucket["tw_registrations"] = regs
        bucket["tw_first_purchases"] = ftd
        bucket["revenue"] = revenue
        bucket["tw_income_usd"] = revenue
        bucket["calculated_spend"] = spend
        bucket["m_spend"] = spend
        
        # Расчетные метрики (всегда добавляем в bucket)
        bucket["tw_install2reg"] = (regs / installs * 100) if installs else 0.0
        bucket["inst2reg"] = (regs / installs * 100) if installs else 0.0
        bucket["tw_reg2dep"] = (ftd / regs * 100) if regs else 0.0
        bucket["inst2dep"] = (ftd / installs * 100) if installs else 0.0
        bucket["calculated_cpi"] = (spend / installs) if installs else 0.0
        bucket["m_cpi"] = (spend / installs) if installs else 0.0
        bucket["profit"] = revenue - spend
        bucket["roi_pct"] = ((revenue - spend) / spend * 100) if spend else 0.0
        bucket["tw_d14_aov"] = acc.get("tw_d14_aov", 0.0)
        bucket["tw_d14_purchases_sum"] = acc.get("tw_d14_purchases_sum", 0.0)
    
    return list(buckets.values())


def prepare_dataset(session: Session, request: Request) -> tuple[list[dict], list[str]]:
    rows = session.query(CampaignStatistics).order_by(CampaignStatistics.tw_installations.desc(), CampaignStatistics.tw_income_usd.desc()).all()
    filtered = apply_filters(rows, request.query_params)
    columns = select_columns(request.query_params.get("columns"))
    aggregated, _ = aggregate_dataset(filtered, columns)
    return aggregated, columns


# --- FastAPI setup и эндпоинты ---

def create_app() -> FastAPI:
    ensure_schema()
    app = FastAPI(title="TW Buy Stat", version="0.1.0")
    templates = Jinja2Templates(directory="templates")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/upload/tw", summary="Загрузка файла TrafficWave")
    async def upload_tw(file: UploadFile = File(...), session: Session = Depends(get_session)) -> dict:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Поддерживаются только CSV-файлы")
        path = Path("data/raw/tw")
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / file.filename
        file_path.write_bytes(await file.read())
        load_tw_file(str(file_path), session)
        rebuild_statistics(session)
        return {"status": "ok"}

    @app.post("/upload/moloco", summary="Загрузка файла Moloco")
    async def upload_moloco(file: UploadFile = File(...), session: Session = Depends(get_session)) -> dict:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Поддерживаются только CSV-файлы")
        path = Path("data/raw/moloco")
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / file.filename
        file_path.write_bytes(await file.read())
        load_moloco_file(str(file_path), session)
        rebuild_statistics(session)
        return {"status": "ok"}

    @app.post("/upload/tw-clicks", summary="Загрузка файла TW data_clicks.csv")
    async def upload_tw_clicks(file: UploadFile = File(...), session: Session = Depends(get_session)) -> dict:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Поддерживаются только CSV-файлы")
        path = Path("data/raw/tw")
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / file.filename
        file_path.write_bytes(await file.read())
        load_tw_clicks_file(str(file_path), session)
        return {"status": "ok"}

    @app.get("/stats", summary="Получение суммарной статистики")
    def get_stats(request: Request, session: Session = Depends(get_session)) -> dict:
        # Проверяем, для какой вкладки запрашиваются статистики
        tab = request.query_params.get("tab") or request.query_params.get("currentTab")
        
        # Для вкладки moloco-camp используем данные из aggregate_moloco_camp
        if tab == "moloco-camp":
            columns = DEFAULT_COLUMNS_MOLOCO_CAMP.copy()
            
            # Получаем параметры фильтрации по датам
            date_from_str = request.query_params.get("date_from")
            date_to_str = request.query_params.get("date_to")
            date_from = None
            date_to = None
            if date_from_str:
                try:
                    date_from = date.fromisoformat(date_from_str)
                except ValueError:
                    pass
            if date_to_str:
                try:
                    date_to = date.fromisoformat(date_to_str)
                except ValueError:
                    pass
            
            aggregated = aggregate_moloco_camp(session, columns, date_from=date_from, date_to=date_to)
            
            hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
            if hide_zero:
                filtered_rows = [r for r in aggregated if any((r.get(col) or 0) != 0 for col in columns if col in NUMERIC_FIELDS)]
            else:
                filtered_rows = aggregated
            
            # Суммируем данные из moloco-camp
            total_revenue = sum(float(row.get("tw_revenue", 0) or 0) for row in filtered_rows)
            total_spend = sum(float(row.get("m_spend", 0) or 0) for row in filtered_rows)
            total_profit = total_revenue - total_spend
            total_roi = (total_profit / total_spend * 100) if total_spend else 0
            
            return {
                "total_revenue": round(total_revenue, 2),
                "total_spend": round(total_spend, 2),
                "total_profit": round(total_profit, 2),
                "total_roi": round(total_roi, 2),
                "records": len(filtered_rows),
            }
        
        # Для остальных вкладок используем CampaignStatistics
        rows = session.query(CampaignStatistics).order_by(CampaignStatistics.tw_installations.desc(), CampaignStatistics.tw_income_usd.desc()).all()
        filtered = apply_filters(rows, request.query_params)
        columns = select_columns(request.query_params.get("columns"))
        aggregated, _ = aggregate_dataset(filtered, columns)
        hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
        if hide_zero:
            aggregated = filter_zero_rows(aggregated, columns)
        total_revenue = sum(row.get("tw_income_usd", 0.0) for row in aggregated)
        total_spend = sum(row.get("calculated_spend", 0.0) for row in aggregated)
        total_profit = total_revenue - total_spend
        total_roi = (total_profit / total_spend * 100) if total_spend else 0
        return {
            "total_revenue": round(total_revenue, 2),
            "total_spend": round(total_spend, 2),
            "total_profit": round(total_profit, 2),
            "total_roi": round(total_roi, 2),
            "records": len(aggregated),
        }

    @app.get("/table", summary="Получение витрины")
    def get_table(request: Request, session: Session = Depends(get_session)) -> dict:
        rows = session.query(CampaignStatistics).order_by(CampaignStatistics.tw_installations.desc(), CampaignStatistics.tw_income_usd.desc()).all()
        filtered = apply_filters(rows, request.query_params)
        columns = select_columns(request.query_params.get("columns"))
        aggregated, dimensions = aggregate_dataset(filtered, columns)
        hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
        if hide_zero:
            filtered_rows = filter_zero_rows(aggregated, columns)
        else:
            filtered_rows = aggregated
        data = [format_row(row, columns) for row in filtered_rows]

        zero_columns = []
        if not hide_zero:
            zero_columns = [col for col in columns if col in NUMERIC_FIELDS]

        limit_param = request.query_params.get("limit")
        limit = int(limit_param) if limit_param and limit_param.isdigit() else None
        if limit:
            data = data[:limit]
        return {
            "items": data,
            "count": len(data),
            "total": len(filtered_rows),
            "group_dimensions": dimensions,
            "zero_columns": zero_columns,
        }

    # Иерархический эндпоинт отложен в конец MVP
    # @app.get("/table/hierarchical", summary="Получение иерархической витрины (кампании → офферы)")
    def get_hierarchical_table(request: Request, session: Session = Depends(get_session)) -> dict:
        rows = session.query(CampaignStatistics).order_by(CampaignStatistics.tw_installations.desc(), CampaignStatistics.tw_income_usd.desc()).all()
        filtered = apply_filters(rows, request.query_params)
        columns = select_columns(request.query_params.get("columns"))
        hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
        
        def prepare_offer_row(data: dict, campaign_cpi: float) -> dict:
            prepared = data.copy()
            installs = float(prepared.get("tw_installations", 0) or 0)
            income = float(prepared.get("tw_income_usd", 0.0) or 0.0)
            raw_m_spend = float(prepared.get("m_spend", 0) or 0.0)
            raw_calc_spend = float(prepared.get("calculated_spend", 0) or 0.0)
            spend = raw_m_spend if raw_m_spend > 0 else raw_calc_spend
            cpi_value = spend / installs if installs else float(campaign_cpi or 0.0)
            for field in [
                "m_impression",
                "m_click",
                "m_install",
                "m_spend",
            ]:
                prepared.pop(field, None)
            prepared["calculated_spend"] = spend
            prepared["calculated_cpi"] = cpi_value
            prepared["m_cpi"] = cpi_value
            profit = income - spend
            prepared["profit"] = profit
            prepared["roi_pct"] = (profit / spend * 100) if spend else 0.0
            return prepared

        # Группируем по кампании и офферу
        campaign_map = {}
        for row in filtered:
            row_dict = row_to_dict(row)
            campaign = row_dict.get("tw_ad_campaign_name") or "Unknown"
            offer = row_dict.get("tw_link_title") or "Unknown"
            
            if campaign not in campaign_map:
                campaign_map[campaign] = {"offers": {}, "rows": []}
            
            if offer not in campaign_map[campaign]["offers"]:
                campaign_map[campaign]["offers"][offer] = []
            
            campaign_map[campaign]["offers"][offer].append(row_dict)
            campaign_map[campaign]["rows"].append(row_dict)
        
        # Агрегируем данные для каждой кампании и оффера
        hierarchical_data = []
        for campaign_name, campaign_data in campaign_map.items():
            # Агрегируем для кампании в целом
            campaign_agg = aggregate_rows(campaign_data["rows"])
            campaign_record = {
                "type": "campaign",
                "campaign": campaign_name,
                "data": format_row(campaign_agg[0], columns) if campaign_agg else {},
                "offers": []
            }
            campaign_cpi_value = 0.0
            if campaign_record["data"]:
                campaign_cpi_value = float(campaign_record["data"].get("calculated_cpi") or 0.0)
            
            # Агрегируем для каждого оффера
            for offer_name, offer_rows in campaign_data["offers"].items():
                offer_agg = aggregate_rows(offer_rows)
                if offer_agg:
                    offer_record = {
                        "type": "offer",
                        "offer": offer_name,
                        "data": prepare_offer_row(format_row(offer_agg[0], columns), campaign_cpi_value),
                    }
                    if not hide_zero or is_non_zero(offer_record["data"], columns):
                        campaign_record["offers"].append(offer_record)
            
            if not hide_zero or is_non_zero(campaign_record["data"], columns):
                hierarchical_data.append(campaign_record)
        
        limit_param = request.query_params.get("limit")
        limit = int(limit_param) if limit_param and limit_param.isdigit() else None
        if limit:
            hierarchical_data = hierarchical_data[:limit]
        
        return {
            "items": hierarchical_data,
            "count": len(hierarchical_data),
        }
    
    def aggregate_rows(rows: list[dict]) -> list[dict]:
        """Агрегирует список строк в одну"""
        if not rows:
            return []
        
        agg = {}
        for row in rows:
            for key, value in row.items():
                if key in AGG_NUMERIC_COLUMNS:
                    agg[key] = agg.get(key, 0.0) + float(value or 0)
                else:
                    agg[key] = value
        
        # Пересчет процентных метрик
        installs = agg.get("tw_installations", 0.0)
        regs = agg.get("tw_registrations", 0.0)
        ftd = agg.get("tw_first_purchases", 0.0)
        income = agg.get("tw_income_usd", 0.0)
        spend = agg.get("calculated_spend", 0.0)
        purchases_sum = agg.get("tw_purchases_sum", 0.0)
        
        agg["tw_install2reg"] = (regs / installs * 100) if installs else 0.0
        agg["tw_reg2dep"] = (ftd / regs * 100) if regs else 0.0
        agg["tw_average_bill"] = purchases_sum / ftd if ftd else 0.0
        agg["profit"] = income - spend
        agg["roi_pct"] = ((income - spend) / spend * 100) if spend else 0.0
        agg["calculated_cpi"] = (spend / installs) if installs else 0.0
        m_spend = agg.get("m_spend", 0.0)
        agg["m_cpi"] = (m_spend / installs) if installs else 0.0
        
        return [agg]
    
    def is_non_zero(data: dict, columns: list[str]) -> bool:
        """Проверяет, есть ли ненулевые значения в числовых колонках"""
        numeric_cols = [col for col in columns if col in NUMERIC_FIELDS]
        if not numeric_cols:
            numeric_cols = list(NUMERIC_FIELDS)
        return any((data.get(col) or 0) != 0 for col in numeric_cols)

    @app.get("/table/moloco-camp", summary="Получение статистики по кампаниям Moloco")
    def get_moloco_camp_table(request: Request, session: Session = Depends(get_session)) -> dict:
        # Всегда используем только столбцы из ТЗ, игнорируя параметр columns
        columns = DEFAULT_COLUMNS_MOLOCO_CAMP.copy()
        
        # Получаем параметры фильтрации по датам
        date_from_str = request.query_params.get("date_from")
        date_to_str = request.query_params.get("date_to")
        date_from = None
        date_to = None
        if date_from_str:
            try:
                date_from = date.fromisoformat(date_from_str)
            except ValueError:
                pass
        if date_to_str:
            try:
                date_to = date.fromisoformat(date_to_str)
            except ValueError:
                pass
        
        # Передаем параметры фильтрации по датам в функцию агрегации
        aggregated = aggregate_moloco_camp(session, columns, date_from=date_from, date_to=date_to)
        
        hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
        if hide_zero:
            filtered_rows = [r for r in aggregated if any((r.get(col) or 0) != 0 for col in columns if col in NUMERIC_FIELDS)]
        else:
            filtered_rows = aggregated
        
        # Форматируем строки, сохраняя порядок столбцов из columns
        # ВАЖНО: возвращаем ТОЛЬКО столбцы из columns, никаких других полей
        data = []
        for row in filtered_rows:
            formatted = {}
            # Проходим ТОЛЬКО по столбцам из columns в правильном порядке (соответствует ТЗ)
            # Игнорируем любые другие поля, которые могут быть в row
            for col in columns:
                value = row.get(col)  # Используем get чтобы не было KeyError
                if value is not None:
                    if col in FLOAT_FIELDS:
                        formatted[col] = float(value or 0)
                    elif col in INT_FIELDS:
                        formatted[col] = int(float(value or 0))
                    else:
                        formatted[col] = value
                else:
                    # Если поля нет, используем значение по умолчанию
                    if col in FLOAT_FIELDS:
                        formatted[col] = 0.0
                    elif col in INT_FIELDS:
                        formatted[col] = 0
                    else:
                        formatted[col] = ""
            # Убеждаемся, что в formatted только поля из columns в правильном порядке
            # Создаем новый словарь в порядке columns для гарантии сохранения порядка
            final_formatted = {}
            for col in columns:
                if col in formatted:
                    final_formatted[col] = formatted[col]
                else:
                    # Если поля нет, используем значение по умолчанию
                    if col in FLOAT_FIELDS:
                        final_formatted[col] = 0.0
                    elif col in INT_FIELDS:
                        final_formatted[col] = 0
                    else:
                        final_formatted[col] = ""
            data.append(final_formatted)
        
        limit_param = request.query_params.get("limit")
        limit = int(limit_param) if limit_param and limit_param.isdigit() else None
        
        # Рассчитываем строку Total для ВСЕХ отфильтрованных данных, а не только текущей страницы
        total_row = None
        if filtered_rows:
            total = {}
            # Суммируем числовые колонки из ВСЕХ отфильтрованных строк
            for col in columns:
                if col in FLOAT_FIELDS or col in INT_FIELDS:
                    total[col] = sum(float(row.get(col) or 0) for row in filtered_rows)
                else:
                    total[col] = "Total"
            
            # Пересчитываем метрики на основе сумм
            m_clicks_total = total.get("m_click", 0.0) or 0.0
            m_installs_total = total.get("m_install", 0.0) or 0.0
            m_spend_total = total.get("m_spend", 0.0) or 0.0
            tw_clicks_total = total.get("tw_click", 0.0) or 0.0
            tw_installs_total = total.get("tw_install", 0.0) or 0.0
            tw_regs_total = total.get("tw_registration", 0.0) or 0.0
            tw_deps_total = total.get("tw_deposits", 0.0) or 0.0
            tw_revenue_total = total.get("tw_revenue", 0.0) or 0.0
            
            # Пересчет расчетных метрик
            if "calc_click2inst" in columns:
                total["calc_click2inst"] = (m_installs_total / m_clicks_total * 100) if m_clicks_total > 0 else 0.0
            if "calc_reg2dep" in columns:
                total["calc_reg2dep"] = (tw_deps_total / tw_regs_total * 100) if tw_regs_total > 0 else 0.0
            if "calc_inst2dep" in columns:
                total["calc_inst2dep"] = (tw_deps_total / tw_installs_total * 100) if tw_installs_total > 0 else 0.0
            if "calc_profit" in columns:
                total["calc_profit"] = tw_revenue_total - m_spend_total
            if "calc_$inst" in columns:
                total["calc_$inst"] = (m_spend_total / m_installs_total) if m_installs_total > 0 else 0.0
            if "calc_$reg" in columns:
                total["calc_$reg"] = (m_spend_total / tw_regs_total) if tw_regs_total > 0 else 0.0
            if "calc_$dep" in columns:
                total["calc_$dep"] = (m_spend_total / tw_deps_total) if tw_deps_total > 0 else 0.0
            if "calc_roi" in columns:
                profit_total = tw_revenue_total - m_spend_total
                total["calc_roi"] = (profit_total / m_spend_total * 100) if m_spend_total > 0 else 0.0
            
            # Форматируем total_row
            total_formatted = {}
            for col in columns:
                value = total.get(col, "")
                if col in FLOAT_FIELDS:
                    total_formatted[col] = float(value or 0)
                elif col in INT_FIELDS:
                    total_formatted[col] = int(float(value or 0))
                else:
                    total_formatted[col] = value
            total_row = total_formatted
        
        # Применяем limit только для отображения, после расчета total
        if limit:
            data = data[:limit]
        
        return {
            "items": data,
            "total_row": total_row,
            "count": len(data),
            "total": len(filtered_rows),
        }

    @app.get("/table/moloco-creo", summary="Получение статистики по креативам Moloco")
    def get_moloco_creo_table(request: Request, session: Session = Depends(get_session)) -> dict:
        rows = session.query(CampaignStatistics).order_by(CampaignStatistics.tw_installations.desc(), CampaignStatistics.tw_income_usd.desc()).all()
        filtered = apply_filters(rows, request.query_params)
        columns = select_columns_for_tab(request.query_params.get("columns"), "moloco-creo")
        aggregated = aggregate_moloco_creo(filtered, columns)
        hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
        if hide_zero:
            filtered_rows = filter_zero_rows(aggregated, columns)
        else:
            filtered_rows = aggregated
        data = [format_row(row, columns) for row in filtered_rows]
        
        # Добавляем строку Total
        total_row = calculate_total_row(filtered_rows, columns)
        if total_row:
            total_formatted = format_row(total_row, columns)
        
        limit_param = request.query_params.get("limit")
        limit = int(limit_param) if limit_param and limit_param.isdigit() else None
        if limit:
            data = data[:limit]
        
        return {
            "items": data,
            "total_row": total_formatted if total_row else None,
            "count": len(data),
            "total": len(filtered_rows),
        }

    @app.get("/table/tw-offer", summary="Получение статистики по офферам TW")
    def get_tw_offer_table(request: Request, session: Session = Depends(get_session)) -> dict:
        rows = session.query(CampaignStatistics).order_by(CampaignStatistics.tw_installations.desc(), CampaignStatistics.tw_income_usd.desc()).all()
        filtered = apply_filters(rows, request.query_params)
        columns = select_columns_for_tab(request.query_params.get("columns"), "tw-offer")
        aggregated = aggregate_tw_offer(filtered, columns)
        hide_zero = request.query_params.get("hide_zero_rows", "true").lower() != "false"
        if hide_zero:
            filtered_rows = filter_zero_rows(aggregated, columns)
        else:
            filtered_rows = aggregated
        data = [format_row(row, columns) for row in filtered_rows]
        
        # Добавляем строку Total
        total_row = calculate_total_row(filtered_rows, columns)
        if total_row:
            total_formatted = format_row(total_row, columns)
        
        limit_param = request.query_params.get("limit")
        limit = int(limit_param) if limit_param and limit_param.isdigit() else None
        if limit:
            data = data[:limit]
        
        return {
            "items": data,
            "total_row": total_formatted if total_row else None,
            "count": len(data),
            "total": len(filtered_rows),
        }

    @app.get("/loads", summary="История загрузок")
    def get_loads(session: Session = Depends(get_session)) -> list[dict]:
        records = session.query(DataLoad).order_by(DataLoad.started_at.desc()).all()
        return [
            {
                "id": record.id,
                "source": record.source,
                "file_name": record.file_name,
                "status": record.status,
                "records_total": record.records_total,
                "records_valid": record.records_valid,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
            }
            for record in records
        ]

    return app


app = create_app()


