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

from src.db.models import CampaignStatistics, DataLoad
from src.db.session import get_session
from src.db.setup import ensure_schema
from src.ingestion import load_moloco_file, load_tw_file
from src.services import rebuild_statistics

# Мета информация о колонках
METADATA = [
    {"key": "stat_date", "dimension": True},
    {"key": "tw_ad_campaign_name", "dimension": True},
    {"key": "tw_link_title", "dimension": True},
    {"key": "tw_carrot_id", "dimension": True},
    {"key": "tw_geo_country_code", "dimension": True},
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
    {"key": "calculated_spend", "dimension": False},
    {"key": "calculated_cpi", "dimension": False},
    {"key": "profit", "dimension": False},
    {"key": "roi_pct", "dimension": False},
    {"key": "kpi_pct", "dimension": False},
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
    "calculated_spend",
    "calculated_cpi",
    "profit",
    "roi_pct",
    "kpi_pct",
}
INT_FIELDS = {
    "tw_installations",
    "tw_registrations",
    "tw_first_purchases",
    "m_impression",
    "m_click",
    "m_install",
}
STRING_FIELDS = {
    "stat_date",
    "tw_ad_campaign_name",
    "tw_link_title",
    "tw_carrot_id",
    "tw_geo_country_code",
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
    "calculated_spend",
    "calculated_cpi",
    "profit",
    "roi_pct",
    "kpi_pct",
}


def select_columns(raw: Optional[str]) -> list[str]:
    if raw and raw.strip():
        columns = [item.strip() for item in raw.split(",") if item.strip() in COLUMN_SET]
        if columns:
            return columns
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
        "calculated_spend": decimal_to_float(row.calculated_spend),
        "calculated_cpi": float(row.calculated_cpi or 0),
        "profit": decimal_to_float(row.profit),
        "roi_pct": float(row.roi_pct or 0),
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
        if column in FLOAT_FIELDS:
            formatted[column] = float(value or 0)
        elif column in INT_FIELDS:
            formatted[column] = int(float(value or 0))
        else:
            formatted[column] = value
    return formatted


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

    @app.get("/stats", summary="Получение суммарной статистики")
    def get_stats(request: Request, session: Session = Depends(get_session)) -> dict:
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

    @app.get("/table/hierarchical", summary="Получение иерархической витрины (кампании → офферы)")
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


