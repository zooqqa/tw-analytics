"""Описание ORM-моделей проекта."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Integer, Numeric, String, Text, UniqueConstraint

from .session import Base


def uuid_str() -> str:
    """Генерирует строковый UUID для ключей загрузок."""

    return str(uuid.uuid4())


class DataLoad(Base):
    __tablename__ = "data_loads"

    id = Column(String(36), primary_key=True, default=uuid_str)
    source = Column(String(32), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime(timezone=True))
    status = Column(String(32), nullable=False, default="processing")
    records_total = Column(Integer, default=0)
    records_valid = Column(Integer, default=0)
    error_log = Column(Text)


class TwEvent(Base):
    __tablename__ = "tw_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    load_id = Column(String(36), nullable=False, index=True)
    source_file = Column(String(255), nullable=False)
    tw_date = Column(Date, nullable=False)
    tw_link_title = Column(String(500), nullable=False)
    tw_ad_campaign_name = Column(String(500), nullable=False)
    tw_geo_country_code = Column(String(10))
    tw_carrot_id = Column(String(128))
    tw_first_purchases = Column(Integer, default=0)
    tw_registrations = Column(Integer, default=0)
    tw_installations = Column(Integer, default=0)
    tw_install2reg = Column(Float, default=0.0)
    tw_reg2dep = Column(Float, default=0.0)
    tw_income_usd = Column(Numeric(14, 4), default=0)
    tw_average_bill = Column(Numeric(12, 2), default=0)
    tw_purchases_sum = Column(Numeric(14, 4), default=0)
    tw_d14_aov = Column(Numeric(12, 2), default=0)
    tw_d14_purchases_sum = Column(Numeric(14, 4), default=0)
    tw_epc = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = ()


class MolocoEvent(Base):
    __tablename__ = "moloco_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    load_id = Column(String(36), nullable=False, index=True)
    source_file = Column(String(255), nullable=False)
    m_date = Column(Date, nullable=False)
    m_campaign = Column(String(500), nullable=False)
    m_campaign_id = Column(String(128))
    m_countries = Column(String(128))
    m_app = Column(String(500))
    m_app_id = Column(String(128))
    m_creative = Column(String(500))
    m_creative_id = Column(String(128))
    m_impression = Column(Integer, default=0)
    m_click = Column(Integer, default=0)
    m_install = Column(Integer, default=0)
    m_conversion = Column(Integer, default=0)
    m_spend = Column(Numeric(14, 4), default=0)
    m_ctr = Column(Float, default=0.0)
    m_cpi = Column(Float, default=0.0)
    m_cpa = Column(Float, default=0.0)
    m_cost_per_conversion = Column(Float, default=0.0)
    m_currency = Column(String(10), default="USD")
    m_registration = Column(Integer, default=0)
    m_first_purchase = Column(Integer, default=0)
    m_purchase = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = ()


class CampaignStatistics(Base):
    __tablename__ = "campaign_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_date = Column(Date)
    tw_ad_campaign_name = Column(String(500))
    tw_link_title = Column(String(500))
    tw_carrot_id = Column(String(128))
    tw_geo_country_code = Column(String(10))
    tw_first_purchases = Column(Integer, default=0)
    tw_registrations = Column(Integer, default=0)
    tw_installations = Column(Integer, default=0)
    tw_install2reg = Column(Float, default=0.0)
    tw_reg2dep = Column(Float, default=0.0)
    tw_income_usd = Column(Numeric(14, 4), default=0)
    tw_average_bill = Column(Numeric(12, 2), default=0)
    tw_purchases_sum = Column(Numeric(14, 4), default=0)
    tw_d14_aov = Column(Numeric(12, 2), default=0)
    tw_d14_purchases_sum = Column(Numeric(14, 4), default=0)
    tw_epc = Column(Float, default=0.0)
    m_campaign = Column(String(500))
    m_campaign_id = Column(String(128))
    m_app = Column(String(500))
    m_app_id = Column(String(128))
    m_creative = Column(String(500))
    m_creative_id = Column(String(128))
    m_impression = Column(Integer, default=0)
    m_click = Column(Integer, default=0)
    m_install = Column(Integer, default=0)
    m_conversion = Column(Integer, default=0)
    m_spend = Column(Numeric(14, 4), default=0)
    m_ctr = Column(Float, default=0.0)
    m_cpi = Column(Float, default=0.0)
    m_cpa = Column(Float, default=0.0)
    m_registration = Column(Integer, default=0)
    m_first_purchase = Column(Integer, default=0)
    m_purchase = Column(Integer, default=0)
    m_currency = Column(String(10))
    calculated_spend = Column(Numeric(14, 4), default=0)
    calculated_cpi = Column(Float, default=0.0)
    profit = Column(Numeric(14, 4), default=0)
    roi_pct = Column(Float, default=0.0)
    kpi_pct = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = ()


