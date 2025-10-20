"""Утилиты для первичной инициализации базы."""

from __future__ import annotations

from sqlalchemy import inspect

from .models import Base
from .session import engine


def ensure_schema() -> None:
    """Создает таблицы, если их еще нет."""

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if {"data_loads", "tw_events", "moloco_events", "campaign_statistics"}.issubset(set(tables)):
        return
    Base.metadata.create_all(bind=engine)


