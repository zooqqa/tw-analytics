"""Инструменты для создания сессий SQLAlchemy."""

from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_database_url


DATABASE_URL = get_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_session() -> Iterator:
    """Поставляет сессию для использования в FastAPI/CLI."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


