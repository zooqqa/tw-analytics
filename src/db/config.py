"""Конфигурация подключения к базе данных."""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote_plus


DEFAULT_DB_URL = "sqlite:///dev.db"


@lru_cache(maxsize=1)
def get_database_url() -> str:
    """Возвращает строку подключения, учитывая переменные окружения."""

    value = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    if value.startswith("postgresql://"):
        prefix, rest = value.split("://", 1)
        if "@" in rest:
            creds, host_part = rest.split("@", 1)
            if ":" in creds:
                user, pwd = creds.split(":", 1)
                if any(ord(ch) > 127 for ch in pwd):
                    pwd = quote_plus(pwd)
                return f"{prefix}://{user}:{pwd}@{host_part}"
    return value


@lru_cache(maxsize=1)
def is_sqlite() -> bool:
    """Проверяет, используется ли SQLite."""
    return get_database_url().startswith("sqlite")


