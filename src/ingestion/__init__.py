"""Модуль загрузки исходных CSV файлов."""

from .tw_loader import load_tw_file
from .moloco_loader import load_moloco_file

__all__ = [
    "load_tw_file",
    "load_moloco_file",
]


