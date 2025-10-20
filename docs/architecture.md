# Архитектура проекта сквозной аналитики

## 1. Общая схема
- Источник данных: CSV-файлы `tw.csv` (TrafficWave) и `moloco.csv` (Moloco).
- Загрузка: CLI `python -m src.main --tw ... --moloco ...` вызывает модуль `src/ingestion`.
- Хранилище: PostgreSQL (`tw_analytics`, пользователь `tw_user/qwas1234`).
- Витрина: таблица `campaign_statistics`, собирается сервисом `src/services/merge_service.py`.
- API: FastAPI в `src/api/main.py`, отдаёт витрину и агрегаты.

## 2. Структура проекта
```
src/
  db/            # engine, ORM модели и создание схемы
  ingestion/     # загрузчики CSV (TW, Moloco)
  services/      # пересборка витрины campaign_statistics
  api/           # FastAPI приложение и эндпоинты
  main.py        # CLI для ручной загрузки файлов
```
Дополнительные данные находятся в `docs/` и `data/raw/`.

## 3. Поток данных
1. `load_tw_file` читает `tw.csv`, фильтрует пустые ссылки, приводит типы и очищает таблицу `tw_events` перед вставкой.
2. `load_moloco_file` читает `moloco.csv`, агрегирует записи по кампании и дате, обновляет `moloco_events`.
3. `rebuild_statistics` объединяет `tw_events` и `moloco_events` по `(date, campaign)`, высчитывает `calculated_spend = CPI * installs`, прибыль и ROI, и перезаписывает `campaign_statistics`.
4. Сумма `tw_income_usd` в витрине совпадает с суммой дохода из TW-файла; расход допускает расхождение до 5% из-за расчёта `CPI * installs`.

## 4. База данных
- Таблицы: `data_loads`, `tw_events`, `moloco_events`, `campaign_statistics`.
- Схема создаётся автоматически при запуске API (`ensure_schema`).
- Пользователь `tw_user` имеет права на базу `tw_analytics`.

## 5. REST API
- `POST /upload/tw` — загрузка нового TW-файла, пересборка витрины.
- `POST /upload/moloco` — загрузка Moloco-файла, пересборка витрины.
- `GET /stats` — агрегированные показатели (доход, расход, ROI, количество записей) с фильтрами по кампании/дате.
- `GET /table` — постраничная выдача строк витрины (`limit`, `offset`).
- `GET /loads` — история загрузок из таблицы `data_loads`.

## 6. Запуск и тестирование
- Загрузка CSV: `python -m src.main --tw tw.csv --moloco moloco.csv`.
- Запуск API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` (при необходимости установить `DATABASE_URL`).
- Проверка: запросы `GET /stats` и `GET /table?limit=5` должны возвращать данные из PostgreSQL.

