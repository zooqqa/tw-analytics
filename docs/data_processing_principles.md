# Принципы обработки CSV для формирования базы данных

Документ описывает текущую логику MVP для загрузки и объединения файлов `tw.csv` и `moloco.csv` в БД PostgreSQL.

## 1. Источники и форматы

### 1.1 TW (TrafficWave)
- Файл загружается вручную в `data/raw/tw/<filename>.csv`.
- Разделитель `;`, кодировка UTF-8.
- Обязательные колонки: `date`, `link_title`, `ad_campaign_name`, `geo_country_code`, `carrot_id`, `first_purchases`, `registrations`, `installations`, `install2reg`, `reg2dep`, `income_usd`.

### 1.2 Moloco
- Файл загружается в `data/raw/moloco/<filename>.csv`.
- Разделитель `,`, кодировка UTF-8.
- Обязательные колонки: `Date`, `Campaign`, `Impression`, `Click`, `Install`, `Spend`, `CPI` (прочие поля игнорируются).

## 2. Загрузка в базу (staging)

### 2.1 Общие правила
- Для каждого файла создаётся запись в `data_loads` с рассчитанным SHA-256.
- Перед вставкой полностью очищаются таблицы соответствующего источника (`tw_events`, `moloco_events`).
- Числовые значения очищаются от запятых и приводятся к типам `int`/`float`.
- Даты приводятся к типу `DATE`. Строки с пустой датой отбрасываются.

### 2.2 TW
- Фильтруются строки без `link_title`, с `link_title = ""` или `link_title = "0"`.
- Каждая строка сохраняется в `tw_events` без агрегации. Дубликаты по кампании/офферу допускаются, агрегирование выполняется при построении витрины.
- Поля `first_purchases`, `registrations`, `installations` округляются до `int`; `install2reg`, `reg2dep`, `income_usd` — `float`.

### 2.3 Moloco
- Строки сортируются по дате и на каждый `(Date, Campaign)` оставляется последняя запись.
- Поля `Impression`, `Click`, `Install` сохраняются как `int`; `Spend`, `CPI` — `float`. Пустой `CPI` трактуется как `0`.

## 3. Построение витрины `campaign_statistics`

1. Загружаем все строки из `tw_events` и `moloco_events`.
2. Группируем TW по `tw_date`, `tw_link_title`, `tw_ad_campaign_name`:
   - суммируем `tw_first_purchases`, `tw_registrations`, `tw_installations`, `tw_income_usd`;
   - `tw_geo_country_code` и `tw_carrot_id` берём из последней строки группы.
3. Группируем Moloco по `m_date`, `m_campaign` (суммы по показам/кликам/установкам/расходам, средний `m_cpi`).
4. Соединяем TW с Moloco левым джойном по `date` и `campaign`.
5. Стоимость рассчитывается как `calculated_spend = tw_installations * m_cpi`. При отсутствии CPI расход = 0.
6. Прибыль `profit = tw_income_usd - calculated_spend`, ROI% = `profit / calculated_spend * 100`, в случае нулевого расхода ROI = 0.
7. Таблица `campaign_statistics` полностью очищается и заполняется пересчитанными строками.

## 4. Требования к корректности
- Сумма `tw_income_usd` в витрине равна сумме всех `income_usd` из текущего файла TW.
- Расход допускает отклонение до 5% по сравнению с исходным `Spend` Moloco из-за перерасчёта `CPI * Install`.
- Повторная загрузка TW или Moloco всегда приводит к полной пересборке витрины.
- Отсутствие данных Moloco не блокирует строку: расходы и ROI = 0.

## 5. Работа с БД
- СУБД: PostgreSQL.
- Строка подключения по умолчанию: `postgresql://tw_user:qwas1234@localhost:5432/tw_analytics`.
- Таблицы создаются автоматически при старте API (`src/db/setup.py`).

## 6. Дальнейшее развитие
- Перейти на инкрементальные загрузки без полного удаления staging.
- Поддержать версионность загрузок и хранение истории.
- Добавить расширенные метрики (install2reg, reg2dep, KPI, капы) и дополнительные источники данных.
