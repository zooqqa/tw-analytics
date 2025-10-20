# Значения, выводимые сервисом

## Кампания (агрегированные данные)
- **stat_date** — дата статистики (TrafficWave `tw_events.tw_date`).
- **tw_ad_campaign_name** — название кампании (TrafficWave).
- **tw_link_title**, **tw_carrot_id**, **tw_geo_country_code** — атрибуты кампании/оффера из TrafficWave.
- **tw_first_purchases**, **tw_registrations**, **tw_installations** — суммы по TrafficWave.
- **tw_install2reg** — `tw_registrations / tw_installations * 100`.
- **tw_reg2dep** — `tw_first_purchases / tw_registrations * 100`.
- **tw_income_usd**, **tw_average_bill**, **tw_purchases_sum**, **tw_epc** — агрегированные метрики TrafficWave.
- **m_impression**, **m_click**, **m_install**, **m_spend** — суммы по Moloco (агрегация по дате и кампании).
- **m_cpi** — исходный Moloco CPI (среднее `Moloco Spend / Moloco Install` в исходных данных). Используется как базовое значение, если в TW нет инсталлов.
- **calculated_spend** — если Moloco spend по кампании > 0, используется он; иначе `m_cpi * tw_installations`.
- **calculated_cpi** — `calculated_spend / tw_installations` (если установки > 0) либо `m_cpi`, если установок нет.
- **profit** — `tw_income_usd - calculated_spend`.
- **roi_pct** — `(profit / calculated_spend) * 100`, если spend > 0.
- **created_at** — время формирования строки витрины.

### Дополнительно (эндпоинт `/stats`)
- **total_revenue**, **total_spend**, **total_profit**, **total_roi**, **records** — агрегированы из `calculated_spend` и `tw_income_usd` после фильтрации.

## Офферы (иерархический вид)
- **tw_link_title**, **tw_carrot_id**, **tw_geo_country_code**, TrafficWave метрики (`tw_installations`, `tw_registrations`, `tw_first_purchases`, `tw_income_usd` и др.) — агрегированы по офферу из TrafficWave.
- **calculated_cpi** (равно **m_cpi**) — наследуется от кампании: `Moloco spend по кампании / TW installs кампании` (или `m_cpi`, если TW инсталлов нет).
- **calculated_spend** — `calculated_cpi * tw_installations` оффера.
- **profit** — `tw_income_usd - calculated_spend`.
- **roi_pct** — `(profit / calculated_spend) * 100`, если spend > 0.
- **Moloco поля (`m_impression`, `m_click`, `m_install`, `m_spend`)** отсутствуют.
