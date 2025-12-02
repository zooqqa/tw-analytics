"""Script to add new columns to existing database."""

from sqlalchemy import text
from src.db.session import engine

with engine.connect() as conn:
    try:
        # Add tw_d14_aov and tw_d14_purchases_sum to tw_events
        conn.execute(text("ALTER TABLE tw_events ADD COLUMN tw_d14_aov NUMERIC(12,2) DEFAULT 0"))
        conn.execute(text("ALTER TABLE tw_events ADD COLUMN tw_d14_purchases_sum NUMERIC(14,4) DEFAULT 0"))
        print("+ Added d14 columns to tw_events")
    except Exception as e:
        print(f"tw_events: {e}")

    try:
        # Add moloco fields
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_campaign_id TEXT"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_countries TEXT"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_app TEXT"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_app_id TEXT"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_creative TEXT"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_creative_id TEXT"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_conversion INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_ctr REAL DEFAULT 0"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_cpa REAL DEFAULT 0"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_cost_per_conversion REAL DEFAULT 0"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_registration INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_first_purchase INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE moloco_events ADD COLUMN m_purchase INTEGER DEFAULT 0"))
        print("+ Added new columns to moloco_events")
    except Exception as e:
        print(f"moloco_events: {e}")

    try:
        # Add campaign_statistics fields
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN tw_d14_aov NUMERIC(12,2) DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN tw_d14_purchases_sum NUMERIC(14,4) DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_campaign_id TEXT"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_app TEXT"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_app_id TEXT"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_creative TEXT"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_creative_id TEXT"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_conversion INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_ctr REAL DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_cpa REAL DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_registration INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_first_purchase INTEGER DEFAULT 0"))
        conn.execute(text("ALTER TABLE campaign_statistics ADD COLUMN m_purchase INTEGER DEFAULT 0"))
        print("+ Added new columns to campaign_statistics")
    except Exception as e:
        print(f"campaign_statistics: {e}")

    conn.commit()
    print("\nMigration completed!")
