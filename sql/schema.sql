-- Customer churn analysis — PostgreSQL schema
-- Placeholder DDL for future CRM integration.
-- Run: psql -d churn_db -f sql/schema.sql

CREATE TABLE IF NOT EXISTS customers (
    customer_id        TEXT PRIMARY KEY,
    gender             TEXT NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen     SMALLINT NOT NULL DEFAULT 0 CHECK (senior_citizen IN (0, 1)),
    partner            TEXT NOT NULL CHECK (partner IN ('Yes', 'No')),
    dependents         TEXT NOT NULL CHECK (dependents IN ('Yes', 'No')),
    tenure             INT NOT NULL CHECK (tenure >= 0),
    phone_service      TEXT NOT NULL CHECK (phone_service IN ('Yes', 'No')),
    multiple_lines     TEXT,
    internet_service   TEXT NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    online_security    TEXT,
    online_backup      TEXT,
    device_protection  TEXT,
    tech_support       TEXT,
    streaming_tv       TEXT,
    streaming_movies   TEXT,
    contract           TEXT NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    paperless_billing  TEXT NOT NULL CHECK (paperless_billing IN ('Yes', 'No')),
    payment_method     TEXT NOT NULL,
    monthly_charges    NUMERIC(10, 2) NOT NULL CHECK (monthly_charges > 0),
    total_charges      NUMERIC(10, 2),
    churned            SMALLINT CHECK (churned IN (0, 1)),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id      BIGSERIAL PRIMARY KEY,
    customer_id        TEXT NOT NULL REFERENCES customers(customer_id),
    model_version      TEXT NOT NULL,
    churn_probability  NUMERIC(6, 4) NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    churn_prediction   SMALLINT NOT NULL CHECK (churn_prediction IN (0, 1)),
    threshold_used     NUMERIC(6, 4) NOT NULL,
    predicted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retention_campaigns (
    campaign_id        BIGSERIAL PRIMARY KEY,
    prediction_id      BIGINT NOT NULL REFERENCES churn_predictions(prediction_id),
    customer_id        TEXT NOT NULL REFERENCES customers(customer_id),
    offer_type         TEXT NOT NULL,
    offer_cost         NUMERIC(10, 2) NOT NULL,
    sent_at            TIMESTAMPTZ,
    accepted           BOOLEAN,
    outcome_recorded_at TIMESTAMPTZ
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON churn_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON churn_predictions(churn_probability DESC);
CREATE INDEX IF NOT EXISTS idx_campaigns_customer ON retention_campaigns(customer_id);
