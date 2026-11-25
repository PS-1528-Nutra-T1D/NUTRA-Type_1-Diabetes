-- ============================================================================
-- GlucoDex (GlucoPredict AI) — Production-Ready PostgreSQL Schema for Supabase
-- ============================================================================
-- Derived from: backend/app.py, frontend/index.html, frontend/login.html, frontend/profile.html
-- Target: PostgreSQL 15+ / Supabase
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. UPDATED_AT TRIGGER FUNCTION (shared across all mutable tables)
-- ============================================================================
CREATE OR REPLACE FUNCTION handle_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ============================================================================
-- 2. USERS / PROFILES TABLE
-- ============================================================================
-- Clerk authentication is external; we store clerk_user_id as PRIMARY KEY.
-- All profile fields from frontend/profile.html and frontend/login.html mapped here.
-- ============================================================================
CREATE TABLE profiles (
    clerk_user_id        TEXT PRIMARY KEY,                    -- Clerk 'sub' claim (e.g., 'user_2x...')
    email                TEXT NOT NULL UNIQUE,
    full_name            TEXT,
    avatar_url           TEXT,                                -- Base64 data URL or Supabase Storage path
    age                  SMALLINT CHECK (age BETWEEN 1 AND 120),
    bio                  TEXT,
    diabetes_type        TEXT NOT NULL DEFAULT 'type1'
        CHECK (diabetes_type IN ('type1','type2','prediabetes','gestational','other','research')),
    cgm_device           TEXT
        CHECK (cgm_device IN ('dexcom-g7','dexcom-g6','libre3','libre2','medtronic','manual','') OR cgm_device IS NULL),
    target_glucose_low   SMALLINT NOT NULL DEFAULT 70
        CHECK (target_glucose_low BETWEEN 40 AND 110),
    target_glucose_high  SMALLINT NOT NULL DEFAULT 180
        CHECK (target_glucose_high BETWEEN 120 AND 400),
    preferred_unit       TEXT NOT NULL DEFAULT 'mg/dL'
        CHECK (preferred_unit IN ('mg/dL','mmol/L')),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at           TIMESTAMPTZ
);

CREATE INDEX idx_profiles_email ON profiles(email);
CREATE TRIGGER trg_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW EXECUTE FUNCTION handle_updated_at();

-- ============================================================================
-- 3. USER PREFERENCES (Notification & UI Settings)
-- ============================================================================
-- From frontend/profile.html: notification toggles, dark mode default
-- ============================================================================
CREATE TABLE user_preferences (
    clerk_user_id        TEXT PRIMARY KEY REFERENCES profiles(clerk_user_id) ON DELETE CASCADE,
    notify_high_glucose  BOOLEAN NOT NULL DEFAULT TRUE,
    notify_low_glucose   BOOLEAN NOT NULL DEFAULT TRUE,
    notify_spike         BOOLEAN NOT NULL DEFAULT TRUE,
    notify_drop          BOOLEAN NOT NULL DEFAULT TRUE,
    dark_mode_default    BOOLEAN NOT NULL DEFAULT FALSE,
    audio_chime_enabled  BOOLEAN NOT NULL DEFAULT TRUE,
    desktop_notify_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER trg_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION handle_updated_at();

-- ============================================================================
-- 4. GLUCOSE READINGS / LOGS
-- ============================================================================
-- Raw CGM or fingerstick readings from CSV uploads (index.html dropzone) or manual entry.
-- Mirrors the columns expected by backend/app.py: GlucoseCGM, HR, ts (+ derived time_sin/time_cos).
-- ============================================================================
CREATE TABLE glucose_readings (
    id                   BIGSERIAL PRIMARY KEY,
    clerk_user_id        TEXT NOT NULL REFERENCES profiles(clerk_user_id) ON DELETE CASCADE,
    recorded_at          TIMESTAMPTZ NOT NULL,                -- Original timestamp from CSV ('ts' column)
    glucose_mgdl         REAL NOT NULL CHECK (glucose_mgdl BETWEEN 20 AND 600),  -- GlucoseCGM
    heart_rate           REAL CHECK (heart_rate BETWEEN 30 AND 220),             -- HR (optional)
    time_sin             REAL,                                -- sin(2π*minutes/1440)
    time_cos             REAL,                                -- cos(2π*minutes/1440)
    source               TEXT NOT NULL DEFAULT 'csv_upload'
        CHECK (source IN ('manual','csv_upload','api','cgm_stream')),
    notes                TEXT,                                -- Meal, insulin, activity tags
    carb_grams           REAL CHECK (carb_grams >= 0),
    insulin_units        REAL CHECK (insulin_units >= 0),
    activity_type        TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_glucose_readings_user_time ON glucose_readings(clerk_user_id, recorded_at DESC);
CREATE INDEX idx_glucose_readings_source ON glucose_readings(source);
CREATE TRIGGER trg_glucose_readings_updated_at
    BEFORE UPDATE ON glucose_readings
    FOR EACH ROW EXECUTE FUNCTION handle_updated_at();

-- ============================================================================
-- 5. CSV UPLOAD BATCHES (Audit Trail)
-- ============================================================================
-- Tracks each CSV file upload from index.html for reproducibility & debugging.
-- ============================================================================
CREATE TABLE csv_upload_batches (
    id                   BIGSERIAL PRIMARY KEY,
    clerk_user_id        TEXT NOT NULL REFERENCES profiles(clerk_user_id) ON DELETE CASCADE,
    filename             TEXT NOT NULL,
    file_size_bytes      BIGINT NOT NULL,
    row_count            INTEGER NOT NULL CHECK (row_count > 0),
    columns_detected     TEXT[] NOT NULL,                     -- e.g., {'GlucoseCGM','HR','ts'}
    has_hr_column        BOOLEAN NOT NULL DEFAULT FALSE,
    has_ts_column        BOOLEAN NOT NULL DEFAULT FALSE,
    processing_status    TEXT NOT NULL DEFAULT 'completed'
        CHECK (processing_status IN ('pending','processing','completed','failed')),
    error_message        TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_csv_batches_user_time ON csv_upload_batches(clerk_user_id, created_at DESC);

-- ============================================================================
-- 6. PREDICTIONS / FORECASTS
-- ============================================================================
-- Stores every inference run from backend/app.py `/predict` endpoint.
-- Includes model identifier, input window snapshot, forecast horizons, and risk classification.
-- ============================================================================
CREATE TABLE predictions (
    id                   BIGSERIAL PRIMARY KEY,
    clerk_user_id        TEXT NOT NULL REFERENCES profiles(clerk_user_id) ON DELETE CASCADE,
    csv_batch_id         BIGINT REFERENCES csv_upload_batches(id) ON DELETE SET NULL,
    model_name           TEXT NOT NULL
        CHECK (model_name IN (
            'Hybrid Bi-LSTM + Transformer',
            'Hybrid LSTM + Transformer',
            'BiLSTMTransformer',
            'HybridLSTMTransformer',
            'Linear Regression'
        )),
    model_version        TEXT,                                -- e.g., checkpoint hash or training date
    input_window         JSONB NOT NULL,                      -- Last 12 rows × 4 features (GlucoseCGM, HR, time_sin, time_cos)
    input_window_start   TIMESTAMPTZ NOT NULL,                -- Timestamp of first row in window
    input_window_end     TIMESTAMPTZ NOT NULL,                -- Timestamp of last row in window (t=0)
    current_glucose      REAL NOT NULL CHECK (current_glucose BETWEEN 20 AND 600),
    forecast_30min       REAL[] NOT NULL,                     -- 6 × 5-min steps
    forecast_60min       REAL[] NOT NULL,                     -- 12 × 5-min steps
    forecast_120min      REAL[] NOT NULL,                     -- 24 × 5-min steps
    risk_classification  TEXT NOT NULL
        CHECK (risk_classification IN ('critical_hypo','hypo','stable','hyper','critical_hyper','rapid_drop','rapid_rise')),
    feeling_text         TEXT NOT NULL,                       -- Full clinical assessment from backend
    metrics_snapshot     JSONB NOT NULL DEFAULT '{}',         -- MODEL_METRICS at time of prediction
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_predictions_user_time ON predictions(clerk_user_id, created_at DESC);
CREATE INDEX idx_predictions_risk ON predictions(risk_classification);
CREATE INDEX idx_predictions_model ON predictions(model_name);

-- ============================================================================
-- 7. ALERT EVENTS (Threshold Breaches)
-- ============================================================================
-- Logs when forecast horizons breach user-defined thresholds (profile.html settings).
-- Used for notification delivery and audit trail.
-- ============================================================================
CREATE TABLE alert_events (
    id                   BIGSERIAL PRIMARY KEY,
    clerk_user_id        TEXT NOT NULL REFERENCES profiles(clerk_user_id) ON DELETE CASCADE,
    prediction_id        BIGINT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    alert_type           TEXT NOT NULL
        CHECK (alert_type IN ('hypo','hyper','rapid_drop','rapid_rise')),
    horizon_minutes      SMALLINT NOT NULL CHECK (horizon_minutes IN (30,60,120)),
    predicted_value      REAL NOT NULL,
    threshold_value      REAL NOT NULL,
    breach_direction     TEXT NOT NULL CHECK (breach_direction IN ('below','above')),
    notified             BOOLEAN NOT NULL DEFAULT FALSE,
    notification_channels TEXT[] DEFAULT '{}',                -- e.g., {'audio','desktop','push'}
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alert_events_user_time ON alert_events(clerk_user_id, created_at DESC);
CREATE INDEX idx_alert_events_notified ON alert_events(notified) WHERE notified = FALSE;

-- ============================================================================
-- 8. SAMPLE SCENARIOS (Reference Data)
-- ============================================================================
-- The four built-in clinical test scenarios from index.html (Sample Deck).
-- ============================================================================
CREATE TABLE sample_scenarios (
    id                   SMALLSERIAL PRIMARY KEY,
    slug                 TEXT NOT NULL UNIQUE,                -- 'spike','drop','dip_test','stable'
    display_name         TEXT NOT NULL,                       -- 'Postprandial Spike', etc.
    description          TEXT NOT NULL,
    expected_risk        TEXT NOT NULL
        CHECK (expected_risk IN ('critical_hypo','hypo','stable','hyper','critical_hyper','rapid_drop','rapid_rise')),
    csv_content          TEXT NOT NULL,                       -- Full CSV content as text
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed the four built-in scenarios (content truncated for brevity; populate via migration)
INSERT INTO sample_scenarios (slug, display_name, description, expected_risk, csv_content) VALUES
('spike', 'Postprandial Spike', 'Simulates rapid post-meal glucose excursion', 'rapid_rise', ''),
('drop', 'Acute Hypoglycemic Drop', 'Simulates insulin-induced hypoglycemia', 'rapid_drop', ''),
('dip_test', 'Hypo Recovery Rebound', 'Simulates counter-regulatory rebound after low', 'hypo', ''),
('stable', 'Euglycemic Baseline', 'Stable in-range glucose profile', 'stable', '');

-- ============================================================================
-- 9. ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================
-- All tables enable RLS. Policies use Clerk JWT 'sub' claim via auth.jwt().
-- Service role bypasses RLS for backend API operations.
-- ============================================================================

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE glucose_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE csv_upload_batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_events ENABLE ROW LEVEL SECURITY;

-- Helper: extract Clerk user ID from JWT
-- Supabase sets auth.jwt() from the Authorization header.
-- Clerk JWT 'sub' claim = clerk_user_id (e.g., 'user_2x...')
-- Service role has 'role' = 'service_role' in JWT.

-- profiles: users can read/update their own profile; service_role full access
CREATE POLICY "profiles_select_own" ON profiles
    FOR SELECT USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "profiles_insert_own" ON profiles
    FOR INSERT WITH CHECK (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "profiles_update_own" ON profiles
    FOR UPDATE USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "profiles_service_role" ON profiles
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- user_preferences: same pattern
CREATE POLICY "prefs_select_own" ON user_preferences
    FOR SELECT USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "prefs_upsert_own" ON user_preferences
    FOR INSERT WITH CHECK (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "prefs_update_own" ON user_preferences
    FOR UPDATE USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "prefs_service_role" ON user_preferences
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- glucose_readings: users can CRUD their own readings; service_role full access
CREATE POLICY "readings_select_own" ON glucose_readings
    FOR SELECT USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "readings_insert_own" ON glucose_readings
    FOR INSERT WITH CHECK (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "readings_update_own" ON glucose_readings
    FOR UPDATE USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "readings_delete_own" ON glucose_readings
    FOR DELETE USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "readings_service_role" ON glucose_readings
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- csv_upload_batches: users can read/insert their own; service_role full access
CREATE POLICY "batches_select_own" ON csv_upload_batches
    FOR SELECT USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "batches_insert_own" ON csv_upload_batches
    FOR INSERT WITH CHECK (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "batches_service_role" ON csv_upload_batches
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- predictions: users can read their own; service_role writes via backend
CREATE POLICY "predictions_select_own" ON predictions
    FOR SELECT USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "predictions_service_role" ON predictions
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- alert_events: users can read their own; service_role writes
CREATE POLICY "alerts_select_own" ON alert_events
    FOR SELECT USING (clerk_user_id = (auth.jwt() ->> 'sub'));
CREATE POLICY "alerts_service_role" ON alert_events
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- ============================================================================
-- 10. HELPER VIEWS FOR COMMON FRONTEND QUERIES
-- ============================================================================

-- Latest prediction per user with risk badge (for dashboard hero)
CREATE VIEW latest_prediction_per_user AS
SELECT DISTINCT ON (p.clerk_user_id)
    p.clerk_user_id,
    p.id AS prediction_id,
    p.model_name,
    p.current_glucose,
    p.forecast_30min,
    p.forecast_60min,
    p.forecast_120min,
    p.risk_classification,
    p.feeling_text,
    p.created_at
FROM predictions p
ORDER BY p.clerk_user_id, p.created_at DESC;

-- Aggregated glucose stats per user (for profile stats cards)
CREATE VIEW user_glucose_stats AS
SELECT
    gr.clerk_user_id,
    COUNT(*) AS total_readings,
    ROUND(AVG(gr.glucose_mgdl)::numeric, 1) AS avg_glucose_mgdl,
    MIN(gr.glucose_mgdl) AS min_glucose_mgdl,
    MAX(gr.glucose_mgdl) AS max_glucose_mgdl,
    COUNT(*) FILTER (WHERE gr.glucose_mgdl < 70) AS hypo_count,
    COUNT(*) FILTER (WHERE gr.glucose_mgdl > 180) AS hyper_count,
    MAX(gr.recorded_at) AS last_reading_at
FROM glucose_readings gr
GROUP BY gr.clerk_user_id;

-- Prediction count per user (for profile stats)
CREATE VIEW user_prediction_stats AS
SELECT
    p.clerk_user_id,
    COUNT(*) AS total_predictions,
    MAX(p.created_at) AS last_prediction_at
FROM predictions p
GROUP BY p.clerk_user_id;

-- ============================================================================
-- 11. GRANTS FOR SERVICE ROLE (Backend API)
-- ============================================================================
-- The FastAPI backend connects with Supabase service_role key.
-- Ensure it has full access to all tables.
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================