-- =====================================================================
-- PASHUX AI - SUPABASE MIGRATION: CLEAN CATTLE TELEMETRY COLUMNS
-- Adds dedicated columns for:
-- 1. Feeding timing & duration (eating_duration_sec)
-- 2. Resting timing & duration (resting_duration_sec)
-- 3. AI Breed classification (breed)
-- 4. AI Disease status (disease_status)
-- 5. Body Condition Score (bcs_score)
-- 6. Live Weight estimation (estimated_weight_kg)
-- 7. Height estimation (estimated_height_cm)
-- 8. Herd counting (herd_count)
-- =====================================================================

ALTER TABLE public.farm_activities 
ADD COLUMN IF NOT EXISTS breed VARCHAR(100) DEFAULT 'Cattle',
ADD COLUMN IF NOT EXISTS disease_status VARCHAR(100) DEFAULT 'Healthy',
ADD COLUMN IF NOT EXISTS bcs_score REAL DEFAULT 3.0,
ADD COLUMN IF NOT EXISTS estimated_weight_kg REAL DEFAULT 400.0,
ADD COLUMN IF NOT EXISTS estimated_height_cm REAL DEFAULT 130.0,
ADD COLUMN IF NOT EXISTS eating_duration_sec REAL DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS resting_duration_sec REAL DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS herd_count INT DEFAULT 1;

-- Add indexes for fast filtering
CREATE INDEX IF NOT EXISTS idx_activities_breed ON public.farm_activities (breed);
CREATE INDEX IF NOT EXISTS idx_activities_disease ON public.farm_activities (disease_status);
CREATE INDEX IF NOT EXISTS idx_activities_bcs ON public.farm_activities (bcs_score);

-- Notify schema cache update
NOTIFY pgrst, 'reload schema';
