-- =====================================================================
-- PASHUX AI - SUPABASE MIGRATION: CATTLE & WORKERS PERSISTENCE TABLES
-- Run this script in your Supabase SQL Editor:
-- (Supabase Dashboard -> SQL Editor -> New Query -> Paste & Run)
-- =====================================================================

-- 1. Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------
-- 2. Cattle Table (Dedicated Cattle Registry per Farm)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.cattle (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES public.farms(id) ON DELETE CASCADE,
    cattle_id VARCHAR(50) NOT NULL,
    camera_id VARCHAR(50) REFERENCES public.cameras(id) ON DELETE SET NULL,
    breed VARCHAR(100) DEFAULT 'Cattle',
    disease_status VARCHAR(100) DEFAULT 'Healthy',
    bcs_score REAL DEFAULT 3.0,
    estimated_weight_kg REAL DEFAULT 400.0,
    estimated_height_cm REAL DEFAULT 130.0,
    image_url TEXT,
    captured_image_storage_path TEXT,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_farm_cattle UNIQUE (farm_id, cattle_id)
);

CREATE INDEX IF NOT EXISTS idx_cattle_farm ON public.cattle (farm_id);
CREATE INDEX IF NOT EXISTS idx_cattle_tag ON public.cattle (cattle_id);
CREATE INDEX IF NOT EXISTS idx_cattle_breed ON public.cattle (breed);
CREATE INDEX IF NOT EXISTS idx_cattle_disease ON public.cattle (disease_status);

-- ---------------------------------------------------------------------
-- 3. Workers Table (Dedicated Farm Workers Registry per Farm)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES public.farms(id) ON DELETE CASCADE,
    worker_id VARCHAR(50) NOT NULL,
    camera_id VARCHAR(50) REFERENCES public.cameras(id) ON DELETE SET NULL,
    image_url TEXT,
    captured_image_storage_path TEXT,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_farm_worker UNIQUE (farm_id, worker_id)
);

CREATE INDEX IF NOT EXISTS idx_workers_farm ON public.workers (farm_id);
CREATE INDEX IF NOT EXISTS idx_workers_tag ON public.workers (worker_id);

-- ---------------------------------------------------------------------
-- 4. Enable Row Level Security (RLS) & Policies
-- ---------------------------------------------------------------------
ALTER TABLE public.cattle ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.workers ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "cattle_farm_all" ON public.cattle;
CREATE POLICY "cattle_farm_all" ON public.cattle
    FOR ALL
    USING (
        farm_id = public.get_user_farm_id()
        OR auth.role() = 'service_role'
        OR auth.role() = 'anon'
    );

DROP POLICY IF EXISTS "workers_farm_all" ON public.workers;
CREATE POLICY "workers_farm_all" ON public.workers
    FOR ALL
    USING (
        farm_id = public.get_user_farm_id()
        OR auth.role() = 'service_role'
        OR auth.role() = 'anon'
    );

-- Reload PostgREST schema cache
NOTIFY pgrst, 'reload schema';

SELECT 'PashuX cattle and workers tables initialized successfully!' as status;
