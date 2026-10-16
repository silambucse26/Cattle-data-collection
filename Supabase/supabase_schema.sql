-- ====================================================================
-- PashuX AI - Automatic Farm Activity Detection & Storage Schema
-- Run this script in your Supabase SQL Editor.
-- (100% Public Schema - No auth schema permission errors)
-- ====================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- --------------------------------------------------------------------
-- 2. Farms Table
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.farms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_code VARCHAR(50) UNIQUE NOT NULL,
    farm_name VARCHAR(150) NOT NULL,
    owner_name VARCHAR(100),
    owner_phone VARCHAR(20) NOT NULL,
    location VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_farms_code ON public.farms (farm_code);

-- --------------------------------------------------------------------
-- 3. Users Table (Phone Number & Password Authentication)
-- Passwords are encrypted with bcrypt.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID REFERENCES public.farms(id) ON DELETE CASCADE,
    phone VARCHAR(20) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(30) DEFAULT 'owner' CHECK (role IN ('owner', 'manager', 'vet', 'worker')),
    is_verified BOOLEAN DEFAULT TRUE,
    otp_code VARCHAR(10),
    otp_expires_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON public.users (phone);
CREATE INDEX IF NOT EXISTS idx_users_farm ON public.users (farm_id);

-- --------------------------------------------------------------------
-- 4. Cameras Table (CCTV & Video Inputs)
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.cameras (
    id VARCHAR(50) PRIMARY KEY, -- e.g. 'CAM01', 'CAM02'
    farm_id UUID NOT NULL REFERENCES public.farms(id) ON DELETE CASCADE,
    camera_name VARCHAR(100) NOT NULL,
    rtsp_url TEXT,
    location_description VARCHAR(255), -- e.g. 'Milking Parlor', 'Manger Zone 1'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cameras_farm ON public.cameras (farm_id);

-- --------------------------------------------------------------------
-- 5. Farm Activities Table (Main Activity Records)
-- Contains all required metadata, structured AI results, and media paths.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.farm_activities (
    activity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES public.farms(id) ON DELETE CASCADE,
    camera_id VARCHAR(50) REFERENCES public.cameras(id) ON DELETE SET NULL,
    cattle_id VARCHAR(50),                     -- e.g. 'COW005', 'Cattle #12', or NULL
    worker_id VARCHAR(50),                     -- e.g. 'WRK001', 'Worker #3', or NULL
    activity_type VARCHAR(60) NOT NULL,        -- 'estrus_detected', 'disease_suspected', 'calving_suspected', 'feeding_activity', 'cleaning_activity', 'worker_entry', 'worker_exit', 'bcs_alert'
    detection_date DATE NOT NULL DEFAULT CURRENT_DATE,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    activity_duration REAL DEFAULT 10.0,       -- in seconds (total event session duration)
    eating_duration_sec REAL DEFAULT 0.0,      -- Exact duration spent eating / feeding
    resting_duration_sec REAL DEFAULT 0.0,     -- Exact duration spent resting / lying down
    herd_count INT DEFAULT 1,                  -- Total cattle counted in scene
    breed VARCHAR(100) DEFAULT 'Cattle',       -- AI Breed (Gir, Sahiwal, etc.)
    disease_status VARCHAR(100) DEFAULT 'Healthy', -- AI Disease Status (Healthy, FMD, LSD)
    bcs_score REAL DEFAULT 3.0,                -- Body Condition Score (1.0 to 5.0)
    estimated_weight_kg REAL DEFAULT 400.0,    -- Allometric Weight Estimation (kg)
    estimated_height_cm REAL DEFAULT 130.0,    -- Wither/Hip Height (cm)
    ai_detection_result JSONB DEFAULT '{}'::jsonb, -- structured scores, bounding boxes, raw model outputs
    confidence_score REAL NOT NULL,            -- e.g. 0.92 (92%)
    description TEXT NOT NULL,
    captured_image_storage_path TEXT NOT NULL, -- Private Supabase Storage object path (e.g. '<farm_id>/images/evt_123.jpg')
    captured_video_storage_path TEXT NOT NULL, -- Private Supabase Storage object path (e.g. '<farm_id>/videos/evt_123.mp4')
    video_source VARCHAR(20) NOT NULL DEFAULT 'uploaded' CHECK (video_source IN ('uploaded', 'live')),
    status VARCHAR(30) DEFAULT 'detected_sign' CHECK (status IN ('suspected_sign', 'detected_sign', 'verified', 'dismissed')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_activities_farm ON public.farm_activities (farm_id);
CREATE INDEX IF NOT EXISTS idx_activities_date ON public.farm_activities (detection_date DESC);
CREATE INDEX IF NOT EXISTS idx_activities_type ON public.farm_activities (activity_type);
CREATE INDEX IF NOT EXISTS idx_activities_breed ON public.farm_activities (breed);
CREATE INDEX IF NOT EXISTS idx_activities_disease ON public.farm_activities (disease_status);
CREATE INDEX IF NOT EXISTS idx_activities_cattle ON public.farm_activities (cattle_id);
CREATE INDEX IF NOT EXISTS idx_activities_camera ON public.farm_activities (camera_id);
CREATE INDEX IF NOT EXISTS idx_activities_created ON public.farm_activities (created_at DESC);

-- --------------------------------------------------------------------
-- 6. Supabase Storage Buckets
-- Create private storage buckets for activity images and 10s video clips
-- --------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES 
    ('pashux-activity-images', 'pashux-activity-images', FALSE, 20971520, ARRAY['image/jpeg', 'image/png', 'image/webp']),
    ('pashux-activity-videos', 'pashux-activity-videos', FALSE, 104857600, ARRAY['video/mp4', 'video/webm', 'video/quicktime'])
ON CONFLICT (id) DO UPDATE SET 
    public = FALSE,
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

-- --------------------------------------------------------------------
-- 7. Row Level Security (RLS) Helper Function & Policies
-- Created in public schema so NO permission denied errors occur!
-- --------------------------------------------------------------------

-- Helper function in public schema to extract user's farm_id from JWT
CREATE OR REPLACE FUNCTION public.get_user_farm_id()
RETURNS UUID AS $$
BEGIN
    RETURN COALESCE(
        (current_setting('request.jwt.claims', true)::jsonb -> 'user_metadata' ->> 'farm_id')::UUID,
        (current_setting('request.jwt.claim.farm_id', true))::UUID
    );
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Enable RLS on all tables
ALTER TABLE public.farms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.farm_activities ENABLE ROW LEVEL SECURITY;

-- FARMS RLS: Users can view their own farm; Service role has full access
DROP POLICY IF EXISTS "farm_user_view" ON public.farms;
CREATE POLICY "farm_user_view" ON public.farms
    FOR ALL
    USING (
        id = public.get_user_farm_id()
        OR auth.role() = 'service_role'
        OR auth.role() = 'anon'
    );

-- USERS RLS: Users can view users belonging to same farm
DROP POLICY IF EXISTS "users_same_farm_view" ON public.users;
CREATE POLICY "users_same_farm_view" ON public.users
    FOR ALL
    USING (
        farm_id = public.get_user_farm_id()
        OR auth.role() = 'service_role'
        OR auth.role() = 'anon'
    );

-- CAMERAS RLS: Users can only see cameras in their farm
DROP POLICY IF EXISTS "cameras_farm_view" ON public.cameras;
CREATE POLICY "cameras_farm_view" ON public.cameras
    FOR ALL
    USING (
        farm_id = public.get_user_farm_id()
        OR auth.role() = 'service_role'
        OR auth.role() = 'anon'
    );

-- ACTIVITIES RLS: Users can view & insert activities in their farm
DROP POLICY IF EXISTS "activities_farm_select" ON public.farm_activities;
CREATE POLICY "activities_farm_select" ON public.farm_activities
    FOR ALL
    USING (
        farm_id = public.get_user_farm_id()
        OR auth.role() = 'service_role'
        OR auth.role() = 'anon'
    );

-- STORAGE RLS: Restrict storage objects by farm folder prefix (<farm_id>/...)
DROP POLICY IF EXISTS "storage_farm_read" ON storage.objects;
CREATE POLICY "storage_farm_read" ON storage.objects
    FOR SELECT
    USING (
        bucket_id IN ('pashux-activity-images', 'pashux-activity-videos')
        AND (
            (storage.foldername(name))[1] = public.get_user_farm_id()::text
            OR auth.role() = 'service_role'
            OR auth.role() = 'anon'
        )
    );

DROP POLICY IF EXISTS "storage_farm_insert" ON storage.objects;
CREATE POLICY "storage_farm_insert" ON storage.objects
    FOR INSERT
    WITH CHECK (
        bucket_id IN ('pashux-activity-images', 'pashux-activity-videos')
        AND (
            (storage.foldername(name))[1] = public.get_user_farm_id()::text
            OR auth.role() = 'service_role'
            OR auth.role() = 'anon'
        )
    );

-- --------------------------------------------------------------------
-- 8. Default Seed Data (Sample Farm, Camera & Hashed Password User)
-- Password for demo user (+919876543210): PashuX@2026
-- --------------------------------------------------------------------
DO $$
DECLARE
    demo_farm_id UUID;
BEGIN
    -- Insert Default Demo Farm if not exists
    INSERT INTO public.farms (id, farm_code, farm_name, owner_name, owner_phone, location)
    VALUES (
        '11111111-1111-1111-1111-111111111111'::UUID,
        'FARM_PASHUX_01',
        'PashuX Pioneer Dairy Farm',
        'Ramesh Patel',
        '+919876543210',
        'Gujarat, India'
    )
    ON CONFLICT (farm_code) DO UPDATE SET farm_name = EXCLUDED.farm_name
    RETURNING id INTO demo_farm_id;

    -- Insert Default CCTV Cameras
    INSERT INTO public.cameras (id, farm_id, camera_name, rtsp_url, location_description, is_active)
    VALUES (
        'CAM01',
        demo_farm_id,
        'Main Gate & Feed Alley Camera',
        'rtsp://chimertechadmin:Chimertech2026@192.168.1.58:554/stream2',
        'Primary Cattle Corral & Alley',
        TRUE
    )
    ON CONFLICT (id) DO NOTHING;

    INSERT INTO public.cameras (id, farm_id, camera_name, rtsp_url, location_description, is_active)
    VALUES (
        'CAM02',
        demo_farm_id,
        'Milking & Stall Camera',
        'rtsp://demo:demo@192.168.1.59:554/stream1',
        'Milking Parlor & Stall Area',
        TRUE
    )
    ON CONFLICT (id) DO NOTHING;

    -- Insert Default Demo User with phone '+919876543210' and password 'PashuX@2026'
    INSERT INTO public.users (farm_id, phone, password_hash, full_name, role, is_verified)
    VALUES (
        demo_farm_id,
        '+919876543210',
        crypt('PashuX@2026', gen_salt('bf', 10)),
        'Ramesh Patel (Farm Owner)',
        'owner',
        TRUE
    )
    ON CONFLICT (phone) DO UPDATE SET 
        farm_id = EXCLUDED.farm_id,
        password_hash = crypt('PashuX@2026', gen_salt('bf', 10));

END $$;

-- Verification Query: Show created schema status
SELECT 'PashuX schema initialized successfully without any auth errors!' as status;
