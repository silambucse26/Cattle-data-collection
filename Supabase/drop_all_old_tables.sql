-- =====================================================================
-- PASHUX AI - COMPLETE DATABASE RESET & CLEANUP SCRIPT
-- Run this script in your Supabase SQL Editor to wipe out all
-- existing tables, sample data, and foreign key constraints.
-- =====================================================================

-- 1. Disable triggers temporarily for clean dropping
SET session_replication_role = 'replica';

-- 2. Drop all previous schema tables (with CASCADE to remove all relationships)
DROP TABLE IF EXISTS public.daily_summaries CASCADE;
DROP TABLE IF EXISTS public.farm_alerts CASCADE;
DROP TABLE IF EXISTS public.cctv_camera_events CASCADE;
DROP TABLE IF EXISTS public.environment_iot CASCADE;
DROP TABLE IF EXISTS public.cattle_activity CASCADE;
DROP TABLE IF EXISTS public.cleaning_events CASCADE;
DROP TABLE IF EXISTS public.feeding_events CASCADE;
DROP TABLE IF EXISTS public.cattle_milk_yields CASCADE;
DROP TABLE IF EXISTS public.daily_milking_totals CASCADE;
DROP TABLE IF EXISTS public.worker_attendance CASCADE;
DROP TABLE IF EXISTS public.cattle CASCADE;
DROP TABLE IF EXISTS public.workers CASCADE;
DROP TABLE IF EXISTS public.cctv_cameras CASCADE;

-- 3. Also drop any previous PashuX tables if resetting fresh
DROP TABLE IF EXISTS public.farm_activities CASCADE;
DROP TABLE IF EXISTS public.cameras CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP TABLE IF EXISTS public.farms CASCADE;

-- 4. Re-enable standard trigger behavior
SET session_replication_role = 'origin';

-- 5. Drop triggers and helper functions
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP FUNCTION IF EXISTS public.handle_new_auth_user() CASCADE;
DROP FUNCTION IF EXISTS auth.get_user_farm_id() CASCADE;

-- Verification
SELECT 'All existing tables and sample data have been completely deleted! Database is clean.' AS status;
