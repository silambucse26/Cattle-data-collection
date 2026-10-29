import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from pashux_supabase import (
    supabase_manager,
    LOCAL_CAPTURE_DIR,
    LOCAL_IMAGES_DIR,
    LOCAL_VIDEOS_DIR,
    LOCAL_DB_FILE,
    CATTLE_ROSTER_FILE
)

print("=" * 60)
print("1. WIPING SUPABASE CLOUD DATABASE...")
if supabase_manager.is_connected and supabase_manager.client:
    try:
        # Delete in chunks or with filter
        res = supabase_manager.client.table("farm_activities").delete().neq("activity_id", "00000000-0000-0000-0000-000000000000").execute()
        print("[SUCCESS] Wiped ALL rows from Supabase farm_activities table!")
    except Exception as e:
        print(f"[Supabase Notice] {e}")
else:
    print("[NOTICE] Supabase client not connected.")

print("\n2. WIPING LOCAL DATABASE & ROSTER FILES...")
with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
    json.dump([], f)
print("[SUCCESS] Cleared captures/local_activities.json")

with open(CATTLE_ROSTER_FILE, "w", encoding="utf-8") as f:
    json.dump({}, f)
print("[SUCCESS] Cleared captures/cattle_roster.json")

print("\n3. DELETING ALL OLD/STALE CROP IMAGES...")
img_count = 0
if LOCAL_IMAGES_DIR.exists():
    for f in LOCAL_IMAGES_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
                img_count += 1
            except Exception:
                pass
print(f"[SUCCESS] Deleted {img_count} old leftover images from captures/images/")

print("\n4. DELETING STALE LOCAL VIDEO CLIPS...")
vid_count = 0
if LOCAL_VIDEOS_DIR.exists():
    for f in LOCAL_VIDEOS_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
                vid_count += 1
            except Exception:
                pass
print(f"[SUCCESS] Deleted {vid_count} old video clips from captures/videos/")

print("=" * 60)
print("CLEANUP COMPLETED: SUPABASE AND LOCAL DATA ARE COMPLETELY RESET TO 0")
print("=" * 60)
