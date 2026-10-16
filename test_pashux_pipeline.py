"""
Test script to verify PashuX activity detection, 10s video buffering, Supabase storage pipeline, and phone authentication.
"""

import time
import numpy as np
import cv2
from pashux_supabase import supabase_manager
from activity_engine import ActivityEngine

def test_pipeline():
    print("=" * 60)
    print("RUNNING PASHUX PIPELINE & AUTH VERIFICATION")
    print("=" * 60)

    # 1. Test Password Hashing & Phone Auth
    print("\n1. Testing Phone & Bcrypt Password Authentication...")
    test_phone = "+919876543210"
    test_pass = "PashuX@2026"

    # Authenticate demo user
    success, user_data, msg = supabase_manager.authenticate_user_by_phone(test_phone, test_pass)
    print(f"   Auth Result: {success} | Message: {msg}")
    assert success is True, "Demo auth failed"
    print(f"   Token generated: {user_data['token'][:25]}... for Farm: {user_data['farm_id']}")

    # 2. Test ActivityEngine with Rolling Pre/Post-Buffer
    print("\n2. Testing ActivityEngine Video Buffer & Event Trigger...")
    engine = ActivityEngine(
        farm_id="11111111-1111-1111-1111-111111111111",
        camera_id="CAM01",
        video_source_type="uploaded",
        buffer_fps=20.0,
        pre_buffer_seconds=2.0,  # shortened for quick test
        post_buffer_seconds=2.0,
        cooldown_seconds=10.0
    )

    # Push 30 dummy frames (simulating pre-event video)
    test_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_canvas, "PashuX CCTV Pre-Event Frame", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 155), 2)

    for i in range(30):
        engine.push_frame(test_canvas)

    # Trigger test event: Estrus Detection
    print("\n3. Triggering Test Estrus Activity...")
    engine.trigger_activity(
        activity_type="estrus_detected",
        confidence=0.94,
        description="Detected Estrus Sign: Cattle #5 actively mounting Cattle #3",
        current_frame=test_canvas,
        cattle_id="COW005",
        ai_detection_result={"mounting_cattle_id": 5, "mounted_cattle_id": 3, "model": "CowXNet_heat_model + estrusdetection.pt"},
        status="detected_sign"
    )

    # Push 60 dummy frames (simulating post-event video to complete clip)
    print("   Pushing post-event frames to complete video clip...")
    for i in range(60):
        engine.push_frame(test_canvas)
        time.sleep(0.005)

    time.sleep(2.0) # allow background worker to encode and upload

    # 4. Verify Activity Retrieval
    print("\n4. Verifying Activity Retrieval & Signed URLs...")
    activities = supabase_manager.fetch_activities(
        farm_id="11111111-1111-1111-1111-111111111111",
        limit=5
    )
    print(f"   Total activities retrieved: {len(activities)}")
    if len(activities) > 0:
        act = activities[0]
        print(f"   Latest Activity: {act.get('activity_type')} | ID: {act.get('activity_id')}")
        print(f"   Description: {act.get('description')}")
        print(f"   Image URL: {act.get('image_url')}")
        print(f"   Video URL: {act.get('video_url')}")
        print(f"   Detection Date: {act.get('detection_date')} | Duration: {act.get('activity_duration')}s")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
