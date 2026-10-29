"""
PashuX AI - Supabase Database, Storage & Phone Authentication Engine
Handles:
- Secure Phone & Bcrypt Password Authentication
- Farm-Level Access Control (RLS-aligned)
- Private Supabase Storage Uploads (Images & 10-Second Videos)
- Temporary Signed URL Generation for Authorized Playback
- Fallback Local Storage & Offline-Safe Queue
"""

import os
import sys
import time
import json
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import bcrypt
import jwt
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()
DEFAULT_FARM_ID = os.getenv("FARM_ID", "584cab92-d75d-4aa0-824a-c07423aa6453").strip()
DEFAULT_CAMERA_ID = os.getenv("CAMERA_ID", "CAM01").strip()
IMAGE_BUCKET = os.getenv("STORAGE_BUCKET_IMAGES", "pashux-activity-images").strip()
VIDEO_BUCKET = os.getenv("STORAGE_BUCKET_VIDEOS", "pashux-activity-videos").strip()
JWT_SECRET = os.getenv("JWT_SECRET", "pashux-cattle-ai-secret-key-2026-production-super-secure")

# Local fallback directory for offline storage & caching
LOCAL_CAPTURE_DIR = Path("captures")
LOCAL_IMAGES_DIR = LOCAL_CAPTURE_DIR / "images"
LOCAL_VIDEOS_DIR = LOCAL_CAPTURE_DIR / "videos"
LOCAL_DB_FILE = LOCAL_CAPTURE_DIR / "local_activities.json"
CATTLE_ROSTER_FILE = LOCAL_CAPTURE_DIR / "cattle_roster.json"
WORKER_ROSTER_FILE = LOCAL_CAPTURE_DIR / "worker_roster.json"

for d in [LOCAL_CAPTURE_DIR, LOCAL_IMAGES_DIR, LOCAL_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class SupabaseManager:
    """Manages Supabase authentication, database queries, and storage operations."""

    def __init__(self):
        self.client = None
        self.is_connected = False
        self._init_client()

    def _init_client(self):
        """Initializes the Supabase client if valid credentials exist."""
        global SUPABASE_URL, SUPABASE_KEY
        SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
        SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()

        if SUPABASE_URL and SUPABASE_KEY and not SUPABASE_URL.startswith("https://your-project"):
            try:
                from supabase import create_client, Client
                self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                self.is_connected = True
                print(f"[PashuX Supabase] Connected successfully to: {SUPABASE_URL}")
            except Exception as e:
                print(f"[PashuX Supabase] Warning: Could not connect to Supabase: {e}")
                self.client = None
                self.is_connected = False
        else:
            self.client = None
            self.is_connected = False
            print("[PashuX Supabase] Running in local/offline fallback mode (Set SUPABASE_URL & SUPABASE_KEY in .env for live cloud sync).")

    # -----------------------------------------------------------------
    # Password Hashing & Phone Authentication
    # -----------------------------------------------------------------
    @staticmethod
    def hash_password(password: str) -> str:
        """Securely hashes plaintext password with bcrypt salt (cost factor 12)."""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifies plaintext password against stored bcrypt hash."""
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception:
            return False

    def generate_jwt_token(self, user_id: str, farm_id: str, phone: str, role: str) -> str:
        """Generates a secure JWT token with user farm membership claims."""
        payload = {
            "sub": user_id,
            "user_id": user_id,
            "farm_id": farm_id,
            "phone": phone,
            "role": role,
            "user_metadata": {
                "farm_id": farm_id,
                "phone": phone,
                "role": role
            },
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7),
            "iat": datetime.datetime.now(datetime.timezone.utc)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    def decode_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validates and decodes JWT session token."""
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except Exception:
            return None

    def authenticate_user_by_phone(self, phone: str, plain_password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Authenticates a user by registered phone number and bcrypt password.
        Returns: (success, user_dict_with_token, message)
        """
        clean_phone = phone.strip()
        if not clean_phone or not plain_password:
            return False, None, "Phone number and password are required."

        if self.is_connected and self.client:
            try:
                res = self.client.table("users").select("id, farm_id, phone, password_hash, full_name, role, is_verified").eq("phone", clean_phone).execute()
                if not res.data or len(res.data) == 0:
                    return False, None, "Invalid phone number or password."

                user = res.data[0]
                if not self.verify_password(plain_password, user["password_hash"]):
                    return False, None, "Invalid phone number or password."

                token = self.generate_jwt_token(user["id"], user["farm_id"], user["phone"], user["role"])

                # Update last login timestamp
                try:
                    self.client.table("users").update({"last_login_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}).eq("id", user["id"]).execute()
                except Exception:
                    pass

                user_info = {
                    "user_id": user["id"],
                    "farm_id": user["farm_id"],
                    "phone": user["phone"],
                    "full_name": user.get("full_name", ""),
                    "role": user.get("role", "owner"),
                    "token": token
                }
                return True, user_info, "Login successful."
            except Exception as e:
                print(f"[Auth Error] {e}")

        # Local fallback user check (Demo account)
        demo_phone = "+919876543210"
        demo_pass = "PashuX@2026"
        if clean_phone in (demo_phone, "9876543210") and plain_password == demo_pass:
            token = self.generate_jwt_token("demo-user-id", DEFAULT_FARM_ID, demo_phone, "owner")
            return True, {
                "user_id": "demo-user-id",
                "farm_id": DEFAULT_FARM_ID,
                "phone": demo_phone,
                "full_name": "Ramesh Patel (Farm Owner)",
                "role": "owner",
                "token": token
            }, "Login successful (Demo Mode)."

        return False, None, "Invalid phone number or password."

    def register_user(self, farm_id: str, phone: str, plain_password: str, full_name: str, role: str = "owner") -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Registers a new user with phone number, hashed password, and farm connection.
        """
        clean_phone = phone.strip()
        if len(plain_password) < 6:
            return False, None, "Password must be at least 6 characters."

        pw_hash = self.hash_password(plain_password)
        user_farm = farm_id.strip() if farm_id else DEFAULT_FARM_ID

        if self.is_connected and self.client:
            try:
                # Check if phone already registered
                chk = self.client.table("users").select("id").eq("phone", clean_phone).execute()
                if chk.data and len(chk.data) > 0:
                    return False, None, "A user with this phone number already exists."

                # Verify farm exists, or create default
                try:
                    farm_chk = self.client.table("farms").select("id").eq("id", user_farm).execute()
                    if not farm_chk.data:
                        # Create farm
                        new_farm = self.client.table("farms").insert({
                            "farm_code": f"FARM_{uuid.uuid4().hex[:6].upper()}",
                            "farm_name": f"{full_name}'s Dairy Farm",
                            "owner_name": full_name,
                            "owner_phone": clean_phone
                        }).execute()
                        if new_farm.data:
                            user_farm = new_farm.data[0]["id"]
                except Exception:
                    pass

                new_user_rec = {
                    "farm_id": user_farm,
                    "phone": clean_phone,
                    "password_hash": pw_hash,
                    "full_name": full_name,
                    "role": role,
                    "is_verified": True
                }
                ins = self.client.table("users").insert(new_user_rec).execute()
                if ins.data and len(ins.data) > 0:
                    u = ins.data[0]
                    token = self.generate_jwt_token(u["id"], u["farm_id"], u["phone"], u["role"])
                    return True, {
                        "user_id": u["id"],
                        "farm_id": u["farm_id"],
                        "phone": u["phone"],
                        "full_name": u["full_name"],
                        "role": u["role"],
                        "token": token
                    }, "Registration successful!"
            except Exception as e:
                return False, None, f"Database error during registration: {e}"

        # Local fallback registration
        token = self.generate_jwt_token(str(uuid.uuid4()), user_farm, clean_phone, role)
        return True, {
            "user_id": str(uuid.uuid4()),
            "farm_id": user_farm,
            "phone": clean_phone,
            "full_name": full_name,
            "role": role,
            "token": token
        }, "Registration successful (Local Mode)."

    # -----------------------------------------------------------------
    # Storage Uploads & Signed URL Generation
    # -----------------------------------------------------------------
    def upload_image(self, farm_id: str, activity_id: str, local_image_path: str) -> Optional[str]:
        """
        Uploads captured image to private Supabase Storage bucket.
        Path format: <farm_id>/images/<filename>
        Returns storage path or None.
        """
        if not os.path.exists(local_image_path):
            return None

        img_filename = os.path.basename(local_image_path)
        remote_path = f"{farm_id}/images/{img_filename}"

        if self.is_connected and self.client:
            try:
                with open(local_image_path, "rb") as f:
                    file_bytes = f.read()

                # Upload to Supabase Storage
                res = self.client.storage.from_(IMAGE_BUCKET).upload(
                    path=remote_path,
                    file=file_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"}
                )
                print(f"[Supabase Storage] Uploaded image -> {IMAGE_BUCKET}/{remote_path}")
                return remote_path
            except Exception as e:
                print(f"[Supabase Storage] Image upload notice: {e}")

        # Local path reference fallback
        return remote_path

    def upload_video(self, farm_id: str, activity_id: str, local_video_path: str) -> Optional[str]:
        """
        Uploads captured 10-second video clip to private Supabase Storage bucket.
        Path format: <farm_id>/videos/<filename>
        Returns storage path or None.
        """
        if not os.path.exists(local_video_path):
            return None

        vid_filename = os.path.basename(local_video_path)
        remote_path = f"{farm_id}/videos/{vid_filename}"

        if self.is_connected and self.client:
            try:
                with open(local_video_path, "rb") as f:
                    file_bytes = f.read()

                res = self.client.storage.from_(VIDEO_BUCKET).upload(
                    path=remote_path,
                    file=file_bytes,
                    file_options={"content-type": "video/mp4", "upsert": "true"}
                )
                print(f"[Supabase Storage] Uploaded 10s video -> {VIDEO_BUCKET}/{remote_path}")
                return remote_path
            except Exception as e:
                print(f"[Supabase Storage] Video upload notice: {e}")

        return remote_path

    def upload_cattle_image(self, farm_id: str, cattle_id: str, local_image_path: str) -> Optional[str]:
        """
        Uploads dedicated cattle photograph to Supabase Storage under <farm_id>/cattle/<filename>.
        Returns storage path or fallback.
        """
        if not os.path.exists(local_image_path):
            return None

        img_filename = os.path.basename(local_image_path)
        remote_path = f"{farm_id}/cattle/{img_filename}"

        if self.is_connected and self.client:
            try:
                with open(local_image_path, "rb") as f:
                    file_bytes = f.read()

                self.client.storage.from_(IMAGE_BUCKET).upload(
                    path=remote_path,
                    file=file_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"}
                )
                print(f"[Supabase Storage] Uploaded cattle photo -> {IMAGE_BUCKET}/{remote_path}")
                return remote_path
            except Exception as e:
                print(f"[Supabase Storage] Cattle photo upload notice: {e}")

        return remote_path

    def upload_worker_image(self, farm_id: str, worker_id: str, local_image_path: str) -> Optional[str]:
        """
        Uploads dedicated worker photograph to Supabase Storage under <farm_id>/workers/<filename>.
        Returns storage path or fallback.
        """
        if not os.path.exists(local_image_path):
            return None

        img_filename = os.path.basename(local_image_path)
        remote_path = f"{farm_id}/workers/{img_filename}"

        if self.is_connected and self.client:
            try:
                with open(local_image_path, "rb") as f:
                    file_bytes = f.read()

                self.client.storage.from_(IMAGE_BUCKET).upload(
                    path=remote_path,
                    file=file_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"}
                )
                print(f"[Supabase Storage] Uploaded worker photo -> {IMAGE_BUCKET}/{remote_path}")
                return remote_path
            except Exception as e:
                print(f"[Supabase Storage] Worker photo upload notice: {e}")

        return remote_path

    def get_signed_image_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Generates temporary signed URL for private image."""
        if not storage_path:
            return ""
        if self.is_connected and self.client:
            try:
                res = self.client.storage.from_(IMAGE_BUCKET).create_signed_url(storage_path, expires_in)
                if isinstance(res, dict) and "signedURL" in res:
                    return res["signedURL"]
                elif hasattr(res, "get") and res.get("signedURL"):
                    return res["signedURL"]
                elif isinstance(res, str):
                    return res
            except Exception:
                pass
        # Fallback local URL
        filename = os.path.basename(storage_path)
        return f"/media/images/{filename}"

    def get_signed_video_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Generates temporary signed URL for private 10-second video."""
        if not storage_path:
            return ""
        if self.is_connected and self.client:
            try:
                res = self.client.storage.from_(VIDEO_BUCKET).create_signed_url(storage_path, expires_in)
                if isinstance(res, dict) and "signedURL" in res:
                    return res["signedURL"]
                elif hasattr(res, "get") and res.get("signedURL"):
                    return res["signedURL"]
                elif isinstance(res, str):
                    return res
            except Exception:
                pass
    def _ensure_farm_exists(self, farm_id: str, farm_name: str = "PashuX Registered Dairy Farm"):
        """Ensures farm record exists in Supabase to prevent foreign key errors."""
        if not self.is_connected or not self.client:
            return
        try:
            res = self.client.table("farms").select("id").eq("id", farm_id).execute()
            if not res.data or len(res.data) == 0:
                self.client.table("farms").insert({
                    "id": farm_id,
                    "farm_name": farm_name,
                    "farm_code": f"FARM-{farm_id[:6].upper()}",
                    "owner_phone": "+919876543210",
                    "location": "India"
                }).execute()
                print(f"[Supabase DB] Farm registered: {farm_id}")
        except Exception as e:
            print(f"[Supabase DB] Note ensuring farm: {e}")

    def _ensure_camera_exists(self, farm_id: str, camera_id: str = "CAM01"):
        """Ensures camera record exists in Supabase to prevent foreign key errors."""
        if not self.is_connected or not self.client:
            return
        try:
            self._ensure_farm_exists(farm_id)
            res = self.client.table("cameras").select("id").eq("id", camera_id).execute()
            if not res.data or len(res.data) == 0:
                self.client.table("cameras").insert({
                    "id": camera_id,
                    "farm_id": farm_id,
                    "camera_name": f"Camera {camera_id}",
                    "location_description": "Main Corral & Feed Area",
                    "is_active": True
                }).execute()
                print(f"[Supabase DB] Camera registered: {camera_id}")
        except Exception as e:
            print(f"[Supabase DB] Note ensuring camera: {e}")

    # -----------------------------------------------------------------
    # Farm Activity Records Database Operations
    # -----------------------------------------------------------------
    def record_activity(self, activity_data: Dict[str, Any]) -> bool:
        """
        Saves structured activity record into Supabase PostgreSQL and local cache.
        """
        # Ensure UUID and required keys
        if "activity_id" not in activity_data:
            activity_data["activity_id"] = str(uuid.uuid4())
        if "farm_id" not in activity_data:
            activity_data["farm_id"] = DEFAULT_FARM_ID
        if "detection_date" not in activity_data:
            activity_data["detection_date"] = datetime.date.today().isoformat()
        if "created_at" not in activity_data:
            activity_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        farm_id = str(activity_data["farm_id"])
        camera_id = str(activity_data.get("camera_id", DEFAULT_CAMERA_ID))

        # 1. Save to local cache first (ensures zero data loss)
        activity_data["_synced_to_supabase"] = False
        self._save_to_local_cache(activity_data)

        # 2. Upload to Supabase PostgreSQL
        if self.is_connected and self.client:
            try:
                self._ensure_farm_exists(farm_id)
                self._ensure_camera_exists(farm_id, camera_id)
                # Prepare record matching public.farm_activities table schema
                ai_res = dict(activity_data.get("ai_detection_result") or {})
                ai_res.update({
                    "breed": str(activity_data.get("breed", "Cattle")),
                    "disease_status": str(activity_data.get("disease_status", "Healthy")),
                    "bcs_score": float(activity_data["bcs_score"]) if (activity_data.get("bcs_score") is not None and float(activity_data.get("bcs_score") or 0) > 0) else None,
                    "estimated_weight_kg": float(activity_data.get("estimated_weight_kg", 400.0)),
                    "estimated_height_cm": float(activity_data.get("estimated_height_cm", 130.0)),
                    "eating_duration_sec": float(activity_data.get("eating_duration_sec", activity_data.get("activity_duration", 10.0) if "feeding" in str(activity_data.get("activity_type")) else 0.0)),
                    "resting_duration_sec": float(activity_data.get("resting_duration_sec", activity_data.get("activity_duration", 10.0) if "resting" in str(activity_data.get("activity_type")) else 0.0)),
                    "herd_count": int(activity_data.get("herd_count", 1))
                })

                db_record = {
                    "activity_id": str(activity_data["activity_id"]),
                    "farm_id": str(activity_data["farm_id"]),
                    "camera_id": str(activity_data.get("camera_id", DEFAULT_CAMERA_ID)),
                    "cattle_id": activity_data.get("cattle_id"),
                    "worker_id": activity_data.get("worker_id"),
                    "activity_type": str(activity_data.get("activity_type", "farm_activity")),
                    "detection_date": str(activity_data.get("detection_date", datetime.date.today().isoformat())),
                    "start_time": str(activity_data.get("start_time", datetime.datetime.now(datetime.timezone.utc).isoformat())),
                    "end_time": str(activity_data.get("end_time", datetime.datetime.now(datetime.timezone.utc).isoformat())),
                    "activity_duration": float(activity_data.get("activity_duration", 10.0)),
                    "eating_duration_sec": float(activity_data.get("eating_duration_sec", activity_data.get("activity_duration", 10.0) if "feeding" in str(activity_data.get("activity_type")) else 0.0)),
                    "resting_duration_sec": float(activity_data.get("resting_duration_sec", activity_data.get("activity_duration", 10.0) if "resting" in str(activity_data.get("activity_type")) else 0.0)),
                    "herd_count": int(activity_data.get("herd_count", 1)),
                    "breed": str(activity_data.get("breed", "Cattle")),
                    "disease_status": str(activity_data.get("disease_status", "Healthy")),
                    "bcs_score": float(activity_data["bcs_score"]) if (activity_data.get("bcs_score") is not None and float(activity_data.get("bcs_score") or 0) > 0) else None,
                    "estimated_weight_kg": float(activity_data.get("estimated_weight_kg", 400.0)),
                    "estimated_height_cm": float(activity_data.get("estimated_height_cm", 130.0)),
                    "ai_detection_result": ai_res,
                    "confidence_score": float(activity_data.get("confidence_score", 0.85)),
                    "description": str(activity_data.get("description", "")),
                    "captured_image_storage_path": str(activity_data.get("captured_image_storage_path") or f"{activity_data.get('farm_id')}/images/{activity_data.get('activity_id')}.jpg"),
                    "captured_video_storage_path": str(activity_data.get("captured_video_storage_path") or f"{activity_data.get('farm_id')}/videos/{activity_data.get('activity_id')}.mp4"),
                    "video_source": str(activity_data.get("video_source", "uploaded")),
                    "status": str(activity_data.get("status", "detected_sign")) if str(activity_data.get("status", "")) in ("suspected_sign", "detected_sign", "verified", "dismissed") else "detected_sign",
                    "created_at": str(activity_data.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()))
                }

                try:
                    res = self.client.table("farm_activities").insert(db_record).execute()
                except Exception as ex:
                    # Fallback to base columns if clean columns haven't been added yet via add_clean_cattle_columns.sql
                    err_msg = str(ex)
                    if "PGRST204" in err_msg or "column" in err_msg.lower():
                        base_keys = ["activity_id", "farm_id", "camera_id", "cattle_id", "worker_id", "activity_type",
                                     "detection_date", "start_time", "end_time", "activity_duration", "ai_detection_result",
                                     "confidence_score", "description", "captured_image_storage_path",
                                     "captured_video_storage_path", "video_source", "status", "created_at"]
                        fallback_rec = {k: db_record[k] for k in base_keys if k in db_record}
                        res = self.client.table("farm_activities").insert(fallback_rec).execute()
                    else:
                        raise ex

                # Record sync state locally
                activity_data["_synced_to_supabase"] = True
                self._save_to_local_cache(activity_data)
                print(f"[Supabase DB] Activity saved: {db_record['activity_type']} ({db_record['activity_id']})")

                # Also automatically persist cattle profile into public.cattle table under farm_id
                if activity_data.get("cattle_id"):
                    try:
                        # Do NOT overwrite cattle's dedicated portrait photo with wide-scene activity snapshots
                        c_img = activity_data.get("cattle_image_url") or ""
                        c_storage = activity_data.get("cattle_image_storage_path") or ""
                        if not c_img and not ("act_" in str(activity_data.get("captured_image_storage_path", ""))):
                            c_img = activity_data.get("captured_image_storage_path", "")
                            c_storage = activity_data.get("captured_image_storage_path", "")

                        self.upsert_cattle(farm_id, {
                            "cattle_id": activity_data["cattle_id"],
                            "camera_id": camera_id,
                            "breed": activity_data.get("breed", "Cattle"),
                            "disease_status": activity_data.get("disease_status", "Healthy"),
                            "bcs_score": activity_data.get("bcs_score"),
                            "estimated_weight_kg": activity_data.get("estimated_weight_kg", 400.0),
                            "estimated_height_cm": activity_data.get("estimated_height_cm", 130.0),
                            "image_url": c_img,
                            "captured_image_storage_path": c_storage,
                            "last_seen": activity_data.get("end_time") or activity_data.get("created_at")
                        })
                    except Exception:
                        pass

                # Also persist worker into public.workers table if worker_id present
                if activity_data.get("worker_id"):
                    try:
                        self.upsert_worker(farm_id, {
                            "worker_id": activity_data["worker_id"],
                            "camera_id": camera_id,
                            "image_url": activity_data.get("captured_image_storage_path", ""),
                            "captured_image_storage_path": activity_data.get("captured_image_storage_path", ""),
                            "last_seen": activity_data.get("end_time") or activity_data.get("created_at")
                        })
                    except Exception:
                        pass

                return True
            except Exception as e:
                print(f"[Supabase DB] Error inserting activity: {e}")
                return False

        return True

    def update_activity(self, *args, **kwargs) -> bool:
        """
        Updates an ongoing activity record (e.g. extending eating duration or resting duration)
        in both Supabase and the local offline cache.
        Supports:
        - update_activity(farm_id, activity_id, updates)
        - update_activity(activity_id, updates)
        - update_activity(activity_id=..., updates=..., farm_id=...)
        """
        farm_id = kwargs.get("farm_id")
        activity_id = kwargs.get("activity_id")
        updates = kwargs.get("updates", {})

        if len(args) == 3:
            farm_id, activity_id, updates = args
        elif len(args) == 2:
            activity_id, updates = args
        elif len(args) == 1 and not activity_id:
            activity_id = args[0]

        if not activity_id:
            return False

        # 1. Update local cache
        try:
            if LOCAL_DB_FILE.exists():
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    records = json.load(f)
                updated = False
                for r in records:
                    if str(r.get("activity_id")) == str(activity_id):
                        r.update(updates)
                        updated = True
                        break
                if updated:
                    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2, default=str)
        except Exception as e:
            print(f"[Local Cache Update Error] {e}")

        # 2. Update Supabase
        if self.is_connected and self.client:
            try:
                valid_fields = [
                    "end_time", "activity_duration", "eating_duration_sec", "resting_duration_sec",
                    "herd_count", "breed", "disease_status", "bcs_score", "estimated_weight_kg",
                    "estimated_height_cm", "ai_detection_result", "status", "description"
                ]
                db_updates = {k: v for k, v in updates.items() if k in valid_fields}
                if not db_updates:
                    return True

                if "status" in db_updates:
                    st_val = str(db_updates["status"])
                    if st_val not in ("suspected_sign", "detected_sign", "verified", "dismissed"):
                        db_updates["status"] = "verified" if "verified" in st_val else "detected_sign"

                # Also ensure ai_detection_result has the latest subfields
                if "ai_detection_result" not in db_updates:
                    db_updates["ai_detection_result"] = {
                        "breed": updates.get("breed", "Cattle"),
                        "disease_status": updates.get("disease_status", "Healthy"),
                        "bcs_score": float(updates["bcs_score"]) if (updates.get("bcs_score") is not None and float(updates.get("bcs_score") or 0) > 0) else None,
                        "estimated_weight_kg": float(updates.get("estimated_weight_kg", 400.0)),
                        "estimated_height_cm": float(updates.get("estimated_height_cm", 130.0)),
                        "eating_duration_sec": float(updates.get("eating_duration_sec", 0.0)),
                        "resting_duration_sec": float(updates.get("resting_duration_sec", 0.0)),
                        "herd_count": int(updates.get("herd_count", 1))
                    }

                try:
                    q = self.client.table("farm_activities").update(db_updates).eq("activity_id", activity_id)
                    if farm_id:
                        q = q.eq("farm_id", farm_id)
                    q.execute()
                except Exception as ex:
                    err_msg = str(ex)
                    if "PGRST204" in err_msg or "column" in err_msg.lower():
                        base_keys = ["end_time", "activity_duration", "ai_detection_result", "status", "description"]
                        fallback_updates = {k: v for k, v in db_updates.items() if k in base_keys}
                        if fallback_updates:
                            q2 = self.client.table("farm_activities").update(fallback_updates).eq("activity_id", activity_id)
                            if farm_id:
                                q2 = q2.eq("farm_id", farm_id)
                    else:
                        raise ex
                return True
            except Exception as e:
                print(f"[Supabase DB Update Error] {e}")
                return False
        return True

    # -----------------------------------------------------------------
    # Cattle Registry Database Operations (public.cattle table)
    # -----------------------------------------------------------------
    def upsert_cattle(self, farm_id: str, cattle_data: Dict[str, Any]) -> bool:
        """
        Stores/upserts cattle profile into Supabase 'cattle' table and local cache.
        Ensures cattle data is strictly organized under farm_id and cattle_id.
        """
        cid = cattle_data.get("cattle_id")
        if not cid:
            return False

        farm_id = str(farm_id or cattle_data.get("farm_id") or DEFAULT_FARM_ID)
        self._ensure_farm_exists(farm_id)

        # Store data directly in Supabase; fallback to offline file only when disconnected
        if not self.is_connected or not self.client:
            try:
                CATTLE_ROSTER_FILE.parent.mkdir(parents=True, exist_ok=True)
                roster = {}
                if CATTLE_ROSTER_FILE.exists() and CATTLE_ROSTER_FILE.stat().st_size > 2:
                    try:
                        with open(CATTLE_ROSTER_FILE, "r", encoding="utf-8") as f:
                            roster = json.load(f)
                    except Exception:
                        roster = {}
                roster[cid] = {
                    "cattle_id": cid,
                    "farm_id": farm_id,
                    "camera_id": cattle_data.get("camera_id", DEFAULT_CAMERA_ID),
                    "breed": cattle_data.get("breed", "Cattle"),
                    "disease_status": cattle_data.get("disease_status", "Healthy"),
                    "bcs_score": float(cattle_data["bcs_score"]) if (cattle_data.get("bcs_score") is not None and float(cattle_data.get("bcs_score") or 0) > 0) else None,
                    "estimated_weight_kg": float(cattle_data.get("estimated_weight_kg", 400.0)),
                    "estimated_height_cm": float(cattle_data.get("estimated_height_cm", 130.0)),
                    "image_url": cattle_data.get("image_url", f"/media/images/{cid}.jpg"),
                    "captured_image_storage_path": cattle_data.get("captured_image_storage_path", ""),
                    "last_seen": cattle_data.get("last_seen", datetime.datetime.now(datetime.timezone.utc).isoformat())
                }
                with open(CATTLE_ROSTER_FILE, "w", encoding="utf-8") as f:
                    json.dump(roster, f, indent=2)
            except Exception as e:
                print(f"[Supabase Cattle Cache Notice] {e}")

        # 2. Upload / Upsert to Supabase PostgreSQL table 'cattle'
        if self.is_connected and self.client:
            try:
                img_url = str(cattle_data.get("image_url", ""))
                storage_p = str(cattle_data.get("captured_image_storage_path", ""))

                # If new image is empty or an activity snapshot, preserve existing dedicated portrait photo
                if not img_url or "act_" in img_url or "act_" in storage_p:
                    try:
                        ex_row = self.client.table("cattle").select("image_url, captured_image_storage_path").eq("farm_id", farm_id).eq("cattle_id", cid).execute()
                        if ex_row.data and len(ex_row.data) > 0:
                            ex_img = ex_row.data[0].get("image_url")
                            ex_sp = ex_row.data[0].get("captured_image_storage_path")
                            if ex_img and "act_" not in ex_img:
                                img_url = ex_img
                            if ex_sp and "act_" not in ex_sp:
                                storage_p = ex_sp
                    except Exception:
                        pass

                db_record = {
                    "farm_id": farm_id,
                    "cattle_id": cid,
                    "camera_id": str(cattle_data.get("camera_id", DEFAULT_CAMERA_ID)),
                    "breed": str(cattle_data.get("breed", "Cattle")),
                    "disease_status": str(cattle_data.get("disease_status", "Healthy")),
                    "bcs_score": float(cattle_data["bcs_score"]) if (cattle_data.get("bcs_score") is not None and float(cattle_data.get("bcs_score") or 0) > 0) else None,
                    "estimated_weight_kg": float(cattle_data.get("estimated_weight_kg", 400.0)),
                    "estimated_height_cm": float(cattle_data.get("estimated_height_cm", 130.0)),
                    "image_url": img_url,
                    "captured_image_storage_path": storage_p,
                    "last_seen": str(cattle_data.get("last_seen", datetime.datetime.now(datetime.timezone.utc).isoformat())),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                self.client.table("cattle").upsert(db_record, on_conflict="farm_id,cattle_id").execute()
                print(f"[Supabase Cattle] Cattle {cid} upserted under Farm {farm_id}")
                return True
            except Exception as e:
                err_str = str(e)
                if "PGRST205" in err_str or "does not exist" in err_str:
                    print(f"[Supabase Cattle Notice] Table 'public.cattle' not yet created in Supabase. Run Supabase/create_cattle_table.sql in SQL editor to enable cloud sync.")
                else:
                    print(f"[Supabase Cattle Upsert Notice] {e}")
                return False

        return True

    def fetch_farm_cattle(self, farm_id: str) -> List[Dict[str, Any]]:
        """
        Fetches all registered cattle for the given farm_id from Supabase 'cattle' table.
        Falls back to local cattle roster cache if table is not yet created or unreachable.
        """
        farm_id = str(farm_id or DEFAULT_FARM_ID)
        cattle_list = []

        if self.is_connected and self.client:
            try:
                res = self.client.table("cattle").select("*").eq("farm_id", farm_id).order("cattle_id").execute()
                if res.data:
                    for row in res.data:
                        sp = row.get("captured_image_storage_path")
                        if sp and not str(row.get("image_url", "")).startswith("http"):
                            signed_url = self.get_signed_image_url(sp)
                            if signed_url:
                                row["image_url"] = signed_url
                        cattle_list.append(row)
                    return cattle_list
            except Exception as e:
                err_str = str(e)
                if "PGRST205" not in err_str:
                    print(f"[Supabase fetch cattle notice] {e}")

        # Local fallback
        if CATTLE_ROSTER_FILE.exists():
            try:
                with open(CATTLE_ROSTER_FILE, "r", encoding="utf-8") as f:
                    local_roster = json.load(f)
                for cid, cdata in local_roster.items():
                    if str(cdata.get("farm_id", farm_id)) == farm_id:
                        cattle_list.append(cdata)
            except Exception:
                pass

        return cattle_list

    # -----------------------------------------------------------------
    # Worker Registry Database Operations (public.workers table)
    # -----------------------------------------------------------------
    def upsert_worker(self, farm_id: str, worker_data: Dict[str, Any]) -> bool:
        """Stores/upserts worker profile into Supabase 'workers' table and local cache."""
        wid = worker_data.get("worker_id")
        if not wid:
            return False

        farm_id = str(farm_id or worker_data.get("farm_id") or DEFAULT_FARM_ID)
        self._ensure_farm_exists(farm_id)

        # Store worker directly in Supabase; fallback to offline file only when disconnected
        if not self.is_connected or not self.client:
            try:
                WORKER_ROSTER_FILE.parent.mkdir(parents=True, exist_ok=True)
                w_roster = {}
                if WORKER_ROSTER_FILE.exists() and WORKER_ROSTER_FILE.stat().st_size > 2:
                    try:
                        with open(WORKER_ROSTER_FILE, "r", encoding="utf-8") as f:
                            w_roster = json.load(f)
                    except Exception:
                        w_roster = {}
                w_roster[wid] = {
                    "worker_id": wid,
                    "farm_id": farm_id,
                    "camera_id": worker_data.get("camera_id", DEFAULT_CAMERA_ID),
                    "image_url": worker_data.get("image_url", f"/media/images/{wid}.jpg"),
                    "captured_image_storage_path": worker_data.get("captured_image_storage_path", ""),
                    "last_seen": worker_data.get("last_seen", datetime.datetime.now(datetime.timezone.utc).isoformat())
                }
                with open(WORKER_ROSTER_FILE, "w", encoding="utf-8") as f:
                    json.dump(w_roster, f, indent=2)
            except Exception as e:
                print(f"[Supabase Worker Cache Notice] {e}")

        # 2. Upload / Upsert to Supabase PostgreSQL table 'workers'
        if self.is_connected and self.client:
            try:
                db_record = {
                    "farm_id": farm_id,
                    "worker_id": wid,
                    "camera_id": str(worker_data.get("camera_id", DEFAULT_CAMERA_ID)),
                    "image_url": str(worker_data.get("image_url", "")),
                    "captured_image_storage_path": str(worker_data.get("captured_image_storage_path", "")),
                    "last_seen": str(worker_data.get("last_seen", datetime.datetime.now(datetime.timezone.utc).isoformat())),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                self.client.table("workers").upsert(db_record, on_conflict="farm_id,worker_id").execute()
                print(f"[Supabase Workers] Worker {wid} upserted under Farm {farm_id}")
                return True
            except Exception as e:
                err_str = str(e)
                if "PGRST205" in err_str or "does not exist" in err_str:
                    print(f"[Supabase Workers Notice] Table 'public.workers' not yet created in Supabase. Run Supabase/create_cattle_table.sql in SQL editor to enable cloud sync.")
                else:
                    print(f"[Supabase Worker Upsert Notice] {e}")
                return False

        return True

    def fetch_farm_workers(self, farm_id: str) -> List[Dict[str, Any]]:
        """Fetches all registered workers for the farm from Supabase or local cache."""
        farm_id = str(farm_id or DEFAULT_FARM_ID)
        workers_list = []

        if self.is_connected and self.client:
            try:
                res = self.client.table("workers").select("*").eq("farm_id", farm_id).order("worker_id").execute()
                if res.data:
                    for row in res.data:
                        sp = row.get("captured_image_storage_path")
                        if sp and not str(row.get("image_url", "")).startswith("http"):
                            signed_url = self.get_signed_image_url(sp)
                            if signed_url:
                                row["image_url"] = signed_url
                        workers_list.append(row)
                    return workers_list
            except Exception as e:
                err_str = str(e)
                if "PGRST205" not in err_str:
                    print(f"[Supabase fetch workers notice] {e}")

        # Local fallback
        if WORKER_ROSTER_FILE.exists():
            try:
                with open(WORKER_ROSTER_FILE, "r", encoding="utf-8") as f:
                    local_w_roster = json.load(f)
                for wid, wdata in local_w_roster.items():
                    if str(wdata.get("farm_id", farm_id)) == farm_id:
                        workers_list.append(wdata)
            except Exception:
                pass

        return workers_list

    def delete_activity(self, farm_id_or_activity_id: Optional[str] = None, activity_id: Optional[str] = None) -> bool:
        """Deletes an activity record from Supabase and local cache."""
        if activity_id is None:
            activity_id = str(farm_id_or_activity_id)
            farm_id = None
        else:
            farm_id = farm_id_or_activity_id

        try:
            if LOCAL_DB_FILE.exists():
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    records = json.load(f)
                records = [r for r in records if str(r.get("activity_id")) != str(activity_id)]
                with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, default=str)
        except Exception as e:
            print(f"[Supabase delete local error] {e}")

        if self.is_connected and self.client:
            try:
                q = self.client.table("farm_activities").delete().eq("activity_id", activity_id)
                if farm_id:
                    q = q.eq("farm_id", farm_id)
                q.execute()
                return True
            except Exception as e:
                print(f"[Supabase delete remote error] {e}")
                return False
        return True

    def clean_duplicate_activities(self, farm_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Consolidates duplicate records for the same cattle/activity within cooldown windows,
        and eliminates spurious false estrus records caused by generic cow detections.
        """
        farm_id = farm_id or DEFAULT_FARM_ID
        deleted_count = 0
        deleted_ids = []

        try:
            if LOCAL_DB_FILE.exists():
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    records = json.load(f)

                cleaned = []
                seen_estrus_ts = 0.0
                seen_sessions = set()

                for r in records:
                    act_type = str(r.get("activity_type", ""))
                    desc = str(r.get("description", ""))
                    ai_res = r.get("ai_detection_result") or {}
                    aid = str(r.get("activity_id"))

                    # Eliminate false estrus (e.g. single grazing cow classified as 'cow' mounting another)
                    if "estrus" in act_type and (ai_res.get("behavior") == "cow" or "(cow)" in desc):
                        deleted_ids.append(aid)
                        continue

                    # Rate-limit estrus events so at most 1 event per 5 minutes exists
                    if "estrus" in act_type:
                        try:
                            created_ts = datetime.datetime.fromisoformat(r.get("created_at", "").replace("Z", "+00:00")).timestamp()
                        except Exception:
                            created_ts = 0.0
                        if abs(created_ts - seen_estrus_ts) < 300.0 and seen_estrus_ts > 0:
                            deleted_ids.append(aid)
                            continue
                        seen_estrus_ts = created_ts

                    # Consolidate duplicate feeding/resting sessions for same cattle & date
                    if "feeding" in act_type:
                        dur_val = float(r.get("activity_duration") or 0.0)
                        # Remove false-positive feeding activities created when worker was milking
                        has_milking = any("milking" in str(rec.get("activity_type", "")) for rec in records)
                        if has_milking and dur_val <= 15.0:
                            deleted_ids.append(aid)
                            continue

                        cid = str(r.get("cattle_id", ""))
                        date_str = str(r.get("detection_date", ""))
                        sess_key = f"{cid}_{act_type}_{date_str}"
                        if sess_key in seen_sessions:
                            prev_item = next((item for item in cleaned if f"{item.get('cattle_id')}_{item.get('activity_type')}_{item.get('detection_date')}" == sess_key), None)
                            if prev_item:
                                prev_dur = float(prev_item.get("activity_duration") or 0.0)
                                cur_dur = float(r.get("activity_duration") or 0.0)
                                if cur_dur > prev_dur:
                                    cleaned.remove(prev_item)
                                    deleted_ids.append(str(prev_item.get("activity_id")))
                                    cleaned.append(r)
                                else:
                                    deleted_ids.append(aid)
                            else:
                                deleted_ids.append(aid)
                            continue
                        seen_sessions.add(sess_key)
                    elif "resting" in act_type:
                        cid = str(r.get("cattle_id", ""))
                        date_str = str(r.get("detection_date", ""))
                        sess_key = f"{cid}_{act_type}_{date_str}"
                        if sess_key in seen_sessions:
                            deleted_ids.append(aid)
                            continue
                        seen_sessions.add(sess_key)
                    elif "milking" in act_type:
                        date_str = str(r.get("detection_date", ""))
                        wid = str(r.get("worker_id") or "WRK001")
                        # Keep the longest / most verified milking session for this worker on this date
                        sess_key = f"{wid}_milking_{date_str}"
                        if sess_key in seen_sessions:
                            prev_item = next((item for item in cleaned if f"{item.get('worker_id', 'WRK001')}_milking_{item.get('detection_date')}" == sess_key), None)
                            if prev_item:
                                prev_dur = float(prev_item.get("activity_duration") or 0.0)
                                cur_dur = float(r.get("activity_duration") or 0.0)
                                if cur_dur > prev_dur:
                                    cleaned.remove(prev_item)
                                    deleted_ids.append(str(prev_item.get("activity_id")))
                                    cleaned.append(r)
                                else:
                                    deleted_ids.append(aid)
                            else:
                                deleted_ids.append(aid)
                            continue
                        seen_sessions.add(sess_key)
                    elif "worker_feeding" in act_type:
                        date_str = str(r.get("detection_date", ""))
                        wid = str(r.get("worker_id") or "WRK001")
                        sess_key = f"{wid}_worker_feeding_{date_str}"
                        if sess_key in seen_sessions:
                            deleted_ids.append(aid)
                            continue
                        seen_sessions.add(sess_key)

                    cleaned.append(r)

                with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, indent=2, default=str)
                deleted_count = len(deleted_ids)
        except Exception as e:
            print(f"[Clean Duplicates Local Error] {e}")

        if self.is_connected and self.client and deleted_ids:
            try:
                for chunk_idx in range(0, len(deleted_ids), 30):
                    chunk = deleted_ids[chunk_idx:chunk_idx + 30]
                    self.client.table("farm_activities").delete().in_("activity_id", chunk).execute()
            except Exception as e:
                print(f"[Clean Duplicates Supabase Error] {e}")

        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deduplication complete: cleaned {deleted_count} duplicate and false entries."
        }

    def clear_all_activities(self, farm_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Permanently wipes all cattle, farm activities, and worker records from Supabase
        and cleans any local caches so all data is strictly fresh.
        """
        target_farm = str(farm_id or DEFAULT_FARM_ID)

        # 1. Clear Supabase Cloud Database Tables (authoritative store)
        if self.is_connected and self.client:
            try:
                # Wipes cattle table
                q_c = self.client.table("cattle").delete()
                if farm_id and farm_id != "all":
                    q_c = q_c.eq("farm_id", target_farm)
                else:
                    q_c = q_c.neq("cattle_id", "___NONE___")
                q_c.execute()
                print(f"[Supabase DB] Wiped all cattle for farm {target_farm}")
            except Exception as e:
                print(f"[Supabase Cattle Delete Notice] {e}")

            try:
                # Wipes farm_activities table
                q_a = self.client.table("farm_activities").delete()
                if farm_id and farm_id != "all":
                    q_a = q_a.eq("farm_id", target_farm)
                else:
                    q_a = q_a.neq("activity_id", "00000000-0000-0000-0000-000000000000")
                q_a.execute()
                print(f"[Supabase DB] Wiped all farm_activities for farm {target_farm}")
            except Exception as e:
                print(f"[Supabase Activities Delete Notice] {e}")

            try:
                # Wipes workers table
                q_w = self.client.table("workers").delete()
                if farm_id and farm_id != "all":
                    q_w = q_w.eq("farm_id", target_farm)
                else:
                    q_w = q_w.neq("worker_id", "___NONE___")
                q_w.execute()
                print(f"[Supabase DB] Wiped all workers for farm {target_farm}")
            except Exception as e:
                print(f"[Supabase Workers Delete Notice] {e}")

        # 2. Reset local json files so no stale data lingers
        try:
            if LOCAL_DB_FILE.exists():
                with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f)
        except Exception:
            pass

        try:
            if CATTLE_ROSTER_FILE.exists():
                with open(CATTLE_ROSTER_FILE, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        except Exception:
            pass

        try:
            if WORKER_ROSTER_FILE.exists():
                with open(WORKER_ROSTER_FILE, "w", encoding="utf-8") as f:
                    json.dump({}, f)
        except Exception:
            pass

        return {
            "success": True,
            "message": "All cattle, farm activities, and worker records permanently cleared from Supabase."
        }

    def _save_to_local_cache(self, activity_data: Dict[str, Any]):
        """Appends activity record to local JSON file for persistence."""
        try:
            records = []
            if LOCAL_DB_FILE.exists():
                try:
                    with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                        records = json.load(f)
                except Exception:
                    records = []

            # Avoid duplicates in cache
            aid = str(activity_data.get("activity_id"))
            records = [r for r in records if str(r.get("activity_id")) != aid]
            records.insert(0, activity_data)

            # Keep latest 500 records locally
            records = records[:500]

            with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, default=str)
        except Exception as e:
            print(f"[Local Cache] Warning saving local activity: {e}")

    def fetch_activities(
        self,
        farm_id: str,
        date_filter: Optional[str] = None,
        camera_id: Optional[str] = None,
        cattle_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves farm activities directly from Supabase (authoritative source).
        Prunes locally cached records if they were deleted directly in Supabase.
        """
        results = []
        supabase_query_succeeded = False

        if self.is_connected and self.client:
            try:
                query = self.client.table("farm_activities").select("*").eq("farm_id", farm_id)

                if date_filter:
                    query = query.eq("detection_date", date_filter)
                if camera_id and camera_id != "all":
                    query = query.eq("camera_id", camera_id)
                if cattle_id and cattle_id != "all":
                    query = query.ilike("cattle_id", f"%{cattle_id}%")
                if activity_type and activity_type != "all":
                    query = query.eq("activity_type", activity_type)

                query = query.order("created_at", desc=True).limit(limit)
                res = query.execute()
                results = res.data or []
                supabase_query_succeeded = True

                # Synchronize and prune local cache so records deleted in Supabase are not kept
                if LOCAL_DB_FILE.exists() and not date_filter and camera_id in (None, "all") and cattle_id in (None, "all") and activity_type in (None, "all"):
                    try:
                        with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                            local_records = json.load(f)
                        live_ids = {str(r.get("activity_id")) for r in results}
                        pruned_local = [
                            r for r in local_records
                            if str(r.get("farm_id")) != str(farm_id) or str(r.get("activity_id")) in live_ids
                        ]
                        if len(pruned_local) != len(local_records):
                            with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                                json.dump(pruned_local, f, indent=2, default=str)
                    except Exception:
                        pass

            except Exception as e:
                now_t = time.time()
                if not hasattr(self, "_last_supabase_err_ts") or (now_t - self._last_supabase_err_ts >= 60.0):
                    self._last_supabase_err_ts = now_t
                    print(f"[Supabase Notice] Remote query unreachable ({e}). Using local offline storage cache.")
                supabase_query_succeeded = False

        # Only fall back to local cache if Supabase is NOT connected or connection/query failed
        if not supabase_query_succeeded and LOCAL_DB_FILE.exists():
            try:
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    local_records = json.load(f)

                filtered = []
                for r in local_records:
                    if str(r.get("farm_id")) != str(farm_id) and farm_id != DEFAULT_FARM_ID:
                        continue
                    if date_filter and str(r.get("detection_date")) != str(date_filter):
                        continue
                    if camera_id and camera_id != "all" and str(r.get("camera_id")) != str(camera_id):
                        continue
                    if cattle_id and cattle_id != "all" and cattle_id.lower() not in str(r.get("cattle_id", "")).lower():
                        continue
                    if activity_type and activity_type != "all" and str(r.get("activity_type")) != str(activity_type):
                        continue
                    filtered.append(r)
                results = filtered[:limit]
            except Exception as e:
                print(f"[Local Read Error] {e}")

        # Augment with temporary signed URLs for private images and videos, and ensure clean metrics
        for item in results:
            img_path = item.get("captured_image_storage_path", "")
            vid_path = item.get("captured_video_storage_path", "")

            item["image_url"] = self.get_signed_image_url(img_path)
            item["video_url"] = self.get_signed_video_url(vid_path)

            ai_res = item.get("ai_detection_result") or {}
            if not item.get("breed") or item.get("breed") == "Cattle":
                item["breed"] = ai_res.get("breed", item.get("breed", "Cattle"))
            if not item.get("disease_status"):
                item["disease_status"] = ai_res.get("disease_status", "Healthy")
            if item.get("bcs_score") is None:
                item["bcs_score"] = ai_res.get("bcs_score")
            if item.get("estimated_weight_kg") is None:
                item["estimated_weight_kg"] = ai_res.get("estimated_weight_kg", 400.0)
            if item.get("estimated_height_cm") is None:
                item["estimated_height_cm"] = ai_res.get("estimated_height_cm", 130.0)
            if not item.get("eating_duration_sec"):
                item["eating_duration_sec"] = ai_res.get("eating_duration_sec", item.get("activity_duration", 0.0) if "feeding" in str(item.get("activity_type")) else 0.0)
            if not item.get("resting_duration_sec"):
                item["resting_duration_sec"] = ai_res.get("resting_duration_sec", item.get("activity_duration", 0.0) if "resting" in str(item.get("activity_type")) else 0.0)
            if not item.get("herd_count"):
                item["herd_count"] = ai_res.get("herd_count", 1)

        return results

    def delete_activity(self, farm_id_or_activity_id: Optional[str] = None, activity_id: Optional[str] = None) -> bool:
        """Deletes an activity record directly from Supabase and local cache."""
        if activity_id is None:
            activity_id = str(farm_id_or_activity_id)
            farm_id = None
        else:
            farm_id = farm_id_or_activity_id

        success = True
        if self.is_connected and self.client:
            try:
                # Primary delete by activity_id ensures record is removed even across farm variations
                self.client.table("farm_activities").delete().eq("activity_id", activity_id).execute()
                print(f"[Supabase DB] Activity deleted: {activity_id}")
            except Exception as e:
                print(f"[Supabase DB] Error deleting activity: {e}")
                success = False

        if LOCAL_DB_FILE.exists():
            try:
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    records = json.load(f)
                records = [r for r in records if str(r.get("activity_id")) != str(activity_id)]
                with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, default=str)
                print(f"[Local Cache] Activity deleted: {activity_id}")
            except Exception as e:
                print(f"[Local Cache] Error deleting activity: {e}")
        return success

    def sync_local_to_supabase(self) -> Dict[str, Any]:
        """
        Synchronizes all local activities, images, and videos from local cache to live Supabase.
        """
        if not self.is_connected or not self.client:
            return {
                "success": False,
                "synced_count": 0,
                "message": "Supabase not connected. Please enter SUPABASE_URL and SUPABASE_KEY in .env"
            }

        if not LOCAL_DB_FILE.exists():
            return {"success": True, "synced_count": 0, "message": "No local records to sync."}

        try:
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            return {"success": False, "synced_count": 0, "message": f"Could not read local cache: {e}"}

        synced = 0
        modified = False
        for rec in records:
            # Never resurrect or re-upload records that were already synchronized to Supabase
            if rec.get("_synced_to_supabase") is True:
                continue

            farm_id = str(rec.get("farm_id", DEFAULT_FARM_ID))
            act_id = str(rec.get("activity_id"))

            # Upload local image if available
            local_img = rec.get("local_snapshot_path")
            if local_img and os.path.exists(local_img):
                remote_img = self.upload_image(farm_id, act_id, local_img)
                if remote_img:
                    rec["captured_image_storage_path"] = remote_img

            # Upload local video if available
            local_vid = LOCAL_VIDEOS_DIR / f"{act_id}.mp4"
            if local_vid.exists():
                remote_vid = self.upload_video(farm_id, act_id, str(local_vid))
                if remote_vid:
                    rec["captured_video_storage_path"] = remote_vid

            # Insert/upsert into farm_activities table
            db_record = {
                "activity_id": act_id,
                "farm_id": farm_id,
                "camera_id": str(rec.get("camera_id", DEFAULT_CAMERA_ID)),
                "cattle_id": rec.get("cattle_id"),
                "worker_id": rec.get("worker_id"),
                "activity_type": str(rec.get("activity_type", "farm_activity")),
                "detection_date": str(rec.get("detection_date", datetime.date.today().isoformat())),
                "start_time": str(rec.get("start_time", datetime.datetime.now(datetime.timezone.utc).isoformat())),
                "end_time": str(rec.get("end_time", datetime.datetime.now(datetime.timezone.utc).isoformat())),
                "activity_duration": float(rec.get("activity_duration", 10.0)),
                "eating_duration_sec": float(rec.get("eating_duration_sec", rec.get("activity_duration", 10.0) if "feeding" in str(rec.get("activity_type")) else 0.0)),
                "resting_duration_sec": float(rec.get("resting_duration_sec", rec.get("activity_duration", 10.0) if "resting" in str(rec.get("activity_type")) else 0.0)),
                "herd_count": int(rec.get("herd_count", 1)),
                "breed": str(rec.get("breed", "Cattle")),
                "disease_status": str(rec.get("disease_status", "Healthy")),
                "bcs_score": float(rec["bcs_score"]) if (rec.get("bcs_score") is not None and float(rec.get("bcs_score") or 0) > 0) else None,
                "estimated_weight_kg": float(rec.get("estimated_weight_kg", 400.0)),
                "estimated_height_cm": float(rec.get("estimated_height_cm", 130.0)),
                "ai_detection_result": rec.get("ai_detection_result", {}),
                "confidence_score": float(rec.get("confidence_score", 0.85)),
                "description": str(rec.get("description", "")),
                "captured_image_storage_path": str(rec.get("captured_image_storage_path", "")),
                "captured_video_storage_path": str(rec.get("captured_video_storage_path", "")),
                "video_source": str(rec.get("video_source", "uploaded")),
                "status": str(rec.get("status", "detected_sign")) if str(rec.get("status", "")) in ("suspected_sign", "detected_sign", "verified", "dismissed") else "detected_sign",
                "created_at": str(rec.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()))
            }
            try:
                self._ensure_farm_exists(farm_id)
                self._ensure_camera_exists(farm_id, db_record["camera_id"])
                try:
                    self.client.table("farm_activities").upsert(db_record).execute()
                except Exception as ex:
                    err_msg = str(ex)
                    if "PGRST204" in err_msg or "column" in err_msg.lower():
                        base_keys = ["activity_id", "farm_id", "camera_id", "cattle_id", "worker_id", "activity_type",
                                     "detection_date", "start_time", "end_time", "activity_duration", "ai_detection_result",
                                     "confidence_score", "description", "captured_image_storage_path",
                                     "captured_video_storage_path", "video_source", "status", "created_at"]
                        fallback_rec = {k: db_record[k] for k in base_keys if k in db_record}
                        self.client.table("farm_activities").upsert(fallback_rec).execute()
                    else:
                        raise ex
                rec["_synced_to_supabase"] = True
                synced += 1
                modified = True
            except Exception as e:
                print(f"[Sync Error] Activity {act_id}: {e}")

        if modified:
            try:
                with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, default=str)
            except Exception:
                pass

        return {
            "success": True,
            "synced_count": synced,
            "message": f"Successfully synced {synced} new activities to Supabase!"
        }

    def sync_local_roster_to_supabase(self, farm_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronizes all local cattle and worker profiles from local roster JSON files
        into Supabase 'cattle' and 'workers' tables, uploading photos to Supabase Storage.
        """
        farm_id = str(farm_id or DEFAULT_FARM_ID)
        cattle_synced = 0
        workers_synced = 0

        # Sync Cattle Roster
        if CATTLE_ROSTER_FILE.exists():
            try:
                with open(CATTLE_ROSTER_FILE, "r", encoding="utf-8") as f:
                    c_roster = json.load(f)
                for cid, c_data in c_roster.items():
                    c_farm = str(c_data.get("farm_id") or farm_id)
                    img_url = c_data.get("image_url", "")
                    storage_path = c_data.get("captured_image_storage_path", "")
                    if not storage_path and img_url:
                        local_name = os.path.basename(img_url)
                        local_path = LOCAL_IMAGES_DIR / local_name
                        if local_path.exists():
                            storage_path = self.upload_cattle_image(c_farm, cid, str(local_path))
                            if storage_path:
                                c_data["captured_image_storage_path"] = storage_path
                    self.upsert_cattle(c_farm, c_data)
                    cattle_synced += 1
            except Exception as e:
                print(f"[Supabase Cattle Sync Notice] {e}")

        # Sync Workers Roster
        if WORKER_ROSTER_FILE.exists():
            try:
                with open(WORKER_ROSTER_FILE, "r", encoding="utf-8") as f:
                    w_roster = json.load(f)
                for wid, w_data in w_roster.items():
                    w_farm = str(w_data.get("farm_id") or farm_id)
                    img_url = w_data.get("image_url", "")
                    storage_path = w_data.get("captured_image_storage_path", "")
                    if not storage_path and img_url:
                        local_name = os.path.basename(img_url)
                        local_path = LOCAL_IMAGES_DIR / local_name
                        if local_path.exists():
                            storage_path = self.upload_worker_image(w_farm, wid, str(local_path))
                            if storage_path:
                                w_data["captured_image_storage_path"] = storage_path
                    self.upsert_worker(w_farm, w_data)
                    workers_synced += 1
            except Exception as e:
                print(f"[Supabase Worker Sync Notice] {e}")

        return {
            "success": True,
            "cattle_synced": cattle_synced,
            "workers_synced": workers_synced
        }

    def clean_duplicate_activities(self, farm_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Consolidates and removes duplicate entries for the same cattle in Supabase and local cache.
        Keeps the single most comprehensive session for each cattle.
        """
        deleted_count = 0
        if self.is_connected and self.client:
            try:
                query = self.client.table("farm_activities").select("*")
                if farm_id and farm_id != "all":
                    query = query.eq("farm_id", farm_id)
                res = query.order("created_at", desc=False).execute()
                if res.data:
                    seen = {}
                    for row in res.data:
                        cid = row.get("cattle_id")
                        atype = row.get("activity_type")
                        row_farm = row.get("farm_id")
                        if not cid:
                            continue
                        key = f"{row_farm}_{cid}_{atype}"
                        if key in seen:
                            # Duplicate found - keep the one with longer duration or more complete data
                            prev = seen[key]
                            prev_dur = float(prev.get("activity_duration") or 0.0)
                            curr_dur = float(row.get("activity_duration") or 0.0)
                            
                            if curr_dur >= prev_dur:
                                del_id = prev["activity_id"]
                                seen[key] = row
                            else:
                                del_id = row["activity_id"]

                            try:
                                self.client.table("farm_activities").delete().eq("activity_id", del_id).execute()
                                deleted_count += 1
                                print(f"[Clean Duplicate] Deleted duplicate row {del_id} for {cid} ({atype})")
                            except Exception as del_err:
                                print(f"[Clean Duplicate Delete Error] {del_err}")
                        else:
                            seen[key] = row
            except Exception as e:
                print(f"[Duplicate Cleanup Error] {e}")

        # Also clean local cache
        if LOCAL_DB_FILE.exists():
            try:
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                seen = set()
                cleaned = []
                for r in recs:
                    cid = r.get("cattle_id")
                    atype = r.get("activity_type")
                    if not cid:
                        cleaned.append(r)
                        continue
                    key = f"{cid}_{atype}"
                    if key not in seen:
                        seen.add(key)
                        cleaned.append(r)
                with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, indent=2, default=str)
            except Exception:
                pass

        return {"success": True, "deleted_count": deleted_count, "message": f"Consolidated duplicates: removed {deleted_count} duplicate rows."}

    def get_farm_cameras(self, farm_id: str) -> List[Dict[str, Any]]:
        """Returns list of cameras configured for this farm."""
        if self.is_connected and self.client:
            try:
                res = self.client.table("cameras").select("*").eq("farm_id", farm_id).execute()
                if res.data and len(res.data) > 0:
                    return res.data
            except Exception:
                pass

        # Default camera list
        return [
            {"id": "CAM01", "name": "Main Gate & Feed Alley", "location_description": "Cattle Corral", "is_active": True},
            {"id": "CAM02", "name": "Milking & Stall Camera", "location_description": "Milking Parlor", "is_active": True}
        ]


# Global singleton instance
supabase_manager = SupabaseManager()
