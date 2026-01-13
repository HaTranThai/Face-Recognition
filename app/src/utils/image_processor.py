"""
Image processing utilities for face recognition system.

New MinIO architecture:

Employee bucket: data-face-employee-images
  store=<store_id>/
    register/
      employee=<employee_id>/
        date=YYYY_MM_DD/
          HH_MM_SS_<real_name>.jpg
    checkin/
      date=YYYY_MM_DD/
        employee=<employee_id>/
          HH_MM_SS_<real_name>.jpg

Customer bucket: data-face-customer-images
  store=<store_id>/
    register/
      customer=<customer_id>/
        date=YYYY_MM_DD/
          HH_MM_SS_<real_name>.jpg
    checkin/
      date=YYYY_MM_DD/
        customer=<customer_id>/
          HH_MM_SS_<real_name>.jpg
"""

import asyncio
import base64
import cv2
import datetime
import numpy as np
import re
import time
from io import BytesIO
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from config.logging import get_minio_logger

logger = get_minio_logger()


class ImageProcessor:
    """Handles image processing and storage operations."""

    # Force new buckets (avoid being overridden by legacy config)
    EMPLOYEE_BUCKET = "data-face-employee-images"
    CUSTOMER_BUCKET = "data-face-customer-images"

    def __init__(self, config):
        self.config = config

        # Optional: avoid overwrite if same key exists (default True)
        self.AVOID_OVERWRITE = bool(getattr(self.config, "AVOID_OVERWRITE", True))

        # Optional: prefer explicit endpoint + credentials from config
        self.MINIO_ENDPOINT = getattr(self.config, "MINIO_ENDPOINT", None)  # e.g. http://minio:9000
        self.MINIO_ACCESS_KEY = getattr(self.config, "MINIO_ACCESS_KEY", "minioadmin")
        self.MINIO_SECRET_KEY = getattr(self.config, "MINIO_SECRET_KEY", "minioadmin1245")
        self.MINIO_REGION = getattr(self.config, "MINIO_REGION", "us-east-1")

        logger.warning(
            f"[INIT] Using buckets: EMPLOYEE_BUCKET={self.EMPLOYEE_BUCKET} | CUSTOMER_BUCKET={self.CUSTOMER_BUCKET}"
        )

    # --------------------------
    # S3 client
    # --------------------------
    def _get_s3_client(self):
        """Get S3 client for operations."""
        endpoint_url = self.MINIO_ENDPOINT
        if not endpoint_url:
            docker_env = bool(getattr(self.config, "DOCKER_ENV", False))
            endpoint_url = "http://minio:9000" if docker_env else "http://localhost:9000"

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self.MINIO_ACCESS_KEY,
            aws_secret_access_key=self.MINIO_SECRET_KEY,
            region_name=self.MINIO_REGION,
        )

    # --------------------------
    # Path / naming helpers
    # --------------------------
    def _safe_name_for_key(self, name: str, max_len: int = 80) -> str:
        """Keep readable but remove characters that can break keys."""
        if not name:
            return "unknown"

        s = name.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r'[\/\\:\*\?"<>\|]', "", s)  # remove dangerous chars
        s = re.sub(r"_+", "_", s)

        if len(s) > max_len:
            s = s[:max_len].rstrip("_")

        return s or "unknown"

    def _split_key_ext(self, key: str) -> Tuple[str, str]:
        """Return (base_without_ext, ext)"""
        dot = key.rfind(".")
        if dot == -1:
            return key, ""
        return key[:dot], key[dot:]

    def _build_object_key(
        self,
        store_id: str,
        person_type: str,   # "employee" | "customer"
        person_id: str,
        name: str,
        is_checkin: bool,
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """
        Build object key according to the final architecture:

        Register:
          store=<store_id>/register/<person_type>=<person_id>/date=YYYY_MM_DD/HH_MM_SS_<name>.jpg

        Checkin:
          store=<store_id>/checkin/date=YYYY_MM_DD/<person_type>=<person_id>/HH_MM_SS_<name>.jpg
        """
        now = now or datetime.datetime.now()
        date_part = now.strftime("%Y_%m_%d")
        time_part = now.strftime("%H_%M_%S")
        safe_name = self._safe_name_for_key(name)

        filename = f"{time_part}_{safe_name}.jpg"

        if is_checkin:
            return (
                f"store={store_id}/"
                f"checkin/"
                f"date={date_part}/"
                f"{person_type}={person_id}/"
                f"{filename}"
            )
        else:
            return (
                f"store={store_id}/"
                f"register/"
                f"{person_type}={person_id}/"
                f"date={date_part}/"
                f"{filename}"
            )

    # --------------------------
    # S3 operations
    # --------------------------
    def _ensure_bucket_exists(self, s3_client, bucket_name: str) -> None:
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError:
            s3_client.create_bucket(Bucket=bucket_name)

    def _object_exists(self, s3_client, bucket_name: str, object_name: str) -> bool:
        try:
            s3_client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError:
            return False

    def _resolve_collision_key(self, s3_client, bucket_name: str, object_name: str, max_tries: int = 200) -> str:
        """
        If object already exists, append suffix _01, _02 ... before extension.
        Keeps readability and prevents overwrite.
        """
        if not self.AVOID_OVERWRITE:
            return object_name

        if not self._object_exists(s3_client, bucket_name, object_name):
            return object_name

        base, ext = self._split_key_ext(object_name)
        for i in range(1, max_tries + 1):
            candidate = f"{base}_{i:02d}{ext}"
            if not self._object_exists(s3_client, bucket_name, candidate):
                return candidate

        # If too many collisions, fallback to timestamp with microseconds
        now = datetime.datetime.now().strftime("%H_%M_%S_%f")
        return f"{base}_{now}{ext}"

    def _upload_to_s3(self, s3_client, bucket_name: str, object_name: str, img_bytes: BytesIO) -> Tuple[bool, str]:
        """
        Upload image to S3 synchronously (to run in thread pool).
        Return (success, final_object_name).
        """
        try:
            self._ensure_bucket_exists(s3_client, bucket_name)

            final_key = self._resolve_collision_key(s3_client, bucket_name, object_name)

            img_bytes.seek(0)
            s3_client.upload_fileobj(
                img_bytes,
                bucket_name,
                final_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )

            logger.info(f"Successfully uploaded image to MinIO: {bucket_name}/{final_key}")
            return True, final_key

        except Exception as e:
            logger.error(f"Failed to upload image to MinIO: bucket={bucket_name} key={object_name} err={e}")
            return False, object_name

    async def save_face_image(self, data, img_decode, face_id: str, name: str, is_checkin: bool = True) -> bool:
        """
        Save face image to MinIO/S3 storage asynchronously.
        - role == "0" -> customer
        - else -> employee
        """
        start_time = time.time()
        img_bytes = None
        img_encoded = None

        try:
            role = str(getattr(data, "role", ""))  # normalize role: can be int 0 or str "0"
            store_id = str(getattr(data, "store_id", ""))

            if role == "0":
                person_type = "customer"
                bucket_name = self.CUSTOMER_BUCKET
            else:
                person_type = "employee"
                bucket_name = self.EMPLOYEE_BUCKET

            # Encode image
            encode_start = time.time()
            ok, img_encoded = cv2.imencode(".jpg", img_decode)
            if not ok:
                logger.error("Failed to encode image to JPG.")
                return False
            img_bytes = BytesIO(img_encoded.tobytes())
            logger.info(f"[TIMING] {store_id} - Image encoding time: {time.time() - encode_start:.3f}s")

            # Build object key with new architecture
            object_name = self._build_object_key(
                store_id=store_id,
                person_type=person_type,
                person_id=str(face_id),
                name=name,
                is_checkin=is_checkin,
            )

            logger.warning(f"[UPLOAD_DEBUG] bucket={bucket_name} key={object_name}")

            loop = asyncio.get_event_loop()
            s3_client = self._get_s3_client()

            upload_start = time.time()
            success, final_key = await loop.run_in_executor(
                None,
                self._upload_to_s3,
                s3_client,
                bucket_name,
                object_name,
                img_bytes,
            )

            logger.info(f"[TIMING] {store_id} - Upload time: {time.time() - upload_start:.3f}s")
            logger.info(f"[TIMING] {store_id} - Total save image time: {time.time() - start_time:.3f}s")
            if success and final_key != object_name:
                logger.warning(f"[UPLOAD_DEBUG] collision resolved -> {final_key}")
            return success

        except Exception as e:
            logger.error(f"Error in save_face_image: {e}")
            return False

        finally:
            if img_bytes:
                img_bytes.close()
            if img_encoded is not None:
                del img_encoded

    # --------------------------
    # Other utilities (unchanged)
    # --------------------------
    def decode_base64_image(self, img_base64: str) -> np.ndarray:
        try:
            contents = base64.b64decode(img_base64)
            img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
            return img_decode
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise

    def resize_image(self, image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
        return cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
