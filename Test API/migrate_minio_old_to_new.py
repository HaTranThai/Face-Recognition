import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ClientError


# -----------------------------
# Old buckets -> New buckets + op
# -----------------------------
BUCKET_MAP = {
    # employees
    "data-face-register-employee-images": ("data-face-employee-images", "employee", "register"),
    "data-face-checkin-employee-images": ("data-face-employee-images", "employee", "checkin"),
    # customers
    "data-face-register-customer-images": ("data-face-customer-images", "customer", "register"),
    "data-face-checkin-customer-images": ("data-face-customer-images", "customer", "checkin"),
}


def make_s3_client(endpoint: str, access_key: str, secret_key: str, region: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def ensure_bucket_exists(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)


def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def iter_objects(s3, bucket: str, prefix: str = ""):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj.get("Size", 0)


def safe_name_for_key(name: str, max_len: int = 80) -> str:
    """Keep readable but remove dangerous chars for key."""
    if not name:
        return "unknown"
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r'[\/\\:\*\?"<>\|]', "", s)
    s = re.sub(r"_+", "_", s)
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "unknown"


def parse_old_key(old_key: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Old key:
      <store_id>/<YYYY_MM_DD>/<face_id>_<name>_<HH_MM_SS>.jpg

    Return:
      (store_id, date_YYYY_MM_DD, person_id, time_HH_MM_SS, name)
    """
    parts = old_key.split("/")
    if len(parts) < 3:
        return None

    store_id = parts[0]
    date_part = parts[1]
    filename = parts[-1]

    if not filename.lower().endswith(".jpg"):
        return None

    base = filename[:-4]
    chunks = base.split("_")
    if len(chunks) < 5:
        return None

    # last 3 chunks are time
    hh, mm, ss = chunks[-3], chunks[-2], chunks[-1]
    time_part = f"{hh}_{mm}_{ss}"

    person_id = chunks[0]
    name_chunks = chunks[1:-3]
    name = "_".join(name_chunks)

    return store_id, date_part, person_id, time_part, name


def build_new_key(
    store_id: str,
    person_type: str,   # employee | customer
    person_id: str,
    op: str,            # register | checkin
    date_part: str,     # YYYY_MM_DD
    time_part: str,     # HH_MM_SS
    name: str
) -> str:
    """
    New architecture (final):

    Employee bucket:
      store=<store_id>/
        register/
          employee=<employee_id>/
            date=YYYY_MM_DD/
              HH_MM_SS_<real_name>.jpg
        checkin/
          date=YYYY_MM_DD/
            employee=<employee_id>/
              HH_MM_SS_<real_name>.jpg

    Customer bucket: same but customer=<customer_id>
    """
    safe_name = safe_name_for_key(name)
    filename = f"{time_part}_{safe_name}.jpg"

    if op == "register":
        # store=175/register/employee=198/date=2025_03_24/10_23_49_beo.jpg
        return f"store={store_id}/register/{person_type}={person_id}/date={date_part}/{filename}"

    # checkin:
    # store=175/checkin/date=2025_03_24/employee=198/10_23_49_beo.jpg
    return f"store={store_id}/checkin/date={date_part}/{person_type}={person_id}/{filename}"


def copy_object_server_side(s3, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str):
    s3.copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key},
        MetadataDirective="COPY",
        ContentType="image/jpeg",
    )


def delete_object(s3, bucket: str, key: str):
    s3.delete_object(Bucket=bucket, Key=key)


def migrate_one(
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
    src_bucket: str,
    src_key: str,
    dst_bucket: str,
    dst_key: str,
    skip_if_exists: bool,
    dry_run: bool,
    delete_src_after: bool,
):
    s3 = make_s3_client(endpoint, access_key, secret_key, region)

    if skip_if_exists and object_exists(s3, dst_bucket, dst_key):
        return True, f"SKIP exists: {dst_bucket}/{dst_key}"

    if dry_run:
        return True, f"DRY_RUN: {src_bucket}/{src_key} -> {dst_bucket}/{dst_key}"

    copy_object_server_side(s3, src_bucket, src_key, dst_bucket, dst_key)

    if delete_src_after:
        delete_object(s3, src_bucket, src_key)

    return True, f"COPIED: {src_bucket}/{src_key} -> {dst_bucket}/{dst_key}"


def main():
    ap = argparse.ArgumentParser(description="Migrate MinIO buckets (old schema) -> (new schema).")
    ap.add_argument("--endpoint", default=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"))
    ap.add_argument("--access-key", default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.getenv("MINIO_SECRET_KEY", "minioadmin1245"))
    ap.add_argument("--region", default=os.getenv("MINIO_REGION", "us-east-1"))

    ap.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "16")))
    ap.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN", "0") == "1")
    ap.add_argument("--skip-if-exists", action="store_true", default=os.getenv("SKIP_IF_EXISTS", "1") == "1")
    ap.add_argument("--delete-src-after", action="store_true", default=os.getenv("DELETE_SRC_AFTER", "0") == "1")

    ap.add_argument("--prefix", default=os.getenv("SRC_PREFIX", ""), help="Only migrate keys with this prefix in old buckets (optional).")
    args = ap.parse_args()

    s3 = make_s3_client(args.endpoint, args.access_key, args.secret_key, args.region)

    # Ensure destination buckets exist
    for _, (dst_bucket, _, _) in BUCKET_MAP.items():
        ensure_bucket_exists(s3, dst_bucket)

    total = 0
    ok = 0
    fail = 0
    futures = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for src_bucket, (dst_bucket, person_type, op) in BUCKET_MAP.items():
            # skip if old bucket missing
            try:
                s3.head_bucket(Bucket=src_bucket)
            except ClientError:
                print(f"[WARN] Missing old bucket, skip: {src_bucket}")
                continue

            print(f"[INFO] Scanning: {src_bucket} (prefix='{args.prefix}')")
            for src_key, _sz in iter_objects(s3, src_bucket, prefix=args.prefix):
                parsed = parse_old_key(src_key)
                if not parsed:
                    # If not match schema, copy to legacy/ to avoid losing anything
                    dst_key = f"legacy/{src_bucket}/{src_key}"
                else:
                    store_id, date_part, person_id, time_part, name = parsed
                    dst_key = build_new_key(store_id, person_type, person_id, op, date_part, time_part, name)

                futures.append(
                    ex.submit(
                        migrate_one,
                        args.endpoint,
                        args.access_key,
                        args.secret_key,
                        args.region,
                        src_bucket,
                        src_key,
                        dst_bucket,
                        dst_key,
                        args.skip_if_exists,
                        args.dry_run,
                        args.delete_src_after,
                    )
                )
                total += 1

        for fut in as_completed(futures):
            try:
                success, _msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                if (ok + fail) % 500 == 0:
                    print(f"[PROGRESS] ok={ok} fail={fail} total={total}")
            except Exception as e:
                fail += 1
                print(f"[ERROR] {e}")

    print(f"\n[DONE] total={total} ok={ok} fail={fail} dry_run={args.dry_run} delete_src_after={args.delete_src_after}")


if __name__ == "__main__":
    main()
