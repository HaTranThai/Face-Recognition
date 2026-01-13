#!/usr/bin/env python3
import os
import asyncio
import aiohttp
import shutil
import logging
import datetime
import time
from pathlib import Path

SNAPSHOT_SOURCE_DIR = "/home/bbsw/Face-Recognition/app/snapshots"
QDRANT_API = "http://127.0.0.1:7005"
MINIO_ALIAS = "MINIO_LOCAL"
BUCKET = "backup-qdrant"
LOG_FILE = "/home/bbsw/Face-Recognition/Backup/Backup_logs/backup-qdrant.log"

CONCURRENCY_LIMIT = 10 

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def blocking_find_largest_snap(collection_path):
    try:
        p = Path(collection_path)
        files = list(p.glob('*.snapshot'))
        if not files:
            return None
            
        largest = max(files, key=os.path.getsize)
        
        if os.path.getsize(largest) == 0:
            return None
        return largest
    except Exception:
        return None

def blocking_copy_file(src, dst):
    shutil.copy2(src, dst)

async def create_snapshot(session, collection):
    try:
        async with session.get(f"{QDRANT_API}/create_snapshot/{collection}") as resp:
            if resp.status != 200:
                text = await resp.text()
                logging.warning(f"⚠️ API Status {resp.status} for {collection}: {text}")
            return True
    except Exception as e:
        logging.error(f"❌ API Call Error {collection}: {str(e)}")
        return False

async def upload_minio(local_path, remote_path):
    proc = await asyncio.create_subprocess_exec(
        "mcli", "cp", str(local_path), remote_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        raise Exception(f"MCLI Error: {stderr.decode().strip()}")
    return True

async def process_collection(sem, session, col_name):
    async with sem:
        temp_path = None
        try:
            if not await create_snapshot(session, col_name):
                return 

            source_col_dir = os.path.join(SNAPSHOT_SOURCE_DIR, col_name)
            latest_snap = None
            
            for _ in range(5):
                latest_snap = await asyncio.to_thread(blocking_find_largest_snap, source_col_dir)
                if latest_snap:
                    break
                await asyncio.sleep(2) 

            if not latest_snap:
                logging.error(f"❌ {col_name}: Snapshot created but file NOT FOUND or Empty.")
                return

            file_size = os.path.getsize(latest_snap)
            file_size_mb = f"{file_size / (1024 * 1024):.2f} MB"
            
            fixed_name = f"{col_name}.snapshot"
            temp_path = os.path.join("/tmp", fixed_name)

            await asyncio.to_thread(blocking_copy_file, latest_snap, temp_path)

            remote_path = f"{MINIO_ALIAS}/{BUCKET}/{col_name}/"
            await upload_minio(temp_path, remote_path)

            logging.info(f"✅ {col_name}: Success | Size: {file_size_mb}")

        except Exception as e:
            logging.error(f"💥 {col_name}: FAILED - {str(e)}")
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

async def main():
    start_time_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    start_perf = time.perf_counter() 
    
    logging.info(f"--- START BACKUP at {start_time_str} ---")

    if not os.path.exists(SNAPSHOT_SOURCE_DIR):
        logging.error(f"CRITICAL: Source directory {SNAPSHOT_SOURCE_DIR} does not exist.")
        return

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{QDRANT_API}/get_collections") as resp:
                data = await resp.json()    
                collections = [c for c in data['collections'] if isinstance(c, str)]
        except Exception as e:
            logging.error(f"Fetch collections failed: {str(e)}")
            return

        if not collections:
            logging.warning("No collections found.")
            return

        logging.info(f"Found {len(collections)} collections. Processing (Safety Mode)...")

        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        
        tasks = [process_collection(sem, session, col) for col in collections]
        
        await asyncio.gather(*tasks)

    end_perf = time.perf_counter()
    duration = end_perf - start_perf
    logging.info(f"--- DONE. Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes) ---")

if __name__ == "__main__":
    asyncio.run(main())