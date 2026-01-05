#!/bin/bash
# Thêm PATH để Crontab tìm được lệnh mcli, curl, jq
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

set -e

SNAPSHOT_SOURCE_DIR="/home/bbsw/Face-Recognition/app/snapshots" 
QDRANT_API="http://127.0.0.1:7005"
MINIO_ALIAS="MINIO_LOCAL"     
BUCKET="backup-qdrant"
LOG="/home/bbsw/Face-Recognition/Backup_logs/backup-qdrant.log"

DAY_OF_MONTH=$(date +%-d)
if [ $((DAY_OF_MONTH % 2)) -eq 0 ]; then
    TAG="Chan" 
else
    TAG="Le"
fi

echo "[$(date)] --- START BACKUP (Mode: $TAG) ---" >> "$LOG"

if [ ! -d "$SNAPSHOT_SOURCE_DIR" ]; then
    echo "ERROR: Directory $SNAPSHOT_SOURCE_DIR does not exist (Check path!)" >> "$LOG"
    exit 1
fi

# 1. Lấy danh sách collection
collections=$(curl -s "$QDRANT_API/get_collections" | jq -r '.collections[]')

if [ -z "$collections" ]; then
    echo "ERROR: No collections found or API failed" >> "$LOG"
    exit 1
fi

for col in $collections; do
    # 2. Tạo snapshot
    curl -s "$QDRANT_API/create_snapshot/$col" > /dev/null
    
    # 3. Vòng lặp chờ file xuất hiện (Retry 5 lần)
    SOURCE_COL_DIR="$SNAPSHOT_SOURCE_DIR/$col"
    FOUND=0
    
    for i in {1..5}; do
        LATEST_SNAP=$(find "$SOURCE_COL_DIR" -maxdepth 1 -name "*.snapshot" -printf "%s %p\n" 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)
        
        if [ -n "$LATEST_SNAP" ]; then
            FOUND=1
            break
        fi
        sleep 2
    done
    
    if [ $FOUND -eq 1 ]; then
        FILE_SIZE=$(du -h "$LATEST_SNAP" | cut -f1)
        REAL_FILENAME=$(basename "$LATEST_SNAP")

        # 4. Đổi tên file
        FIXED_NAME="${col}_${TAG}.snapshot"
        TEMP_PATH="/tmp/$FIXED_NAME"
        
        cp "$LATEST_SNAP" "$TEMP_PATH"
        
        # 5. Upload lên MinIO
        mcli cp "$TEMP_PATH" "$MINIO_ALIAS/$BUCKET/$col/"
        
        echo "[$(date)] Uploaded: $FIXED_NAME | Size: $FILE_SIZE | Original: $REAL_FILENAME" >> "$LOG"
        
        rm -f "$TEMP_PATH"
    else
        echo "[$(date)] WARNING: No snapshot found for $col (checked path: $SOURCE_COL_DIR)" >> "$LOG"
    fi
done

echo "[$(date)] --- DONE ---" >> "$LOG"