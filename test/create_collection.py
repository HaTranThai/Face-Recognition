import os
import httpx
import glob

SERVER_IP = "localhost"
DB_API_PORT = 7005       
QDRANT_PORT = 6333       
SNAPSHOT_DIR = "/home/bbsw/Face-Recognition/app/snapshots" 

def process_snapshots():
    folders = [f for f in os.listdir(SNAPSHOT_DIR) if os.path.isdir(os.path.join(SNAPSHOT_DIR, f))]
    
    for collection_name in folders:

        folder_path = os.path.join(SNAPSHOT_DIR, collection_name)
        snapshot_files = glob.glob(os.path.join(folder_path, "*.snapshot"))
        
        if not snapshot_files:
            print(f"⚠️ Không tìm thấy file .snapshot trong {folder_path}")
            continue
            
        # snapshot_path = snapshot_files[-1]  
        snapshot_path = max(snapshot_files, key=os.path.getsize)
        print("USING:", os.path.basename(snapshot_path), "SIZE:", os.path.getsize(snapshot_path))
        print(f"🚀 Đang upload: {os.path.basename(snapshot_path)}...")

        upload_url = f"http://{SERVER_IP}:{QDRANT_PORT}/collections/{collection_name}/snapshots/upload"
        try:
            with httpx.Client(timeout=600.0) as client:
                with open(snapshot_path, "rb") as f:
                    files = {"snapshot": (os.path.basename(snapshot_path), f)}
                    response = client.post(upload_url, files=files)
                    
                    if response.status_code == 200:
                        print(f"✨ Import thành công snapshot cho {collection_name}")
                    else:
                        print(f"❌ Lỗi upload snapshot: {response.status_code} - {response.text}")
                    
                    print("-" * 50)
                    
        except Exception as e:
            print(f"❌ Lỗi kết nối khi upload: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists(SNAPSHOT_DIR):
        print(f"❌ Thư mục {SNAPSHOT_DIR} không tồn tại!")
    else:
        process_snapshots()