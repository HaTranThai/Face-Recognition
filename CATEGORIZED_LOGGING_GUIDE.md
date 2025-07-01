# 📝 Hướng Dẫn Quản Lý Logs Phân Loại

## 📊 Tổng Quan

Hệ thống logging đã được chia thành 4 loại chính để dễ quản lý:

- **📸 face.log** - Các hoạt động nhận diện khuôn mặt
- **🗄️ database.log** - Các thao tác với database (Qdrant)
- **💾 minio.log** - Các hoạt động lưu trữ MinIO
- **⚙️ app.log** - Logs tổng quát của ứng dụng

## 🎯 Tính Năng Logging

### Rotating File Handler
- Mỗi file log tối đa 10MB
- Giữ lại 5 file backup cũ
- Tự động rotate khi đạt giới hạn

### Định Dạng Log
```
YYYY-MM-DD HH:MM:SS - logger_name - LEVEL - function_name:line_number - message
```

## 🔧 Sử dụng Script Quản Lý Logs

### Xem danh sách tất cả logs
```bash
./app/scripts/logs.sh list
```

### Xem logs realtime theo loại
```bash
# Theo dõi face logs
./app/scripts/logs.sh follow face

# Theo dõi database logs  
./app/scripts/logs.sh follow database

# Theo dõi minio logs
./app/scripts/logs.sh follow minio

# Theo dõi app logs
./app/scripts/logs.sh follow app
```

### Xem N dòng cuối của logs
```bash
# Xem 50 dòng cuối (mặc định)
./app/scripts/logs.sh tail face

# Xem 100 dòng cuối
./app/scripts/logs.sh tail database 100
```

### Tìm kiếm trong logs
```bash
# Tìm kiếm từ khóa "error" trong face logs
./app/scripts/logs.sh search face error

# Tìm kiếm "upload" trong minio logs
./app/scripts/logs.sh search minio upload
```

### Xem chỉ các lỗi
```bash
# Xem lỗi trong face logs
./app/scripts/logs.sh errors face

# Xem lỗi trong database logs
./app/scripts/logs.sh errors database
```

### Xem thống kê logs
```bash
./app/scripts/logs.sh stats
```

### Dọn dẹp logs cũ
```bash
# Xóa logs cũ hơn 7 ngày (mặc định)
./app/scripts/logs.sh clean

# Xóa logs cũ hơn 3 ngày
./app/scripts/logs.sh clean 3
```

## 📁 Cấu Trúc File Logs

```
logs/
├── face.log              # Current face logs
├── face.log.1            # Rotated face logs
├── face.log.2            # Older rotated logs
├── database.log          # Current database logs
├── database.log.1        # Rotated database logs
├── minio.log            # Current minio logs
├── minio.log.1          # Rotated minio logs
├── app.log              # Current app logs
└── app.log.1            # Rotated app logs
```

## 🎯 Logging Trong Code

### Face Recognition Operations
```python
from config.logging import get_face_logger

logger = get_face_logger()

# Log face recognition activities
logger.info(f"Face detection started for image {image_id}")
logger.warning(f"Low confidence score: {confidence}")
logger.error(f"Face detection failed: {error}")
```

### Database Operations
```python
from config.logging import get_database_logger

logger = get_database_logger()

# Log database activities
logger.info(f"Creating collection: {collection_name}")
logger.warning(f"Collection already exists: {collection_name}")
logger.error(f"Database connection failed: {error}")
```

### MinIO Operations
```python
from config.logging import get_minio_logger

logger = get_minio_logger()

# Log storage activities
logger.info(f"Uploading image to bucket: {bucket_name}")
logger.warning(f"Bucket not found: {bucket_name}")
logger.error(f"Upload failed: {error}")
```

### General Application
```python
from config.logging import get_app_logger

logger = get_app_logger()

# Log general app activities
logger.info("Application started")
logger.warning("Configuration missing")
logger.error(f"Unexpected error: {error}")
```

## 🔍 Monitoring và Troubleshooting

### Kiểm tra logs theo thời gian thực
```bash
# Terminal 1: Face logs
./app/scripts/logs.sh follow face

# Terminal 2: Database logs  
./app/scripts/logs.sh follow database

# Terminal 3: MinIO logs
./app/scripts/logs.sh follow minio
```

### Phân tích lỗi
```bash
# Xem tất cả lỗi gần đây
./app/scripts/logs.sh errors face
./app/scripts/logs.sh errors database
./app/scripts/logs.sh errors minio
./app/scripts/logs.sh errors app
```

### Tìm kiếm các sự kiện cụ thể
```bash
# Tìm các hoạt động upload
./app/scripts/logs.sh search minio upload

# Tìm các lỗi kết nối database
./app/scripts/logs.sh search database connection

# Tìm các hoạt động face recognition
./app/scripts/logs.sh search face "face detection"
```

## 📊 Log Level Guide

- **DEBUG**: Thông tin chi tiết cho debugging
- **INFO**: Thông tin hoạt động bình thường
- **WARNING**: Cảnh báo, hệ thống vẫn hoạt động
- **ERROR**: Lỗi nghiêm trọng, cần xử lý
- **CRITICAL**: Lỗi nguy hiểm, có thể crash hệ thống

## 🚀 Performance Tips

1. **Sử dụng appropriate log level**: Tránh log quá nhiều ở production
2. **Log rotation**: Logs tự động rotate, không cần lo lắng về dung lượng
3. **Structured logging**: Sử dụng format nhất quán
4. **Category separation**: Logs đã được phân loại, dễ troubleshoot

## 🔧 Cấu hình

Các cấu hình logging trong `app/config/logging.py`:

- File size limit: 10MB per file
- Backup count: 5 files
- Log format: Timestamp - Logger - Level - Function:Line - Message
- Console output: Chỉ ở DEBUG mode

## 📝 Best Practices

1. **Log tại đúng level**: INFO cho hoạt động bình thường, ERROR cho lỗi
2. **Include context**: Thêm ID, tên file, parameters vào log message
3. **Avoid sensitive data**: Không log password, token, personal info
4. **Use structured format**: Consistent message format
5. **Monitor regularly**: Thường xuyên check logs để phát hiện issue sớm
