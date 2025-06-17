# 🔍 CMD Face Recognition System

Hệ thống nhận diện khuôn mặt tiên tiến được xây dựng bằng FastAPI với kiến trúc microservices, sử dụng Qdrant vector database và MinIO object storage.

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────┐    HTTP API    ┌──────────────────┐    Direct    ┌─────────────┐
│ Face Recognition│───────────────→│ qdrant_database_ │─────────────→│   Qdrant    │
│      API        │    (httpx)     │       FE         │   (client)   │  Database   │
│   (Port 2024)   │                │   (Port 7005)    │              │ (Port 6333) │
└─────────────────┘                └──────────────────┘              └─────────────┘
        │                                                                     ▲
        │                                                                     │
        ▼                                                                     │
┌─────────────────┐                                              ┌─────────────────┐
│     MinIO       │                                              │   Snapshots     │
│   Storage       │                                              │   & Backups     │
│ (Port 9000)     │                                              │                 │
└─────────────────┘                                              └─────────────────┘
```

### Các Service Chính

1. **Face Recognition API** (Port 2024) - API chính xử lý nhận diện khuôn mặt
2. **Database API** (Port 7005) - Service quản lý database operations
3. **Qdrant Database** (Port 6333) - Vector database lưu trữ face embeddings
4. **MinIO Storage** (Port 9000) - Object storage cho hình ảnh

## 🚀 Khởi Động Nhanh

### 1. Khởi động tất cả services

```bash
chmod +x start_services.sh
./start_services.sh
```

**Script tự động:**
- 🔍 Kiểm tra Docker có sẵn không
- 🛑 Dừng các containers đang chạy
- 🎯 **Auto-detect GPU**: Tự động kiểm tra GPU và chọn cấu hình phù hợp
  - ✅ Có GPU: Sử dụng `docker-compose.yml` (GPU accelerated)
  - ⚠️ Không có GPU: Sử dụng `docker-compose-cpu.yml` (CPU only)
- 🔨 Build và khởi động tất cả services
- 📊 Hiển thị status và links truy cập

### 2. Kiểm tra integration

```bash
chmod +x test_integration.sh
./test_integration.sh
```

### 3. Truy cập API Documentation

- **Database API**: http://localhost:7005/docs
- **Face Recognition API**: http://localhost:2024/docs
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin1245)

## 📋 Mô Tả Chi Tiết Các Service

### 1. Face Recognition API (Port 2024)
Ứng dụng chính xử lý:
- Phát hiện và nhận diện khuôn mặt
- Đăng ký khuôn mặt mới
- Quản lý hình ảnh thông qua MinIO
- Xử lý batch operations
- Backup và restore dữ liệu

**Công nghệ sử dụng:**
- FastAPI framework
- YOLOv8 cho face detection
- DeepFace cho face embeddings
- OpenCV cho image processing
- AsyncIO cho xử lý bất đồng bộ

### 2. Database API Service (Port 7005)
Service chuyên biệt cho database operations:
- CRUD operations trên Qdrant
- Quản lý collections
- Search operations
- Snapshot management

**Endpoints chính:**
```http
GET    /get_collections          # Lấy danh sách collections
POST   /create_collection        # Tạo collection mới
POST   /insert_point            # Thêm face embedding
POST   /search_point            # Tìm kiếm face tương tự
DELETE /delete_point            # Xóa face theo ID
GET    /create_snapshot/{collection}  # Tạo snapshot
POST   /recover_snapshot        # Khôi phục từ snapshot
```

### 3. Qdrant Vector Database (Port 6333)
Database chuyên dụng cho vector embeddings:
- Lưu trữ face embeddings dạng vector
- Fast similarity search
- Persistent storage
- Snapshot support

### 4. MinIO Object Storage (Port 9000)
Lưu trữ hình ảnh:
- Checkin customer images
- Checkin employee images  
- Register customer images
- Register employee images

## 🔌 API Endpoints Chính

### Face Recognition APIs

#### Đăng ký khuôn mặt
```http
POST /create_face_img_base64
Content-Type: application/json

{
    "img_base64": "base64_encoded_image",
    "id": "person_id",
    "name": "Person Name",
    "role": "1",  // 1: Employee, 0: Customer
    "store_id": "store_123"
}
```

#### Nhận diện khuôn mặt
```http
POST /face_recog_img_base64
Content-Type: application/json

{
    "img_base64": "base64_encoded_image",
    "role": "1",  // 1: Employee, 0: Customer
    "store_id": "store_123"
}
```

#### Xóa khuôn mặt
```http
DELETE /delete_face_img_base64
Content-Type: application/json

{
    "id": "person_id",
    "store_id": "store_123"
}
```

### Health Check APIs

```http
GET /health                  # Kiểm tra API cơ bản
GET /health/database        # Kiểm tra kết nối database
GET /health/minio          # Kiểm tra kết nối MinIO
GET /health/full           # Kiểm tra toàn bộ hệ thống
```

### Database Management APIs

```http
GET    /get_list_collections           # Danh sách collections
DELETE /delete_collection/{name}       # Xóa collection
POST   /create_snapshot/{collection}   # Tạo snapshot
POST   /recover_from_snapshot         # Khôi phục từ snapshot
```

### Batch Operations

```http
POST /batch_customers                  # Xử lý batch customers
GET  /backup_data/{store_id}          # Backup dữ liệu store
GET  /backup_all                      # Backup toàn bộ
POST /recover_db                      # Khôi phục database
```

## 📁 Cấu Trúc Dự Án

```
CMD_Face_Recognition/
├── 📄 docker-compose.yml              # Cấu hình Docker services
├── 📄 docker-compose-cpu.yml          # Cấu hình cho CPU-only
├── 🚀 start_services.sh               # Script khởi động services
├── 🧪 test_integration.sh             # Script test integration
├── 📋 requirements.txt                # Dependencies chính
│
├── 📁 app/                            # Face Recognition API
│   ├── 📄 main.py                     # Entry point
│   ├── 📄 requirements.txt            # Python dependencies
│   ├── 🐳 Dockerfile                  # Docker config for GPU
│   ├── 🐳 Dockerfile_cpu              # Docker config for CPU
│   │
│   ├── 📁 config/                     # Cấu hình
│   │   ├── 📄 settings.py             # App settings & env vars
│   │   ├── 📄 logging.py              # Logging configuration
│   │   └── 📄 __init__.py
│   │
│   ├── 📁 src/                        # Source code (Clean Architecture)
│   │   ├── 📁 api/                    # API layer
│   │   │   ├── 📄 app.py              # FastAPI factory
│   │   │   ├── 📄 routes.py           # Route aggregation
│   │   │   └── 📁 routers/            # Individual routers
│   │   │       ├── 📄 face.py         # Face operations
│   │   │       ├── 📄 database.py     # Database operations
│   │   │       ├── 📄 health.py       # Health checks
│   │   │       ├── 📄 test.py         # Test endpoints
│   │   │       └── 📄 default.py      # Default routes
│   │   │
│   │   ├── 📁 core/                   # Domain models
│   │   │   └── 📄 models.py           # Pydantic models
│   │   │
│   │   ├── 📁 services/               # Business logic
│   │   │   └── 📄 face_service.py     # Face recognition logic
│   │   │
│   │   └── 📁 utils/                  # Utilities
│   │       ├── 📄 image_processor.py  # MinIO image handling
│   │       ├── 📄 database_client.py  # HTTP client for DB API
│   │       └── 📄 legacy.py           # Legacy face functions
│   │
│   ├── 📁 models/                     # ML models
│   │   ├── 📄 yolo_onnx.py           # YOLOv8 implementation
│   │   ├── 📄 yolov8n-face.onnx      # YOLO face detection weights
│   │   └── 📄 best_face_mask.pt      # Face mask detection weights
│   │
│   ├── 📁 snapshots/                  # Database snapshots
│   │   ├── 📁 {store_id}_Customers/
│   │   └── 📁 {store_id}_Employees/
│   │
│   ├── 📁 logs/                       # Application logs
│   │   └── 📄 face.log
│   │
│   └── 📁 static/                     # Static files
│
├── 📁 qdrant_database_FE/             # Database API Service
│   ├── 📄 app.py                      # FastAPI app for DB operations
│   ├── 📄 main.py                     # Entry point
│   ├── 📄 requirements.txt            # Dependencies
│   └── 🐳 Dockerfile                  # Docker config
│
├── 📁 qdrant_storage/                 # Persistent Qdrant data
    ├── 📄 raft_state.json
    ├── 📁 aliases/
    └── 📁 collections/

```

## ⚙️ Cấu Hình

### Environment Variables

**Face Recognition API (.env):**
```bash
# Database API Configuration
QDRANT_DB_HOST=localhost
QDRANT_DB_PORT=7005

# Docker Environment
DOCKER_ENV=false

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin1245

# Face Detection Settings
CONF_THRESHOLD=0.7
BLUR_THRESHOLD=100
FACE_EXT=0.3

# Storage Paths
CHECKIN_CUSTOMER_PATH=data-face-checkin-customer-images
CHECKIN_EMPLOYEE_PATH=data-face-checkin-employee-images
REGISTER_CUSTOMER_PATH=data-face-register-customer-images
REGISTER_EMPLOYEE_PATH=data-face-register-employee-images
```

**Database API (.env):**
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
HOST=0.0.0.0
PORT=7005
```

### Docker Compose Configuration

```yaml
services:
  qdrant:        # Vector database (port 6333)
  api_db:        # Database API service (port 7005)
  api_fr:        # Face recognition API (port 2024)
  minio:         # Object storage (port 9000)
```

## 🚀 Các Tính Năng Chính

### Core Functionality
- **🎯 Face Detection**: YOLOv8-based với độ chính xác cao
- **🔍 Face Recognition**: Deep learning embeddings với similarity matching
- **📝 Face Registration**: Đăng ký faces vào collections (customer/employee)
- **✅ Face Check-in**: Xác thực khuôn mặt real-time
- **🔍 Quality Assessment**: Đánh giá chất lượng ảnh, blur detection, eye state
- **🛡️ Anti-spoofing**: Phát hiện liveness cơ bản

### Technical Features
- **🏗️ Microservices Architecture**: Tách biệt API và database services
- **🌐 HTTP-based Database Layer**: Tất cả database operations qua REST API
- **💾 Object Storage**: MinIO/S3 compatible image storage
- **❤️ Health Monitoring**: Health checks toàn diện
- **🐳 Docker Support**: Containerization đầy đủ với Docker Compose
- **⚡ Async Processing**: Xử lý bất đồng bộ cho images và database
- **🚨 Error Handling**: Logging và error management toàn diện

### API Capabilities
- **📊 Multiple Collections**: Tổ chức faces theo store/role
- **📦 Batch Operations**: Xử lý nhiều operations hiệu quả
- **💾 Snapshot Management**: Backup và restore database
- **⚡ Real-time Processing**: Face detection và recognition nhanh
- **📖 RESTful Design**: API endpoints clean và có documentation

## 🛠️ Cài Đặt & Thiết Lập

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU support (optional, tự động detect)
- 8GB+ RAM recommended
- 50GB+ storage cho images và models

**Lưu ý**: Script `start_services.sh` sẽ tự động kiểm tra GPU và chọn cấu hình phù hợp:
- 🎯 **GPU Auto-Detection**: Sử dụng `nvidia-smi` để kiểm tra GPU
- ✅ **GPU có sẵn**: Chạy với GPU acceleration (nhanh hơn)
- ⚠️ **Không có GPU**: Fallback về CPU mode (vẫn hoạt động bình thường)

### Quick Start

1. **Clone repository**
   ```bash
   git clone <repository-url> -b <branch-name>
   cd CMD_Face_Recognition
   ```

2. **Cấu hình environment variables**
   ```bash
   # Copy và chỉnh sửa environment file
   cp app/.env.example app/.env
   # Edit app/.env với cấu hình của bạn
   ```

3. **Khởi động tất cả services**
   ```bash
   # Sử dụng script có sẵn (Recommended - Auto GPU Detection)
   ./start_services.sh
   
   # Hoặc thủ công với Docker Compose
   # GPU version (nếu có NVIDIA GPU)
   docker compose up -d
   
   # CPU version (nếu không có GPU)
   docker compose -f docker-compose-cpu.yml up -d
   ```

4. **Xác nhận cài đặt**
   ```bash
   # Test connectivity
   ./test_integration.sh
   
   # Hoặc kiểm tra thủ công
   curl http://localhost:2024/health/full
   ```

### Manual Installation

1. **Cài đặt Python dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   
   cd ../qdrant_database_FE
   pip install -r requirements.txt
   ```

2. **Khởi động services riêng lẻ**
   ```bash
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant:latest
   
   # Start MinIO
   docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
   
   # Start Database API
   cd qdrant_database_FE && uvicorn app:app --host 0.0.0.0 --port 7005
   
   # Start Face Recognition API
   cd app && python main.py
   ```

## 🐳 Docker Commands

```bash
# Khởi động tất cả services (Auto GPU Detection)
./start_services.sh

# Khởi động manual với GPU support
docker compose up -d

# Khởi động manual CPU-only (không cần GPU)
docker compose -f docker-compose-cpu.yml up -d

# Xem logs
docker compose logs -f

# Dừng tất cả services
docker compose down

# Rebuild và khởi động
docker compose up --build -d
```

## 🔍 Health Monitoring

### Service Health Checks
```bash
# Kiểm tra main API
curl http://localhost:2024/health

# Kiểm tra database connectivity
curl http://localhost:2024/health/database

# Kiểm tra MinIO storage
curl http://localhost:2024/health/minio

# Kiểm tra toàn bộ hệ thống
curl http://localhost:2024/health/full
```

### Integration Testing
```bash
# Chạy integration tests
./test_integration.sh

# Manual integration test
curl -X POST http://localhost:2024/test/basic
```

## 🔧 Development Guide

### Thêm Features Mới

1. **API Endpoints mới**: Thêm vào `src/api/routers/`
2. **Business Logic**: Implement trong `src/services/face_service.py`
3. **Database Operations**: Sử dụng `database_client.py` cho HTTP calls
4. **Image Processing**: Thêm vào `image_processor.py`
5. **Data Models**: Định nghĩa trong `src/core/models.py`

### Database Operations

Tất cả database operations đi qua HTTP API:
```python
# Example usage
database_client = DatabaseClient(host, port)
collections = await database_client.get_collections()
points = await database_client.search_face(collection, embedding, limit)
```

### Testing & Debugging

```bash
# Xem logs
tail -f app/logs/face.log

# Kiểm tra service status
docker compose ps

# Debug container
docker compose exec api_fr bash

# Test specific endpoint
curl -X POST http://localhost:2024/create_face_img_base64 \
  -H "Content-Type: application/json" \
  -d '{"img_base64":"...","id":"test","name":"Test","role":"1","store_id":"123"}'
```

## 📊 Performance Considerations

- **⚡ Async Processing**: Tất cả I/O operations đều asynchronous
- **🔄 Connection Pooling**: HTTP connection reuse cho database calls
- **🧵 Thread Pools**: CPU-intensive operations trong thread pools
- **🧹 Memory Management**: Cleanup explicit cho large image objects
- **📦 Batch Processing**: Xử lý hiệu quả cho multiple operations

## 🔒 Security Features

- **✅ Input Validation**: Pydantic models cho request validation
- **🛡️ Error Sanitization**: Safe error messages không có sensitive data
- **🔐 Access Control**: Collection-based access patterns
- **🏢 Data Isolation**: Store-based data separation

## 📈 Monitoring & Logging

- **📊 Structured Logging**: JSON-formatted logs với context
- **⏱️ Performance Metrics**: Timing information cho tất cả operations
- **❤️ Health Endpoints**: Multi-level health checking
- **🚨 Error Tracking**: Comprehensive error logging và tracking

## 🏛️ Architecture Benefits

### Microservices Design
1. **🔄 Service Separation**: Database và API services được tách biệt
2. **📈 Scalability**: Mỗi service có thể scale độc lập
3. **🛡️ Reliability**: Service failures được cô lập
4. **🔧 Maintainability**: Service boundaries và responsibilities rõ ràng

### Clean Architecture
1. **📚 Layered Structure**: API → Services → Utils → Data
2. **🔄 Dependency Inversion**: Business logic độc lập với external services
3. **🧪 Testability**: Mỗi layer có thể test riêng biệt
4. **🚀 Extensibility**: Dễ dàng thêm features và endpoints mới

### HTTP-based Database Layer
1. **🔧 Technology Agnostic**: Database implementation có thể thay đổi
2. **🌐 Network Resilience**: HTTP retry mechanisms và error handling
3. **🔍 Service Discovery**: Database service có thể deploy anywhere
4. **📖 API Documentation**: Database operations tự document

## 🔧 Troubleshooting

### Common Issues

1. **Service không khởi động được**
   ```bash
   # Kiểm tra ports có bị occupied
   netstat -tulpn | grep -E ':(2024|6333|7005|9000)'
   
   # Kiểm tra Docker logs
   docker compose logs api_fr
   docker compose logs api_db
   ```

2. **Database connection lỗi**
   ```bash
   # Kiểm tra Qdrant service
   curl http://localhost:6333
   
   # Kiểm tra Database API
   curl http://localhost:7005/get_collections
   ```

3. **Face detection không hoạt động**
   ```bash
   # Kiểm tra GPU support
   nvidia-smi
   
   # Nếu GPU có vấn đề, force sử dụng CPU mode
   docker compose down
   docker compose -f docker-compose-cpu.yml up -d
   
   # Hoặc restart với script (auto-detect lại)
   ./start_services.sh
   ```

## 📚 Documentation Links

- **API Documentation**: http://localhost:2024/docs
- **Database API**: http://localhost:7005/docs
- **MinIO Console**: http://localhost:9001
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

**🏗️ Built with ❤️ using FastAPI, Docker, Qdrant, MinIO, and YOLOv8**

*For technical support or questions, please refer to the API documentation or health monitoring endpoints.*
