# Face Recognition API

A production-ready face recognition system built with FastAPI, featuring microservices architecture with separate API and database services.

## 🏗️ Architecture Overview

This project consists of two main services:
- **Face Recognition API** (`api_fr`): Main application handling face detection, recognition, and image processing
- **Database API** (`api_db`): Dedicated service for Qdrant vector database operations
- **External Services**: MinIO for object storage, Qdrant for vector database

## 📁 Project Structure

```
CMD_Face_Recognition/
├── docker-compose.yml          # Multi-service Docker configuration
├── start_services.sh           # Service startup script
├── test_integration.sh         # Integration testing script
├── README_REFACTORED.md        # Refactoring documentation
│
├── app/                        # Main Face Recognition API
│   ├── main.py                 # FastAPI application entry point
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile             # API service Docker config
│   ├── .env                   # Environment variables
│   │
│   ├── config/                # Configuration management
│   │   ├── settings.py        # Application settings & env vars
│   │   └── logging.py         # Logging configuration
│   │
│   ├── src/                   # Source code (Clean Architecture)
│   │   ├── api/               # API layer
│   │   │   ├── app.py         # FastAPI factory & middleware
│   │   │   └── routes.py      # API endpoints & health checks
│   │   │
│   │   ├── core/              # Domain models
│   │   │   └── models.py      # Pydantic models (CreateFace, FaceRecog, etc.)
│   │   │
│   │   ├── services/          # Business logic layer
│   │   │   └── face_service.py # Face recognition business logic
│   │   │
│   │   └── utils/             # Utility modules
│   │       ├── image_processor.py   # MinIO/S3 image storage
│   │       ├── database_client.py   # HTTP client for database API
│   │       └── legacy.py           # Legacy face detection functions
│   │
│   ├── models/                # ML models & YOLO implementation
│   │   ├── yolo.py           # YOLOv8 face detection model
│   │   ├── yolov8n-face.onnx # YOLO face detection weights
│   │   └── best_face_mask.pt # Face mask detection weights
│   │
│   └── logs/                 # Application logs
│
├── qdrant_database_FE/        # Database API Service
│   ├── app.py                # Database service endpoints
│   ├── requirements.txt      # Database service dependencies
│   └── Dockerfile           # Database service Docker config
│
├── qdrant_storage/           # Persistent Qdrant data
│   └── collections/          # Vector collections data
│
├── snapshots/                # Qdrant snapshots & backups
│   └── {collection_name}/    # Collection-specific snapshots
│
└── data/                     # MinIO data storage
    └── miniodata/            # Persistent MinIO data
        ├── data-face-checkin-customer-images/
        ├── data-face-checkin-employee-images/
        ├── data-face-register-customer-images/
        └── data-face-register-employee-images/
```

## 🚀 Features

### Core Functionality
- **Face Detection**: YOLOv8-based face detection with high accuracy
- **Face Recognition**: Deep learning face embeddings with similarity matching
- **Face Registration**: Register faces to customer/employee collections
- **Face Check-in**: Real-time face verification for access control
- **Quality Assessment**: Face quality, blur detection, eye state validation
- **Anti-spoofing**: Basic liveness detection capabilities

### Technical Features
- **Microservices Architecture**: Separate API and database services
- **HTTP-based Database Layer**: All database operations via REST API
- **Object Storage**: MinIO/S3 compatible image storage
- **Health Monitoring**: Comprehensive health checks for all services
- **Docker Support**: Full containerization with Docker Compose
- **Async Processing**: Asynchronous image processing and database operations
- **Error Handling**: Comprehensive logging and error management
- **Configuration Management**: Environment-based configuration

### API Capabilities
- **Multiple Collections**: Organize faces by store/role (customers/employees)
- **Batch Operations**: Handle multiple face operations efficiently
- **Snapshot Management**: Database backup and restore functionality
- **Real-time Processing**: Fast face detection and recognition
- **RESTful Design**: Clean, documented API endpoints

## 🛠️ Installation & Setup

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU support (for face detection acceleration)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CMD_Face_Recognition
   ```

2. **Configure environment variables**
   ```bash
   # Copy and edit environment file
   cp app/.env.example app/.env
   # Edit app/.env with your configuration
   ```

3. **Start all services**
   ```bash
   # Using the provided script
   ./start_services.sh
   
   # Or manually with Docker Compose
   docker compose up -d
   ```

4. **Verify installation**
   ```bash
   # Test service connectivity
   ./test_integration.sh
   
   # Or manually check health
   curl http://localhost:2024/health/full
   ```

### Manual Installation

1. **Install Python dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

2. **Install database service dependencies**
   ```bash
   cd qdrant_database_FE
   pip install -r requirements.txt
   ```

3. **Start services individually**
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

## 🔌 API Endpoints

### Health & Monitoring
- `GET /health` - Basic API health check
- `GET /health/database` - Database connectivity check
- `GET /health/minio` - MinIO storage connectivity check
- `GET /health/full` - Complete system health check
- `GET /test/basic` - Basic functionality test

### Face Recognition
- `POST /create_face_img_base64` - Register face from base64 image
- `POST /recognize_face` - Recognize/verify face
- `POST /create_face_img_file` - Register face from uploaded file
- `POST /recognize_face_img_file` - Recognize face from uploaded file

### Database Management
- `GET /get_list_collections` - List all face collections
- `DELETE /delete_collection/{collection_name}` - Delete a collection
- `POST /create_snapshot/{collection_name}` - Create collection snapshot
- `POST /recover_from_snapshot` - Restore from snapshot

### Utility Endpoints
- `GET /get_face_img/{store_id}/{face_id}` - Retrieve stored face image
- `GET /download_snapshot_zip/{store_id}` - Download collection backup

## ⚙️ Configuration

### Environment Variables

**Main API Service (.env)**:
```bash
# Database API Configuration
QDRANT_DB_HOST=localhost          # Database API host
QDRANT_DB_PORT=7005              # Database API port

# Docker Environment
DOCKER_ENV=false                 # Set to true when running in Docker

# MinIO Configuration  
MINIO_ENDPOINT=localhost:9000    # MinIO endpoint
MINIO_ACCESS_KEY=minioadmin      # MinIO access key
MINIO_SECRET_KEY=minioadmin1245  # MinIO secret key

# Face Detection Settings
CONF_THRESHOLD=0.7               # Face detection confidence threshold
BLUR_THRESHOLD=100               # Blur detection threshold
FACE_EXT=0.3                     # Face extraction expansion percentage

# Storage Paths (MinIO buckets)
CHECKIN_CUSTOMER_PATH=data-face-checkin-customer-images
CHECKIN_EMPLOYEE_PATH=data-face-checkin-employee-images
REGISTER_CUSTOMER_PATH=data-face-register-customer-images
REGISTER_EMPLOYEE_PATH=data-face-register-employee-images

# Model Paths
MODELS_PATH=models               # ML models directory
```

**Docker Compose Configuration**:
```yaml
services:
  qdrant:        # Vector database
  api_db:        # Database API service  
  api_fr:        # Face recognition API
  minio:         # Object storage
```

## 🔍 Health Monitoring

The system provides comprehensive health monitoring:

### Service Health Checks
```bash
# Check main API
curl http://localhost:2024/health

# Check database connectivity
curl http://localhost:2024/health/database

# Check MinIO storage
curl http://localhost:2024/health/minio

# Complete system check
curl http://localhost:2024/health/full
```

### Integration Testing
```bash
# Run integration tests
./test_integration.sh

# Manual integration test
curl -X POST http://localhost:2024/test/basic
```

## 🏛️ Architecture Benefits

### Microservices Design
1. **Service Separation**: Database and API services are decoupled
2. **Scalability**: Each service can be scaled independently
3. **Reliability**: Service failures are isolated
4. **Maintainability**: Clear service boundaries and responsibilities

### Clean Architecture
1. **Layered Structure**: API → Services → Utils → Data
2. **Dependency Inversion**: Business logic independent of external services
3. **Testability**: Each layer can be tested in isolation
4. **Extensibility**: Easy to add new features and endpoints

### HTTP-based Database Layer
1. **Technology Agnostic**: Database implementation can be changed
2. **Network Resilience**: HTTP retry mechanisms and error handling
3. **Service Discovery**: Database service can be deployed anywhere
4. **API Documentation**: Database operations are self-documenting

## 🔧 Development Guide

### Adding New Features

1. **New API Endpoints**: Add to `src/api/routes.py`
2. **Business Logic**: Implement in `src/services/face_service.py`
3. **Database Operations**: Use `database_client.py` for HTTP calls
4. **Image Processing**: Add to `image_processor.py`
5. **Data Models**: Define in `src/core/models.py`

### Database Operations

All database operations go through the HTTP API:
```python
# Example usage
database_client = DatabaseClient(host, port)
collections = await database_client.get_collections()
points = await database_client.search_face(collection, embedding, limit)
```

### Testing

```bash
# Test API endpoints
python app/test_api.py

# Test Docker services
docker compose ps
docker compose logs api_fr
```

### Debugging

```bash
# View logs
tail -f app/logs/face.log

# Check service status
docker compose ps

# Debug container
docker compose exec api_fr bash
```

## 📊 Performance Considerations

- **Async Processing**: All I/O operations are asynchronous
- **Connection Pooling**: HTTP connection reuse for database calls
- **Thread Pools**: CPU-intensive operations in thread pools
- **Memory Management**: Explicit cleanup of large image objects
- **Batch Processing**: Efficient handling of multiple operations

## 🔒 Security Features

- **Input Validation**: Pydantic models for request validation
- **Error Sanitization**: Safe error messages without sensitive data
- **Access Control**: Collection-based access patterns
- **Data Isolation**: Store-based data separation

## 📈 Monitoring & Logging

- **Structured Logging**: JSON-formatted logs with context
- **Performance Metrics**: Timing information for all operations
- **Health Endpoints**: Multi-level health checking
- **Error Tracking**: Comprehensive error logging and tracking

---

**Built with ❤️ using FastAPI, Docker, Qdrant, and MinIO**
