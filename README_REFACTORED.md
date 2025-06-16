# Face Recognition System - Refactored Architecture

## 🏗️ Architecture Overview

This system has been refactored to use a microservices architecture:

```
┌─────────────────┐    HTTP API    ┌──────────────────┐    Direct    ┌─────────────┐
│ Face Recognition│───────────────→│ qdrant_database_ │─────────────→│   Qdrant    │
│      App        │    (httpx)     │       FE         │   (client)   │  Database   │
│   (Port 2024)   │                │   (Port 7005)    │              │ (Port 6333) │
└─────────────────┘                └──────────────────┘              └─────────────┘
```

## 🚀 Quick Start

### 1. Start All Services

```bash
./start_services.sh
```

This will start:
- **Qdrant Database** (localhost:6333)
- **Database API Service** (localhost:7005) 
- **Face Recognition API** (localhost:2024)
- **MinIO Storage** (localhost:9000)

### 2. Test Integration

```bash
./test_integration.sh
```

### 3. View API Documentation

- Database API: http://localhost:7005/docs
- Face Recognition API: http://localhost:2024/docs

## 📋 Services Description

### 1. Qdrant Database (Port 6333)
- Vector database for storing face embeddings
- Direct access from Database API service only

### 2. Database API Service (Port 7005)
- Handles all database operations
- Provides REST API for CRUD operations
- Located in `qdrant_database_FE/`

**Available Endpoints:**
- `GET /get_collections` - List all collections
- `POST /create_collection` - Create new collection
- `POST /insert_point` - Insert face embedding
- `POST /search_point` - Search similar faces
- `DELETE /delete_point` - Delete face by ID
- `GET /create_snapshot/{collection}` - Create snapshot
- `POST /recover_snapshot` - Recover from snapshot

### 3. Face Recognition API (Port 2024)
- Main application handling face detection and recognition
- Calls Database API for all database operations
- Located in `app/`

**Available Endpoints:**
- `POST /create_face` - Register new face
- `POST /face_recog` - Recognize face
- `DELETE /delete_face` - Delete employee face
- `POST /batch_customers` - Batch process customers
- `GET /backup_data/{store_id}` - Backup store data
- `GET /backup_all` - Backup all data
- `POST /recover_db` - Recover from backup

## 🛠️ Development

### Environment Variables

**Face Recognition App (.env):**
```env
HOST=0.0.0.0
PORT=8000
DEBUG=false
QDRANT_DB_HOST=localhost
QDRANT_DB_PORT=7005
```

**Database API (.env):**
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
HOST=0.0.0.0
PORT=7005
```

### Manual Development Setup

1. **Start Qdrant Database:**
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

2. **Start Database API:**
```bash
cd qdrant_database_FE
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7005
```

3. **Start Face Recognition API:**
```bash
cd app
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 2024
```

## 🐳 Docker Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Rebuild and start
docker compose up --build -d
```

## 📁 Project Structure

```
CMD_Face_Recognition/
├── app/                          # Face Recognition API
│   ├── src/
│   │   ├── api/                 # FastAPI routes and app
│   │   ├── services/            # Business logic
│   │   ├── utils/               # Utilities (DatabaseClient, etc.)
│   │   └── core/                # Models and schemas
│   ├── config/                  # Configuration
│   ├── main.py                  # App entry point
│   └── requirements.txt
├── qdrant_database_FE/          # Database API Service
│   ├── app.py                   # FastAPI app for DB operations
│   └── requirements.txt
├── docker-compose.yml           # Docker services configuration
├── start_services.sh           # Start all services script
└── test_integration.sh         # Integration test script
```

## 🔧 Key Changes in Refactor

1. **Separated Database Logic:** All direct Qdrant operations moved to `qdrant_database_FE` service
2. **API-First Approach:** Face Recognition app only communicates via HTTP API
3. **Improved Modularity:** Each service has single responsibility
4. **Better Error Handling:** Centralized database error handling in Database API
5. **Scalability:** Services can be scaled independently

## 🧪 Testing

The system includes comprehensive testing scripts to verify:
- Service connectivity
- API endpoint availability
- Integration between services
- Database operations

## 📝 Notes

- All database operations now go through the Database API service
- Face Recognition app no longer has direct database dependencies
- Backward compatibility maintained for all existing endpoints
- Configuration simplified with environment variables
