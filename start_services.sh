#!/bin/bash

# Start Face Recognition Services Script
echo "🚀 Starting Face Recognition Services..."
echo "======================================"

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down

# Build and start all services
echo "🔨 Building and starting services..."
# docker compose up --build -d

# Kiểm tra GPU bằng nvidia-smi
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "✅ GPU detected. Using docker-compose.yml (GPU version)."
    docker compose -f docker-compose.yml up --build -d
else
    echo "⚠️ No GPU detected. Using docker-compose-cpu.yml (CPU version)."
    docker compose -f docker-compose-cpu.yml up --build -d
fi


# Check status
echo "📊 Checking service status..."
sleep 5

echo ""
echo "📋 Service Status:"
echo "=================="
docker compose ps

echo ""
echo "🌐 Available Services:"
echo "====================="
echo "• Qdrant Database:     http://localhost:6333"
echo "• Database API:        http://localhost:7005"
echo "• Face Recognition:    http://localhost:2024"
echo "• MinIO Storage:       http://localhost:9000"
echo "• MinIO Console:       http://localhost:9001"

echo ""
echo "📖 API Documentation:"
echo "===================="
echo "• Database API Docs:   http://localhost:7005/docs"
echo "• Face API Docs:       http://localhost:2024/docs"

echo ""
echo "✅ All services started successfully!"
echo "📝 Use 'docker compose logs -f [service_name]' to view logs"
echo "🛑 Use 'docker compose down' to stop all services"
