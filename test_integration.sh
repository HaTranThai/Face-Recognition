#!/bin/bash

# Test Integration Between Services
echo "🧪 Testing Integration Between Services..."
echo "========================================="

# Test if services are running
echo "📊 Checking if services are accessible..."

# Test Qdrant Database
echo -n "• Qdrant Database (6333): "
if curl -s http://localhost:6333 > /dev/null; then
    echo "✅ Running"
else
    echo "❌ Not accessible"
fi

# Test Database API
echo -n "• Database API (7005): "
if curl -s http://localhost:7005 > /dev/null; then
    echo "✅ Running"
else
    echo "❌ Not accessible"
fi

# Test Face Recognition API
echo -n "• Face Recognition API (2024): "
if curl -s http://localhost:2024 > /dev/null; then
    echo "✅ Running"
else
    echo "❌ Not accessible"
fi

echo ""
echo "🔍 Testing API endpoints..."

# Test Database API endpoints
echo "• Testing Database API endpoints:"
echo -n "  - GET /get_collections: "
if curl -s -X GET http://localhost:7005/get_collections > /dev/null; then
    echo "✅ OK"
else
    echo "❌ Failed"
fi

echo -n "  - POST /create_collection: "
if curl -s -X POST http://localhost:7005/create_collection \
    -H "Content-Type: application/json" \
    -d '{"collection_name": "test_collection"}' > /dev/null; then
    echo "✅ OK"
else
    echo "❌ Failed"
fi

# Test Face Recognition API endpoints
echo "• Testing Face Recognition API endpoints:"
echo -n "  - GET /: "
if curl -s http://localhost:2024/ > /dev/null; then
    echo "✅ OK"
else
    echo "❌ Failed"
fi

echo -n "  - GET /minio/list_buckets: "
if curl -s http://localhost:2024/minio/list_buckets > /dev/null; then
    echo "✅ OK"
else
    echo "❌ Failed"
fi

echo ""
echo "📖 View API Documentation:"
echo "========================="
echo "• Database API:        http://localhost:7005/docs"
echo "• Face Recognition:    http://localhost:2024/docs"
echo "• MinIO APIs:          http://localhost:2024/docs#/MinIO"

echo ""
echo "🗄️ MinIO Management:"
echo "===================="
echo "• MinIO Console:       http://localhost:9001"
echo "• Test MinIO APIs:     ./test_minio_api.sh"

echo ""
echo "✅ Integration test completed!"
