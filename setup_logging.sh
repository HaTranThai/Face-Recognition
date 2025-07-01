#!/bin/bash

# Quick setup script for categorized logging system

echo "🎯 Setting up Categorized Logging System for Face Recognition"
echo "=============================================================="

# Create logs directory if it doesn't exist
mkdir -p logs
echo "✅ Created logs directory"

# Make scripts executable
chmod +x app/scripts/logs.sh
chmod +x test_categorized_logging.py
echo "✅ Made scripts executable"

# Test the logging system
echo ""
echo "🧪 Testing logging system..."
cd app && python3 -c "
import sys
sys.path.append('.')
from config.logging import setup_logging, get_face_logger, get_database_logger, get_minio_logger, get_app_logger

# Setup logging
setup_logging()

# Test each logger
face_logger = get_face_logger()
face_logger.info('Face recognition logging test - SUCCESS')

db_logger = get_database_logger()
db_logger.info('Database logging test - SUCCESS')

minio_logger = get_minio_logger()
minio_logger.info('MinIO logging test - SUCCESS')

app_logger = get_app_logger()
app_logger.info('Application logging test - SUCCESS')

print('✅ All loggers working correctly!')
"

cd ..

echo ""
echo "📊 Checking created log files..."
ls -la logs/

echo ""
echo "✅ Categorized Logging System Setup Complete!"
echo ""
echo "📝 Available Commands:"
echo "  ./app/scripts/logs.sh list          - List all log files"
echo "  ./app/scripts/logs.sh tail face     - View face logs"
echo "  ./app/scripts/logs.sh follow minio  - Follow MinIO logs realtime"
echo "  ./app/scripts/logs.sh stats         - Show log statistics"
echo ""
echo "📖 Full documentation: CATEGORIZED_LOGGING_GUIDE.md"
