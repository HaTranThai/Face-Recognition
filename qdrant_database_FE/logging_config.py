"""
Database logging configuration for qdrant_database_FE service.
"""
import logging
import logging.handlers
import os
from datetime import datetime

def setup_database_logging():
    """Setup database service logging configuration."""
    # Ensure logs directory exists
    logs_dir = "../logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Common log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # ==== DATABASE SERVICE LOGGER ====
    database_logger = logging.getLogger("database")
    database_logger.setLevel(logging.INFO)
    database_logger.propagate = False
    
    database_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, "database.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    database_handler.setFormatter(formatter)
    database_logger.addHandler(database_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    database_logger.addHandler(console_handler)
    
    database_logger.info("Database service logging initialized")
    return database_logger

def get_database_logger() -> logging.Logger:
    """Get database logger instance."""
    return logging.getLogger("database")
