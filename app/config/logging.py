"""
Logging configuration for the application.
"""
import logging
import logging.handlers
import os
from datetime import datetime
from config.settings import get_settings

settings = get_settings()

# Global flag to prevent multiple logging setup
_logging_setup_done = False


def setup_logging():
    """Setup categorized logging configuration for face, database, and minio operations."""
    global _logging_setup_done
    
    # If logging is already setup, return early to prevent duplicates
    if _logging_setup_done:
        return logging.getLogger("app")
    
    # Ensure logs directory exists
    os.makedirs(settings.LOGS_PATH, exist_ok=True)
    
    # Common log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Clear any existing handlers to avoid duplicates
    logging.getLogger().handlers.clear()
    
    # Clear handlers for all category loggers to prevent duplicates
    for category in ["face", "database", "minio", "app"]:
        category_logger = logging.getLogger(category)
        category_logger.handlers.clear()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    
    # Create formatters
    formatter = logging.Formatter(log_format, date_format)
    
    # ==== FACE RECOGNITION LOGGER ====
    face_logger = logging.getLogger("face")
    face_logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    face_logger.propagate = False
    
    face_handler = logging.handlers.RotatingFileHandler(
        os.path.join(settings.LOGS_PATH, "face.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    face_handler.setFormatter(formatter)
    face_logger.addHandler(face_handler)
    
    # ==== DATABASE LOGGER ====
    database_logger = logging.getLogger("database")
    database_logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    database_logger.propagate = False
    
    database_handler = logging.handlers.RotatingFileHandler(
        os.path.join(settings.LOGS_PATH, "database.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    database_handler.setFormatter(formatter)
    database_logger.addHandler(database_handler)
    
    # ==== MINIO LOGGER ====
    minio_logger = logging.getLogger("minio")
    minio_logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    minio_logger.propagate = False
    
    minio_handler = logging.handlers.RotatingFileHandler(
        os.path.join(settings.LOGS_PATH, "minio.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    minio_handler.setFormatter(formatter)
    minio_logger.addHandler(minio_handler)
    
    # ==== GENERAL APPLICATION LOGGER ====
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    app_logger.propagate = False
    
    app_handler = logging.handlers.RotatingFileHandler(
        os.path.join(settings.LOGS_PATH, "app.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    app_handler.setFormatter(formatter)
    app_logger.addHandler(app_handler)
    
    # Add console handler for development (optional)
    if settings.DEBUG:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        face_logger.addHandler(console_handler)
        database_logger.addHandler(console_handler)
        minio_logger.addHandler(console_handler)
        app_logger.addHandler(console_handler)
    
    # Log successful setup
    app_logger.info("Categorized logging setup completed")
    face_logger.info("Face recognition logging initialized")
    database_logger.info("Database logging initialized")
    minio_logger.info("MinIO logging initialized")
    
    # Mark logging as setup to prevent future duplicate setups
    _logging_setup_done = True
    
    return app_logger


def get_logger(category: str = "app") -> logging.Logger:
    """
    Get logger instance for specific category.
    
    Args:
        category: Logger category ('face', 'database', 'minio', 'app')
    
    Returns:
        Logger instance for the specified category
    """
    valid_categories = ["face", "database", "minio", "app"]
    if category not in valid_categories:
        category = "app"
    
    return logging.getLogger(category)


def get_face_logger() -> logging.Logger:
    """Get face recognition specific logger."""
    return logging.getLogger("face")


def get_database_logger() -> logging.Logger:
    """Get database operations specific logger."""
    return logging.getLogger("database")


def get_minio_logger() -> logging.Logger:
    """Get MinIO operations specific logger."""
    return logging.getLogger("minio")


def get_app_logger() -> logging.Logger:
    """Get general application logger."""
    return logging.getLogger("app")


def reset_logging_setup():
    """Reset the logging setup flag. Useful for testing or reinitialization."""
    global _logging_setup_done
    _logging_setup_done = False
