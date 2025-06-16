"""
Logging configuration for the application.
"""
import logging
import os
from datetime import datetime
from config.settings import get_settings

settings = get_settings()


def setup_logging():
    """Setup logging configuration."""
    # Ensure logs directory exists
    os.makedirs(settings.LOGS_PATH, exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=logging.INFO if not settings.DEBUG else logging.DEBUG,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(
                os.path.join(settings.LOGS_PATH, "face.log"), 
                mode="a",
                encoding="utf-8"
            ),
            # Uncomment for console output
            # logging.StreamHandler()
        ]
    )
    
    # Get logger for the application
    logger = logging.getLogger("face_recognition")
    logger.info("Logging setup completed")
    
    return logger


def get_logger(name: str = "face_recognition") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
