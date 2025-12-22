"""
Main entry point for the Face Recognition API application.
"""
import uvicorn
from src.api.app import create_app
from config.settings import get_settings

settings = get_settings()
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=2024,
        reload=False,
        log_level="info"
    )
