"""
FastAPI application factory and configuration.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from config.logging import setup_logging, get_app_logger
from .routes import router

settings = get_settings()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Setup logging first
    setup_logging()
    logger = get_app_logger()
    logger.info("Starting Face Recognition API application")
    
    # Create FastAPI instance
    app = FastAPI(
        title="FACE API",
        description="API for FACE",
        version="1.0",
        debug=settings.DEBUG,
        openapi_tags=[
            {
                "name": "Face",
                "description": "APIs for Face"
            },
            {
                "name": "Database",
                "description": "APIs for Database"
            },
            {
                "name": "MinIO",
                "description": "APIs for MinIO Storage Operations - Backup, Restore, and Sync"
            }
        ]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=settings.STATIC_PATH), name="static")
    
    # Include API routes
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        try:
            import os
            import cv2
            from deepface import DeepFace
            import gc
            
            # Create logs directory if not exists
            os.makedirs("./logs", exist_ok=True)
            
            logger.info("Performing startup connection checks...")
            
            # Init service tạm để check
            from src.services.face_service import FaceService
            svc = FaceService(settings)
            
            # Check Qdrant
            try:
                col = await svc.database_client.get_collections()
                if isinstance(col, list):
                    logger.info("✅ Startup Check: Qdrant CONNECTED")
                else:
                    logger.error("❌ Startup Check: Qdrant FAILED")
            except Exception as e:
                logger.error(f"❌ Startup Check: Qdrant ERROR - {e}")

            # Check MinIO
            try:
                svc.image_processor._get_s3_client().list_buckets()
                logger.info("✅ Startup Check: MinIO CONNECTED")
            except Exception as e:
                logger.error(f"❌ Startup Check: MinIO ERROR - {e}")

            test_image_path = './static/images/testface.jpg' 
            
            if os.path.exists(test_image_path):
                logger.info(f"Found test image at {test_image_path}. Warming up AI models...")
                
                image = cv2.imread(test_image_path)
                
                logger.info("Loading Detection Model...")
                DeepFace.extract_faces(
                    img_path=image,
                    detector_backend="yolov8",
                    align=True,
                    enforce_detection=False 
                )
                
                logger.info("Loading Recognition Model (VGG-Face)...")
                DeepFace.represent(
                    img_path=image,
                    model_name="VGG-Face",
                    detector_backend="skip", 
                    align=True,
                    normalization="VGGFace2",
                )
                
                logger.info("✅ All AI Models warmed up successfully! Ready to serve requests fast.")
            else:
                logger.warning(f"⚠️ Test image NOT found at {test_image_path}. Cold start issues will occur!")
                logger.warning(f"Current working directory: {os.getcwd()}")
                
        except Exception as e:
            logger.error(f"Error during startup warm-up: {str(e)}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup resources when the application shuts down."""
        try:
            import gc
            # Force garbage collection
            gc.collect()
            logger.info("Application shutdown successfully, resources cleaned up")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app
