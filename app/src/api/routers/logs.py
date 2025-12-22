"""Log management routes."""
import os
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from config.settings import get_settings
from config.logging import get_app_logger

logger = get_app_logger()
router = APIRouter(tags=["Logs"])

settings = get_settings()


@router.get("/logs/list")
async def list_log_files():
    """List available log files."""
    try:
        logs_dir = settings.LOGS_PATH
        if not os.path.exists(logs_dir):
            return JSONResponse(content={
                "status": 1,
                "message": "Logs directory not found",
                "files": []
            })
        
        log_files = []
        for file in os.listdir(logs_dir):
            if file.endswith('.log'):
                file_path = os.path.join(logs_dir, file)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    log_files.append({
                        "name": file,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": stat.st_mtime
                    })
        
        return JSONResponse(content={
            "status": 0,
            "message": "Log files retrieved successfully",
            "files": sorted(log_files, key=lambda x: x['modified'], reverse=True)
        })
        
    except Exception as e:
        logger.error(f"Failed to list log files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 2,
                "message": f"Failed to list log files: {str(e)}"
            }
        )


@router.get("/logs/{log_type}")
async def get_log_content(
    log_type: str,
    lines: Optional[int] = Query(100, description="Number of lines to read from the end"),
    format: Optional[str] = Query("json", description="Response format: json or text")
):
    """
    Get log content for specific log type.
    
    log_type: face, database, storage, app
    lines: Number of lines to read from the end of file
    format: Response format (json or text)
    """
    try:
        # Validate log type
        valid_types = ["face", "database", "storage", "app"]
        if log_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log type. Must be one of: {', '.join(valid_types)}"
            )
        
        log_file = os.path.join(settings.LOGS_PATH, f"{log_type}.log")
        
        if not os.path.exists(log_file):
            return JSONResponse(content={
                "status": 1,
                "message": f"Log file {log_type}.log not found",
                "content": []
            })
        
        # Read last N lines
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # Get last N lines
        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        if format == "text":
            return PlainTextResponse(
                content=''.join(last_lines),
                media_type="text/plain"
            )
        else:
            return JSONResponse(content={
                "status": 0,
                "log_type": log_type,
                "lines_requested": lines,
                "lines_returned": len(last_lines),
                "total_lines": len(all_lines),
                "content": [line.strip() for line in last_lines if line.strip()]
            })
            
    except Exception as e:
        logger.error(f"Failed to read log file {log_type}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 2,
                "message": f"Failed to read log file: {str(e)}"
            }
        )


@router.delete("/logs/{log_type}")
async def clear_log_file(log_type: str):
    """Clear/truncate a specific log file."""
    try:
        # Validate log type
        valid_types = ["face", "database", "storage", "app"]
        if log_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log type. Must be one of: {', '.join(valid_types)}"
            )
        
        log_file = os.path.join(settings.LOGS_PATH, f"{log_type}.log")
        
        if os.path.exists(log_file):
            # Truncate the file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("")
            
            logger.info(f"Log file {log_type}.log cleared successfully")
            return JSONResponse(content={
                "status": 0,
                "message": f"Log file {log_type}.log cleared successfully"
            })
        else:
            return JSONResponse(content={
                "status": 1,
                "message": f"Log file {log_type}.log not found"
            })
            
    except Exception as e:
        logger.error(f"Failed to clear log file {log_type}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 2,
                "message": f"Failed to clear log file: {str(e)}"
            }
        )
