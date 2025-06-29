"""
Logging configuration for the Multimodal Consumer Segmentation Project
"""

import sys
from pathlib import Path
from loguru import logger
from config import LOG_LEVEL, PROJECT_ROOT


def setup_logging():
    """Configure logging for the entire project"""
    
    # Remove default logger
    logger.remove()
    
    # Create logs directory
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Console logging with colors
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging - general log
    logger.add(
        log_dir / "application.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    # File logging - error log
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="5 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Data processing specific log
    logger.add(
        log_dir / "data_processing.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: "data" in record["name"].lower(),
        rotation="5 MB",
        retention="14 days"
    )
    
    logger.info("Logging system initialized")


# Initialize logging when module is imported
setup_logging()