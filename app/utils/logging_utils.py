"""
Logging utilities for the Deepfake Detection Platform.
"""
import os
import logging
import sys
from datetime import datetime
from typing import Union, Optional

def setup_logging(level: Union[str, int] = "INFO", log_file: Optional[str] = None):
    """
    Set up logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, if None, logs will only go to console
    """
    # Create logs directory if log_file is specified
    if log_file is None:
        logs_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs'))
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"deepfake_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Convert string level to numeric level if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Log the initialization
    logging.info(f"Logging initialized at level {logging.getLevelName(level)}")
    if log_file:
        logging.info(f"Log file: {log_file}")

def get_logger(name: str, level: Optional[Union[str, int]] = None):
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Optional logging level override
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    
    return logger

def log_exception(logger, message: str, exc: Exception = None):
    """
    Log an exception with a custom message.
    
    Args:
        logger: Logger instance
        message: Custom message to log with the exception
        exc: Exception to log, or None to use sys.exc_info()
    """
    if exc:
        logger.exception(f"{message}: {str(exc)}")
    else:
        logger.exception(message)
