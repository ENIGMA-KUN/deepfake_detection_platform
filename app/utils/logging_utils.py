import logging
import os
import sys
from datetime import datetime
import yaml
from typing import Dict, Any, Optional

# Configure logging constants
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Load configuration file for logging settings
def load_logging_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load logging configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file (optional)
        
    Returns:
        Dictionary containing logging configuration
    """
    default_config = {
        "level": "INFO",
        "format": DEFAULT_LOG_FORMAT,
        "date_format": DEFAULT_DATE_FORMAT,
        "file": {
            "enabled": True,
            "path": "logs",
            "filename": "deepfake_detection_{date}.log",
            "level": "DEBUG"
        },
        "console": {
            "enabled": True,
            "level": "INFO"
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                if 'logging' in config_data:
                    return config_data['logging']
        except Exception as e:
            print(f"Error loading logging config: {e}")
    
    return default_config

# Setup loggers
def setup_logger(name: str, config: Dict[str, Any] = None, log_file: str = None) -> logging.Logger:
    """
    Configure and return a logger with the specified name and settings.
    
    Args:
        name: Name of the logger
        config: Logging configuration dictionary
        log_file: Override default log file location
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = load_logging_config()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level
    level_str = config.get('level', 'INFO')
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Configure formatter
    log_format = config.get('format', DEFAULT_LOG_FORMAT)
    date_format = config.get('date_format', DEFAULT_DATE_FORMAT)
    formatter = logging.Formatter(log_format, date_format)
    
    # Add console handler if enabled
    if config.get('console', {}).get('enabled', True):
        console_level_str = config.get('console', {}).get('level', 'INFO')
        console_level = getattr(logging, console_level_str.upper(), logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if config.get('file', {}).get('enabled', True):
        # Get file configuration
        file_config = config.get('file', {})
        file_level_str = file_config.get('level', 'DEBUG')
        file_level = getattr(logging, file_level_str.upper(), logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        log_dir = file_config.get('path', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log filename
        if log_file is None:
            filename_template = file_config.get('filename', 'deepfake_detection_{date}.log')
            date_str = datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(log_dir, filename_template.format(date=date_str))
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Get application logger
def get_logger(name: str = None, config_path: str = None, log_file: str = None) -> logging.Logger:
    """
    Get a configured logger for the specified module.
    
    Args:
        name: Name for the logger (default: root logger)
        config_path: Path to logging configuration file
        log_file: Override default log file location
        
    Returns:
        Configured logger instance
    """
    if name is None:
        name = 'deepfake_detection'
    
    config = load_logging_config(config_path)
    return setup_logger(name, config, log_file)

# Log with additional context data
def log_with_context(logger: logging.Logger, level: str, message: str, context: Dict[str, Any] = None):
    """
    Log a message with additional context data.
    
    Args:
        logger: Logger instance
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        message: Log message
        context: Additional context data to include in the log
    """
    if context:
        message = f"{message} - Context: {context}"
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)

# Track detection performance
def log_detection_performance(logger: logging.Logger, media_type: str, detector_name: str, 
                           duration: float, success: bool, error: Optional[str] = None):
    """
    Log performance metrics for detection operations.
    
    Args:
        logger: Logger instance
        media_type: Type of media ('image', 'audio', 'video')
        detector_name: Name of the detector used
        duration: Processing time in seconds
        success: Whether detection was successful
        error: Error message if detection failed
    """
    context = {
        'media_type': media_type,
        'detector': detector_name,
        'duration_seconds': round(duration, 3),
        'success': success
    }
    
    if error:
        context['error'] = error
        log_with_context(logger, 'error', f"Detection failed with {detector_name}", context)
    else:
        log_with_context(logger, 'info', f"Detection completed with {detector_name}", context)
