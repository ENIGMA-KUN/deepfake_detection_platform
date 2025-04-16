"""
Main entry point for the Deepfake Detection Platform.
"""
import os
import sys
import logging
import yaml
from typing import Dict, Any

# Add the project root to the path to enable absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.interface.app import create_app
from app.core.processor import MediaProcessor
from app.utils.logging_utils import setup_logging

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from the config file.
    
    Args:
        config_path: Path to the config file, uses default if None
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the config file is not found
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    return config

def main():
    """
    Main function to initialize and run the Deepfake Detection Platform.
    """
    # Load configuration
    config = load_config()
    
    # Setup logging
    log_level = config['general'].get('log_level', 'INFO')
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Deepfake Detection Platform")
    
    # Initialize the media processor
    processor = MediaProcessor(config)
    
    # Create and run the web application
    app = create_app(processor, config)
    
    # Run the app (using the newer Dash API)
    app.run(
        debug=config['general'].get('debug_mode', False),
        host='0.0.0.0',
        port=8050
    )

if __name__ == "__main__":
    main()
