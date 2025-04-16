#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core components
from app.core.workflow import DeepfakeDetectionWorkflow
from app.utils.logging_utils import get_logger
from app.interface.app import launch_interface
from models.model_loader import ensure_models_downloaded
from detectors.detector_factory import get_available_detectors

# Configure logger
logger = get_logger('main')

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Deepfake Detection Platform")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(project_root, "config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gui", "cli"],
        default="gui",
        help="Run in GUI or CLI mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory (for CLI mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/output",
        help="Output directory for reports (for CLI mode)"
    )
    parser.add_argument(
        "--media-type",
        type=str,
        choices=["image", "audio", "video", "auto"],
        default="auto",
        help="Media type to analyze (for CLI mode). 'auto' will detect based on file extension."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in input directory (for CLI mode)"
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download models and exit"
    )
    parser.add_argument(
        "--list-detectors",
        action="store_true",
        help="List available detectors and exit"
    )
    
    return parser.parse_args()

def detect_media_type(file_path: str) -> str:
    """
    Detect media type based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Media type ('image', 'audio', 'video')
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    if extension in image_extensions:
        return "image"
    elif extension in audio_extensions:
        return "audio"
    elif extension in video_extensions:
        return "video"
    else:
        logger.warning(f"Unknown file extension: {extension}, defaulting to 'image'")
        return "image"

def process_file(file_path: str, output_dir: str, media_type: str, config: Dict[str, Any]) -> bool:
    """
    Process a single file with the deepfake detection workflow.
    
    Args:
        file_path: Path to the input file
        output_dir: Output directory for reports
        media_type: Media type ('image', 'audio', 'video', 'auto')
        config: Configuration dictionary
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Auto-detect media type if needed
        if media_type == "auto":
            media_type = detect_media_type(file_path)
        
        logger.info(f"Processing {media_type} file: {file_path}")
        
        # Initialize workflow
        workflow = DeepfakeDetectionWorkflow(config)
        
        # Process file
        result = workflow.process_file(file_path, media_type)
        
        # Generate report
        report_path = workflow.generate_report(result, output_dir)
        logger.info(f"Report generated: {report_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return False

def process_batch(directory: str, output_dir: str, media_type: str, config: Dict[str, Any]) -> Dict[str, int]:
    """
    Process all supported files in a directory.
    
    Args:
        directory: Input directory
        output_dir: Output directory for reports
        media_type: Media type filter ('image', 'audio', 'video', 'auto')
        config: Configuration dictionary
        
    Returns:
        Dictionary with counts of processed files
    """
    if not os.path.isdir(directory):
        logger.error(f"Input directory does not exist: {directory}")
        return {"total": 0, "success": 0, "failure": 0}
    
    # Get supported file extensions
    extensions = []
    if media_type == "image" or media_type == "auto":
        extensions.extend([".jpg", ".jpeg", ".png", ".bmp", ".gif"])
    if media_type == "audio" or media_type == "auto":
        extensions.extend([".wav", ".mp3", ".flac", ".ogg"])
    if media_type == "video" or media_type == "auto":
        extensions.extend([".mp4", ".avi", ".mov", ".mkv", ".webm"])
    
    # Find all files with supported extensions
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    
    logger.info(f"Found {len(files)} files to process in {directory}")
    
    # Process each file
    results = {"total": len(files), "success": 0, "failure": 0}
    
    for file_path in files:
        success = process_file(file_path, output_dir, "auto", config)
        if success:
            results["success"] += 1
        else:
            results["failure"] += 1
    
    return results

def main():
    """
    Main entry point for the application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle special commands
    if args.download_models:
        logger.info("Downloading models...")
        ensure_models_downloaded(config.get("models", {}))
        logger.info("Models downloaded successfully.")
        return
    
    if args.list_detectors:
        detectors = get_available_detectors()
        print("Available detectors:")
        for media_type, detector_list in detectors.items():
            print(f"\n{media_type.upper()} detectors:")
            for detector in detector_list:
                print(f"  - {detector}")
        return
    
    # Run in appropriate mode
    if args.mode == "gui":
        logger.info("Starting GUI mode")
        launch_interface(config)
    else:  # CLI mode
        logger.info("Starting CLI mode")
        
        # Ensure required arguments are provided
        if not args.input:
            logger.error("Input file or directory is required for CLI mode")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process file or batch
        if args.batch or os.path.isdir(args.input):
            results = process_batch(args.input, args.output_dir, args.media_type, config)
            logger.info(f"Batch processing completed: {results['success']} succeeded, {results['failure']} failed")
        else:
            success = process_file(args.input, args.output_dir, args.media_type, config)
            logger.info(f"Processing {'succeeded' if success else 'failed'}")

if __name__ == "__main__":
    main()
