"""
Detection Manager for the Deepfake Detection Platform.
Coordinates the integration of detection components with the core processing pipeline.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time

from detectors.base_detector import MediaType, DetectionStatus
from detectors.detector_factory import DetectorFactory
from detectors.detection_result import DetectionResult
from models.model_loader import ModelLoader
from models.model_config import ModelConfiguration
from app.core.queue_manager import QueueManager
from app.core.result_handler import ResultHandler
from app.utils.file_handler import FileHandler
from app.utils.logging_utils import setup_logger
from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# Set up logger
logger = setup_logger(__name__)

class DetectionManager:
    """
    Manages the integration between the core processing pipeline and the detector modules.
    Acts as the central coordinator for the detection workflow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detection manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logger
        
        # Initialize components
        model_config = ModelConfiguration(config.get("models", {}))
        self.model_loader = ModelLoader(
            cache_dir=config.get("model_cache_dir", "models/cache"),
            config=model_config
        )
        
        self.detector_factory = DetectorFactory(
            model_loader=self.model_loader,
            config=config.get("detectors", {})
        )
        
        self.preprocessing_pipeline = PreprocessingPipeline(
            config=config.get("preprocessing", {})
        )
        
        self.queue_manager = QueueManager(
            config=config.get("queue", {}),
            worker_count=config.get("worker_count", 2)
        )
        
        self.result_handler = ResultHandler(
            output_dir=config.get("results_dir", "reports/output"),
            config=config.get("results", {})
        )
        
        self.file_handler = FileHandler(
            upload_dir=config.get("upload_dir", "data/uploads"),
            allowed_extensions=config.get("allowed_extensions", {})
        )
        
        # Register detection tasks with the queue manager
        self._register_detection_tasks()
        
        self.logger.info("Detection Manager initialized successfully.")
    
    def _register_detection_tasks(self) -> None:
        """Register detection tasks with the queue manager."""
        # Register image detection task
        self.queue_manager.register_task_type(
            "image_detection",
            lambda task_data: self._process_image_detection(task_data)
        )
        
        # Register audio detection task
        self.queue_manager.register_task_type(
            "audio_detection",
            lambda task_data: self._process_audio_detection(task_data)
        )
        
        # Register video detection task
        self.queue_manager.register_task_type(
            "video_detection",
            lambda task_data: self._process_video_detection(task_data)
        )
        
        # Register batch detection task
        self.queue_manager.register_task_type(
            "batch_detection",
            lambda task_data: self._process_batch_detection(task_data)
        )
    
    def submit_detection_task(self, file_path: str, media_type: Optional[MediaType] = None) -> str:
        """
        Submit a detection task to the queue.
        
        Args:
            file_path: Path to the media file
            media_type: Type of media (MediaType enum) or None for auto-detection
            
        Returns:
            Task ID for tracking the task
        """
        # Validate the file
        if not self.file_handler.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
        
        # Auto-detect media type if not provided
        if media_type is None:
            media_type = self.file_handler.detect_media_type(file_path)
        
        # Determine task type based on media type
        task_type = f"{media_type.value}_detection"
        
        # Create task data
        task_data = {
            "file_path": file_path,
            "media_type": media_type.value,
            "status": DetectionStatus.PENDING.value,
            "timestamp": time.time()
        }
        
        # Submit task to queue
        task_id = self.queue_manager.submit_task(task_type, task_data)
        
        self.logger.info(f"Submitted {media_type.value} detection task for {file_path}. Task ID: {task_id}")
        
        return task_id
    
    def submit_batch_detection_task(self, file_paths: List[str]) -> str:
        """
        Submit a batch detection task to the queue.
        
        Args:
            file_paths: List of paths to the media files
            
        Returns:
            Task ID for tracking the batch task
        """
        # Validate all files and get their media types
        batch_items = []
        for file_path in file_paths:
            if not self.file_handler.validate_file(file_path):
                self.logger.warning(f"Skipping invalid file: {file_path}")
                continue
            
            media_type = self.file_handler.detect_media_type(file_path)
            batch_items.append({
                "file_path": file_path,
                "media_type": media_type.value
            })
        
        # Create batch task data
        task_data = {
            "batch_items": batch_items,
            "timestamp": time.time()
        }
        
        # Submit batch task to queue
        task_id = self.queue_manager.submit_task("batch_detection", task_data)
        
        self.logger.info(f"Submitted batch detection task with {len(batch_items)} files. Task ID: {task_id}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a detection task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        return self.queue_manager.get_task_status(task_id)
    
    def get_result(self, result_id: str) -> DetectionResult:
        """
        Get a detection result by ID.
        
        Args:
            result_id: Result ID
            
        Returns:
            Detection result
        """
        return self.result_handler.get_result(result_id)
    
    def _process_media_detection(self, task_data: Dict[str, Any], media_type: MediaType) -> Dict[str, Any]:
        """
        Process a media detection task.
        
        Args:
            task_data: Task data including file path
            media_type: Type of media to process
            
        Returns:
            Processing result
        """
        file_path = task_data["file_path"]
        self.logger.info(f"Processing {media_type.value} detection for {file_path}")
        
        try:
            # Update task status
            task_data["status"] = DetectionStatus.PROCESSING.value
            
            # Preprocess the media
            if media_type == MediaType.IMAGE:
                preprocessed_data = self.preprocessing_pipeline.preprocess_image(file_path)
            elif media_type == MediaType.AUDIO:
                preprocessed_data = self.preprocessing_pipeline.preprocess_audio(file_path)
            elif media_type == MediaType.VIDEO:
                preprocessed_data = self.preprocessing_pipeline.preprocess_video(file_path)
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
            
            # Get the appropriate detector
            detector = self.detector_factory.create_detector(media_type.value)
            
            # Run detection
            result = detector.detect(preprocessed_data)
            
            # Store the result
            result_id = self.result_handler.save_result(result)
            
            return {
                "status": DetectionStatus.COMPLETED.value,
                "result_id": result_id,
                "is_deepfake": result.is_deepfake,
                "confidence_score": result.confidence_score
            }
        
        except Exception as e:
            self.logger.error(f"Error processing {media_type.value} detection for {file_path}: {str(e)}")
            return {
                "status": DetectionStatus.FAILED.value,
                "error": str(e)
            }
    
    def _process_image_detection(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image detection task."""
        return self._process_media_detection(task_data, MediaType.IMAGE)
    
    def _process_audio_detection(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an audio detection task."""
        return self._process_media_detection(task_data, MediaType.AUDIO)
    
    def _process_video_detection(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video detection task."""
        return self._process_media_detection(task_data, MediaType.VIDEO)
    
    def _process_batch_detection(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch detection task.
        
        Args:
            task_data: Task data including batch items
            
        Returns:
            Processing result
        """
        batch_items = task_data["batch_items"]
        self.logger.info(f"Processing batch detection with {len(batch_items)} files")
        
        results = []
        failed_items = []
        
        # Process each item in the batch
        for item in batch_items:
            file_path = item["file_path"]
            media_type = MediaType(item["media_type"])
            
            try:
                # Create a subtask for processing
                subtask_data = {
                    "file_path": file_path,
                    "media_type": media_type.value,
                    "status": DetectionStatus.PENDING.value
                }
                
                result = self._process_media_detection(subtask_data, media_type)
                
                if result["status"] == DetectionStatus.COMPLETED.value:
                    results.append(result)
                else:
                    failed_items.append({
                        "file_path": file_path,
                        "error": result.get("error", "Unknown error")
                    })
            
            except Exception as e:
                self.logger.error(f"Error processing batch item {file_path}: {str(e)}")
                failed_items.append({
                    "file_path": file_path,
                    "error": str(e)
                })
        
        return {
            "status": DetectionStatus.COMPLETED.value,
            "total_items": len(batch_items),
            "successful_items": len(results),
            "failed_items": len(failed_items),
            "results": results,
            "failures": failed_items
        }
    
    def start(self) -> None:
        """Start the detection manager and its components."""
        self.logger.info("Starting Detection Manager...")
        self.queue_manager.start()
        self.logger.info("Detection Manager started.")
    
    def stop(self) -> None:
        """Stop the detection manager and its components."""
        self.logger.info("Stopping Detection Manager...")
        self.queue_manager.stop()
        self.logger.info("Detection Manager stopped.")