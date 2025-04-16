#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Media processor module for the Deepfake Detection Platform.
Handles routing media to appropriate detectors and managing the detection workflow.
"""

import logging
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from detectors.base_detector import MediaType
from detectors.detection_result import DetectionResult, DetectionStatus
from detectors.detector_factory import DetectorFactory

# Configure logger
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enum for processing status."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ProcessingTask:
    """Class representing a media processing task."""
    
    def __init__(
        self,
        task_id: str,
        media_path: str,
        media_type: MediaType,
        config: Dict[str, Any],
        priority: int = 0,
        callback: Optional[callable] = None
    ):
        """Initialize a processing task.
        
        Args:
            task_id: Unique identifier for the task
            media_path: Path to the media file
            media_type: Type of media to process
            config: Configuration for the processing
            priority: Task priority (higher value = higher priority)
            callback: Optional callback function to call when processing completes
        """
        self.task_id = task_id
        self.media_path = media_path
        self.media_type = media_type
        self.config = config
        self.priority = priority
        self.callback = callback
        
        self.status = ProcessingStatus.PENDING
        self.progress = 0.0
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "media_path": self.media_path,
            "media_type": self.media_type.value,
            "priority": self.priority,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class MediaProcessor:
    """Processor for handling media detection tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the media processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detectors = {}  # Cache for detector instances
    
    def process_media(self, task: ProcessingTask) -> DetectionResult:
        """Process a media file for deepfake detection.
        
        Args:
            task: Processing task to execute
            
        Returns:
            DetectionResult with the detection results
            
        Raises:
            ValueError: If media type is not supported
            FileNotFoundError: If media file does not exist
        """
        # Validate media file
        if not os.path.exists(task.media_path):
            raise FileNotFoundError(f"Media file not found: {task.media_path}")
        
        # Set task status
        task.status = ProcessingStatus.PROCESSING
        task.start_time = time.time()
        task.progress = 0.0
        
        try:
            logger.info(f"Processing {task.media_type.value} file: {task.media_path}")
            
            # Get appropriate detector
            detector = self._get_detector(task.media_type)
            
            # Update progress
            task.progress = 0.2
            
            # Perform detection
            result = detector.detect(task.media_path)
            
            # Update progress
            task.progress = 0.8
            
            # Validate result
            if not isinstance(result, DetectionResult):
                raise ValueError(f"Invalid result type: {type(result)}")
            
            # Post-process result if needed
            result = self._post_process_result(result)
            
            # Update task
            task.result = result
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            
            logger.info(f"Processing completed for task {task.task_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            
            # Update task
            task.status = ProcessingStatus.FAILED
            task.error = str(e)
            
            # Create failed result if needed
            if not task.result:
                task.result = self._create_failed_result(task, str(e))
            
            return task.result
            
        finally:
            # Update end time
            task.end_time = time.time()
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Error in callback for task {task.task_id}: {e}")
    
    def _get_detector(self, media_type: MediaType) -> Any:
        """Get the appropriate detector for the media type.
        
        Args:
            media_type: Type of media to detect
            
        Returns:
            Detector instance
            
        Raises:
            ValueError: If media type is not supported
        """
        # Check if detector is cached
        if media_type in self.detectors:
            return self.detectors[media_type]
        
        # Get detector config
        detector_config = self.config.get("models", {}).get(media_type.value, {})
        
        # Create detector
        detector = DetectorFactory.create_detector(media_type, detector_config)
        
        # Cache detector
        self.detectors[media_type] = detector
        
        return detector
    
    def _post_process_result(self, result: DetectionResult) -> DetectionResult:
        """Perform any additional processing on the result.
        
        Args:
            result: Detection result to process
            
        Returns:
            Processed detection result
        """
        # Add any additional processing logic here
        # For example, saving results to a database, generating reports, etc.
        
        return result
    
    def _create_failed_result(
        self,
        task: ProcessingTask,
        error_message: str
    ) -> DetectionResult:
        """Create a failed detection result.
        
        Args:
            task: Processing task that failed
            error_message: Error message
            
        Returns:
            Failed detection result
        """
        # Create a new result with failed status
        filename = os.path.basename(task.media_path)
        result = DetectionResult.create_new(task.media_type, filename)
        result.status = DetectionStatus.FAILED
        result.metadata["error"] = error_message
        
        return result
    
    def get_supported_media_types(self) -> List[MediaType]:
        """Get a list of supported media types.
        
        Returns:
            List of supported MediaType values
        """
        return [MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO]
    
    def get_supported_formats(self) -> Dict[MediaType, List[str]]:
        """Get a list of supported file formats for each media type.
        
        Returns:
            Dictionary mapping MediaType to list of supported extensions
        """
        # Get formats from config or use defaults
        formats = self.config.get("ui", {}).get("supported_formats", {})
        
        # Convert to MediaType keys
        result = {}
        
        for media_type in MediaType:
            key = media_type.value
            if key in formats:
                result[media_type] = formats[key]
            else:
                # Use defaults if not specified
                if media_type == MediaType.IMAGE:
                    result[media_type] = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
                elif media_type == MediaType.AUDIO:
                    result[media_type] = [".wav", ".mp3", ".flac", ".ogg"]
                elif media_type == MediaType.VIDEO:
                    result[media_type] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        
        return result
    
    def infer_media_type(self, file_path: str) -> Optional[MediaType]:
        """Infer media type from file extension.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            MediaType or None if not recognized
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Get supported formats
        formats = self.get_supported_formats()
        
        # Check each media type
        for media_type, extensions in formats.items():
            if ext in extensions:
                return media_type
        
        return None


class BatchProcessor:
    """Processor for handling batch processing of multiple media files."""
    
    def __init__(self, processor: MediaProcessor):
        """Initialize the batch processor.
        
        Args:
            processor: MediaProcessor instance to use for processing
        """
        self.processor = processor
    
    def process_batch(
        self,
        media_files: List[str],
        config: Dict[str, Any],
        callback: Optional[callable] = None,
    ) -> Dict[str, ProcessingTask]:
        """Process a batch of media files.
        
        Args:
            media_files: List of paths to media files
            config: Configuration for processing
            callback: Optional callback function for each completed task
            
        Returns:
            Dictionary mapping file paths to processing tasks
        """
        tasks = {}
        
        for i, media_path in enumerate(media_files):
            # Create task ID
            task_id = f"batch_{i}_{os.path.basename(media_path)}"
            
            # Infer media type
            media_type = self.processor.infer_media_type(media_path)
            
            if media_type is None:
                logger.warning(f"Could not determine media type for {media_path}")
                continue
            
            # Create task
            task = ProcessingTask(
                task_id=task_id,
                media_path=media_path,
                media_type=media_type,
                config=config,
                priority=0,
                callback=callback
            )
            
            # Process task
            try:
                self.processor.process_media(task)
            except Exception as e:
                logger.error(f"Error processing {media_path}: {e}")
                task.status = ProcessingStatus.FAILED
                task.error = str(e)
            
            # Store task
            tasks[media_path] = task
        
        return tasks