#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Processing workflow module for the Deepfake Detection Platform.
Integrates all core components into a cohesive workflow.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.processor import (BatchProcessor, MediaProcessor, ProcessingStatus,
                               ProcessingTask)
from app.core.queue_manager import ProgressTracker, QueueManager
from app.core.result_handler import (ReportGenerator, ResultFormat, ResultHandler)
from detectors.base_detector import MediaType
from detectors.detection_result import DetectionResult

# Configure logger
logger = logging.getLogger(__name__)


class ProcessingWorkflow:
    """Integrated workflow for media processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processing workflow.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.processor = MediaProcessor(config)
        self.queue_manager = QueueManager(config, self.processor)
        self.result_handler = ResultHandler(config)
        self.report_generator = ReportGenerator(config, self.result_handler)
        self.progress_tracker = ProgressTracker(self.queue_manager)
        self.batch_processor = BatchProcessor(self.processor)
        
        # Start queue manager
        self.queue_manager.start()
        
        logger.info("Processing workflow initialized")
    
    def process_file(
        self,
        file_path: str,
        media_type: Optional[MediaType] = None,
        priority: int = 0,
        wait: bool = False,
        progress_callback: Optional[callable] = None,
        completion_callback: Optional[callable] = None,
        generate_report: bool = False
    ) -> Union[ProcessingTask, DetectionResult]:
        """Process a media file.
        
        Args:
            file_path: Path to the media file
            media_type: Type of media (or None to infer)
            priority: Processing priority (higher = higher priority)
            wait: Whether to wait for processing to complete
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback for task completion
            generate_report: Whether to generate a comprehensive report
            
        Returns:
            ProcessingTask if wait=False, DetectionResult if wait=True
            
        Raises:
            ValueError: If media type cannot be determined
            FileNotFoundError: If file not found
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Infer media type if not provided
        if media_type is None:
            media_type = self.processor.infer_media_type(file_path)
            
            if media_type is None:
                raise ValueError(f"Could not determine media type for {file_path}")
        
        # Set metadata
        metadata = {"original_path": file_path}
        
        if wait:
            # Process synchronously
            task = ProcessingTask(
                task_id=str(uuid.uuid4()),
                media_path=file_path,
                media_type=media_type,
                config=self.config,
                priority=priority,
                callback=completion_callback
            )
            
            # Store original path in metadata
            task.config.setdefault("metadata", {}).update(metadata)
            
            # Process media
            result = self.processor.process_media(task)
            
            # Save result
            self.result_handler.save_result(result)
            
            # Generate report if requested
            if generate_report:
                self.report_generator.generate_report(result)
            
            return result
            
        else:
            # Create task config with metadata
            task_config = self.config.copy()
            task_config.setdefault("metadata", {}).update(metadata)
            
            # Process asynchronously
            task = self.queue_manager.create_task(
                media_path=file_path,
                media_type=media_type,
                priority=priority,
                config=task_config,
                callback=self._create_task_callback(completion_callback, generate_report)
            )
            
            if task is None:
                raise RuntimeError(f"Failed to create task for {file_path}")
            
            # Register progress callback if provided
            if progress_callback:
                self.progress_tracker.register_callback(task.task_id, progress_callback)
            
            return task
    
    def process_batch(
        self,
        file_paths: List[str],
        wait: bool = False,
        progress_callback: Optional[callable] = None,
        generate_reports: bool = False
    ) -> Dict[str, Union[ProcessingTask, DetectionResult]]:
        """Process a batch of media files.
        
        Args:
            file_paths: List of paths to media files
            wait: Whether to wait for all processing to complete
            progress_callback: Optional callback for overall progress
            generate_reports: Whether to generate comprehensive reports
            
        Returns:
            Dictionary mapping file paths to tasks or results
        """
        if wait:
            # Process synchronously
            results = {}
            
            for i, file_path in enumerate(file_paths):
                try:
                    # Update progress
                    if progress_callback:
                        progress = i / len(file_paths)
                        progress_callback(progress, f"Processing {i+1}/{len(file_paths)}")
                    
                    # Process file
                    result = self.process_file(
                        file_path=file_path,
                        wait=True,
                        generate_report=generate_reports
                    )
                    
                    results[file_path] = result
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Continue with next file
            
            # Final progress update
            if progress_callback:
                progress_callback(1.0, f"Completed {len(results)}/{len(file_paths)}")
            
            return results
            
        else:
            # Process files with batch processor
            wrapper_callback = None
            if progress_callback:
                wrapper_callback = self._create_batch_callback(len(file_paths), progress_callback)
            
            # Create task config with generate_reports flag
            task_config = self.config.copy()
            task_config["generate_reports"] = generate_reports
            
            # Process batch
            tasks = self.batch_processor.process_batch(
                media_files=file_paths,
                config=task_config,
                callback=wrapper_callback
            )
            
            return tasks
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        return self.queue_manager.get_task(task_id)
    
    def get_result(self, result_id: str) -> Optional[DetectionResult]:
        """Get a result by ID.
        
        Args:
            result_id: Result ID
            
        Returns:
            Result if found, None otherwise
        """
        return self.result_handler.load_result(result_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if canceled, False otherwise
        """
        return self.queue_manager.cancel_task(task_id)
    
    def generate_report(
        self,
        result: Union[DetectionResult, str],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate a comprehensive report for a result.
        
        Args:
            result: Result or result ID
            output_dir: Output directory (or None for default)
            
        Returns:
            Dictionary mapping output types to file paths
        """
        return self.report_generator.generate_report(result, output_dir)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of the workflow.
        
        Returns:
            Dictionary with workflow status
        """
        # Get queue status
        queue_status = self.queue_manager.get_queue_status()
        
        # Count available results
        num_results = len(self.result_handler.list_results())
        
        # Combine status information
        status = {
            "queue": queue_status,
            "results": {
                "count": num_results
            },
            "uptime": time.time(),  # Could track real uptime if needed
            "media_types": [mt.value for mt in self.processor.get_supported_media_types()],
            "formats": {mt.value: fmt for mt, fmt in self.processor.get_supported_formats().items()}
        }
        
        return status
    
    def shutdown(self) -> None:
        """Shut down the workflow and all components."""
        # Stop queue manager
        self.queue_manager.stop()
        
        logger.info("Processing workflow shut down")
    
    def _create_task_callback(
        self,
        user_callback: Optional[callable],
        generate_report: bool
    ) -> callable:
        """Create a callback function for task completion.
        
        Args:
            user_callback: User-provided callback function
            generate_report: Whether to generate a report
            
        Returns:
            Callback function
        """
        def callback(task: ProcessingTask) -> None:
            try:
                # Save result
                if task.result:
                    self.result_handler.save_result(task.result)
                    
                    # Generate report if requested
                    if generate_report and task.status == ProcessingStatus.COMPLETED:
                        self.report_generator.generate_report(task.result)
                
                # Call user callback if provided
                if user_callback:
                    user_callback(task)
                    
            except Exception as e:
                logger.error(f"Error in task callback: {e}")
        
        return callback
    
    def _create_batch_callback(
        self,
        total_files: int,
        progress_callback: callable
    ) -> callable:
        """Create a callback function for batch processing.
        
        Args:
            total_files: Total number of files in the batch
            progress_callback: Function to call with progress updates
            
        Returns:
            Callback function
        """
        completed = [0]  # Use list for mutable closure value
        
        def callback(task: ProcessingTask) -> None:
            try:
                # Increment completed count
                completed[0] += 1
                
                # Calculate progress
                progress = completed[0] / total_files
                
                # Call progress callback
                progress_callback(progress, f"Completed {completed[0]}/{total_files}")
                
            except Exception as e:
                logger.error(f"Error in batch callback: {e}")
        
        return callback