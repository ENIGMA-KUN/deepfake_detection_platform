#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Queue management module for the Deepfake Detection Platform.
Handles task queuing, prioritization, and execution.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from app.core.processor import MediaProcessor, ProcessingStatus, ProcessingTask

# Configure logger
logger = logging.getLogger(__name__)


class QueueManager:
    """Manager for processing queue and execution."""
    
    def __init__(self, config: Dict[str, Any], processor: MediaProcessor):
        """Initialize the queue manager.
        
        Args:
            config: Configuration dictionary
            processor: MediaProcessor instance to use for processing
        """
        self.config = config
        self.processor = processor
        
        # Queue settings
        self.max_queue_size = config.get("processing", {}).get("queue_max_size", 100)
        self.max_workers = config.get("system", {}).get("max_workers", 4)
        
        # Task collections
        self.tasks = {}  # Map of task_id to ProcessingTask
        self.task_queue = PriorityQueue(maxsize=self.max_queue_size)
        
        # Thread pool for task execution
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Control flags
        self.running = False
        self.worker_thread = None
        
        # Stats
        self.processed_count = 0
        self.failed_count = 0
    
    def start(self) -> None:
        """Start the queue manager."""
        if self.running:
            logger.warning("Queue manager is already running")
            return
        
        logger.info("Starting queue manager")
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop(self) -> None:
        """Stop the queue manager."""
        if not self.running:
            logger.warning("Queue manager is not running")
            return
        
        logger.info("Stopping queue manager")
        self.running = False
        
        # Wait for worker thread to terminate
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def add_task(self, task: ProcessingTask) -> bool:
        """Add a task to the queue.
        
        Args:
            task: Task to add
            
        Returns:
            True if task was added, False otherwise
        """
        # Check if queue is full
        if self.task_queue.full():
            logger.error("Task queue is full, cannot add task")
            return False
        
        # Check if task already exists
        if task.task_id in self.tasks:
            logger.warning(f"Task {task.task_id} already exists")
            return False
        
        # Add task to collections
        self.tasks[task.task_id] = task
        
        # Update task status
        task.status = ProcessingStatus.QUEUED
        
        # Add to priority queue
        # Use negative priority so that higher values have higher priority
        self.task_queue.put((-task.priority, task.task_id))
        
        logger.info(f"Added task {task.task_id} to queue with priority {task.priority}")
        return True
    
    def create_task(
        self,
        media_path: str,
        media_type: Optional[Any] = None,
        priority: int = 0,
        config: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
    ) -> Optional[ProcessingTask]:
        """Create and add a new task.
        
        Args:
            media_path: Path to the media file
            media_type: Type of media (if None, will be inferred)
            priority: Task priority (higher values = higher priority)
            config: Configuration overrides for this task (or None to use defaults)
            callback: Optional callback function to call when processing completes
            
        Returns:
            Created task if successful, None otherwise
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Infer media type if not provided
        if media_type is None:
            media_type = self.processor.infer_media_type(media_path)
            
            if media_type is None:
                logger.error(f"Could not determine media type for {media_path}")
                return None
        
        # Use default config if not provided
        if config is None:
            config = self.config
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            media_path=media_path,
            media_type=media_type,
            config=config,
            priority=priority,
            callback=callback
        )
        
        # Add to queue
        if self.add_task(task):
            return task
        else:
            return None
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was canceled, False otherwise
        """
        task = self.tasks.get(task_id)
        
        if task is None:
            logger.warning(f"Task {task_id} not found")
            return False
        
        # Can only cancel pending or queued tasks
        if task.status not in [ProcessingStatus.PENDING, ProcessingStatus.QUEUED]:
            logger.warning(f"Cannot cancel task {task_id} with status {task.status}")
            return False
        
        # Update status
        task.status = ProcessingStatus.CANCELED
        
        logger.info(f"Canceled task {task_id}")
        return True
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the queue.
        
        Returns:
            Dictionary with queue status
        """
        # Count tasks by status
        status_counts = {}
        for status in ProcessingStatus:
            status_counts[status.value] = 0
        
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
        
        return {
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "workers": self.max_workers,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "status_counts": status_counts,
            "running": self.running
        }
    
    def get_all_tasks(self) -> Dict[str, ProcessingTask]:
        """Get all tasks.
        
        Returns:
            Dictionary mapping task IDs to tasks
        """
        return self.tasks.copy()
    
    def clear_completed_tasks(self, age_seconds: Optional[float] = None) -> int:
        """Clear completed or failed tasks from the task collection.
        
        Args:
            age_seconds: Only clear tasks older than this many seconds (or None for all)
            
        Returns:
            Number of tasks cleared
        """
        to_remove = []
        current_time = time.time()
        
        for task_id, task in self.tasks.items():
            if task.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELED]:
                # Check age if specified
                if age_seconds is not None and task.end_time is not None:
                    age = current_time - task.end_time
                    if age < age_seconds:
                        continue
                
                to_remove.append(task_id)
        
        # Remove tasks
        for task_id in to_remove:
            del self.tasks[task_id]
        
        logger.info(f"Cleared {len(to_remove)} completed tasks")
        return len(to_remove)
    
    def _process_queue(self) -> None:
        """Process tasks from the queue (worker thread function)."""
        logger.info("Queue processing thread started")
        
        while self.running:
            try:
                # Get next task from queue
                if self.task_queue.empty():
                    # Sleep briefly if queue is empty
                    time.sleep(0.1)
                    continue
                
                # Get task from queue
                _, task_id = self.task_queue.get(block=False)
                
                # Check if task exists and is still valid
                task = self.tasks.get(task_id)
                if task is None or task.status != ProcessingStatus.QUEUED:
                    logger.warning(f"Task {task_id} not found or not in QUEUED status")
                    self.task_queue.task_done()
                    continue
                
                # Submit task to thread pool
                self.executor.submit(self._execute_task, task)
                
                # Mark queue item as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                time.sleep(1.0)  # Sleep briefly on error
        
        logger.info("Queue processing thread stopped")
    
    def _execute_task(self, task: ProcessingTask) -> None:
        """Execute a task in a worker thread.
        
        Args:
            task: Task to execute
        """
        logger.info(f"Executing task {task.task_id}")
        
        try:
            # Process the task
            self.processor.process_media(task)
            
            # Update stats
            self.processed_count += 1
            if task.status == ProcessingStatus.FAILED:
                self.failed_count += 1
                
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            
            # Update task
            task.status = ProcessingStatus.FAILED
            task.error = str(e)
            
            # Update stats
            self.processed_count += 1
            self.failed_count += 1


class ProgressTracker:
    """Tracker for processing progress."""
    
    def __init__(self, queue_manager: QueueManager):
        """Initialize the progress tracker.
        
        Args:
            queue_manager: QueueManager instance to track
        """
        self.queue_manager = queue_manager
        self.callbacks = {}  # Map of task_id to progress callback functions
    
    def register_callback(
        self,
        task_id: str,
        callback: Callable[[str, float, Optional[str]], None]
    ) -> bool:
        """Register a progress callback for a task.
        
        Args:
            task_id: Task ID
            callback: Callback function (task_id, progress, message)
            
        Returns:
            True if callback was registered, False otherwise
        """
        # Check if task exists
        task = self.queue_manager.get_task(task_id)
        if task is None:
            logger.warning(f"Cannot register callback for unknown task {task_id}")
            return False
        
        # Register callback
        self.callbacks[task_id] = callback
        
        # Call immediately with current progress
        callback(task_id, task.progress, None)
        
        return True
    
    def unregister_callback(self, task_id: str) -> bool:
        """Unregister a progress callback.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if callback was unregistered, False otherwise
        """
        if task_id in self.callbacks:
            del self.callbacks[task_id]
            return True
        else:
            return False
    
    def update_progress(self, task_id: str, progress: float, message: Optional[str] = None) -> None:
        """Update progress for a task and notify callback.
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
        """
        # Check if task exists
        task = self.queue_manager.get_task(task_id)
        if task is None:
            logger.warning(f"Cannot update progress for unknown task {task_id}")
            return
        
        # Update task progress
        task.progress = progress
        
        # Notify callback if registered
        callback = self.callbacks.get(task_id)
        if callback:
            try:
                callback(task_id, progress, message)
            except Exception as e:
                logger.error(f"Error in progress callback for task {task_id}: {e}")
    
    def get_progress(self, task_id: str) -> Tuple[float, ProcessingStatus]:
        """Get progress for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Tuple of (progress, status)
            
        Raises:
            ValueError: If task not found
        """
        # Check if task exists
        task = self.queue_manager.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        
        return task.progress, task.status