#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for the core processing pipeline.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import yaml

from app.core.processor import (BatchProcessor, MediaProcessor, ProcessingStatus,
                               ProcessingTask)
from app.core.queue_manager import ProgressTracker, QueueManager
from app.core.result_handler import ResultHandler
from app.core.workflow import ProcessingWorkflow
from detectors.base_detector import MediaType
from detectors.detection_result import DetectionResult, DetectionStatus


class TestProcessingPipeline(unittest.TestCase):
    """Test cases for the processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.config = {
            "system": {
                "debug_mode": True,
                "log_level": "DEBUG",
                "temp_directory": tempfile.mkdtemp(),
                "max_workers": 2
            },
            "processing": {
                "queue_max_size": 10,
                "timeout_seconds": 5
            },
            "reports": {
                "save_directory": tempfile.mkdtemp(),
                "include_visualization": False
            },
            "models": {
                "image": {
                    "confidence_threshold": 0.7
                },
                "audio": {
                    "confidence_threshold": 0.7
                },
                "video": {
                    "confidence_threshold": 0.7
                }
            }
        }
        
        # Create test media files
        self.test_files = self._create_test_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        for path in self.test_files.values():
            if os.path.exists(path):
                os.remove(path)
        
        # Clean up temp directories
        if os.path.exists(self.config["system"]["temp_directory"]):
            os.rmdir(self.config["system"]["temp_directory"])
        
        if os.path.exists(self.config["reports"]["save_directory"]):
            os.rmdir(self.config["reports"]["save_directory"])
    
    def _create_test_files(self):
        """Create test media files.
        
        Returns:
            Dictionary mapping media types to file paths
        """
        files = {}
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.config["system"]["temp_directory"], exist_ok=True)
        
        # Create dummy image file
        image_path = os.path.join(self.config["system"]["temp_directory"], "test_image.jpg")
        with open(image_path, 'wb') as f:
            f.write(b'dummy image content')
        files["image"] = image_path
        
        # Create dummy audio file
        audio_path = os.path.join(self.config["system"]["temp_directory"], "test_audio.wav")
        with open(audio_path, 'wb') as f:
            f.write(b'dummy audio content')
        files["audio"] = audio_path
        
        # Create dummy video file
        video_path = os.path.join(self.config["system"]["temp_directory"], "test_video.mp4")
        with open(video_path, 'wb') as f:
            f.write(b'dummy video content')
        files["video"] = video_path
        
        return files
    
    @patch("detectors.base_detector.BaseDetector")
    @patch("detectors.detector_factory.DetectorFactory.create_detector")
    def test_media_processor(self, mock_create_detector, mock_base_detector):
        """Test MediaProcessor functionality."""
        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = DetectionResult.create_new(
            MediaType.IMAGE, "test_image.jpg"
        )
        mock_create_detector.return_value = mock_detector
        
        # Create processor
        processor = MediaProcessor(self.config)
        
        # Create task
        task = ProcessingTask(
            task_id="test_task",
            media_path=self.test_files["image"],
            media_type=MediaType.IMAGE,
            config=self.config,
            priority=0,
            callback=None
        )
        
        # Process media
        result = processor.process_media(task)
        
        # Verify result
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.media_type, MediaType.IMAGE)
        self.assertEqual(result.filename, "test_image.jpg")
        
        # Verify task
        self.assertEqual(task.status, ProcessingStatus.COMPLETED)
        self.assertEqual(task.progress, 1.0)
        self.assertIsNotNone(task.start_time)
        self.assertIsNotNone(task.end_time)
    
    @patch("detectors.detector_factory.DetectorFactory.create_detector")
    def test_queue_manager(self, mock_create_detector):
        """Test QueueManager functionality."""
        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = DetectionResult.create_new(
            MediaType.IMAGE, "test_image.jpg"
        )
        mock_create_detector.return_value = mock_detector
        
        # Create processor and queue manager
        processor = MediaProcessor(self.config)
        queue_manager = QueueManager(self.config, processor)
        
        # Start queue manager
        queue_manager.start()
        
        try:
            # Create task
            task = queue_manager.create_task(
                media_path=self.test_files["image"],
                media_type=MediaType.IMAGE
            )
            
            self.assertIsNotNone(task)
            self.assertEqual(task.status, ProcessingStatus.QUEUED)
            
            # Wait for task to complete
            max_wait = 5.0  # seconds
            start_time = time.time()
            
            while task.status != ProcessingStatus.COMPLETED:
                time.sleep(0.1)
                
                # Check for timeout
                if time.time() - start_time > max_wait:
                    self.fail("Task processing timed out")
            
            # Verify task
            self.assertEqual(task.status, ProcessingStatus.COMPLETED)
            self.assertEqual(task.progress, 1.0)
            self.assertIsNotNone(task.result)
            
        finally:
            # Stop queue manager
            queue_manager.stop()
    
    @patch("detectors.detector_factory.DetectorFactory.create_detector")
    def test_result_handler(self, mock_create_detector):
        """Test ResultHandler functionality."""
        # Create result
        result = DetectionResult.create_new(MediaType.IMAGE, "test_image.jpg")
        result.status = DetectionStatus.COMPLETED
        
        # Create result handler
        handler = ResultHandler(self.config)
        
        # Save result
        paths = handler.save_result(result)
        
        # Verify paths
        self.assertIn("json", paths)
        self.assertTrue(os.path.exists(paths["json"]))
        
        # Load result
        loaded_result = handler.load_result(result.id)
        
        # Verify loaded result
        self.assertIsNotNone(loaded_result)
        self.assertEqual(loaded_result.id, result.id)
        self.assertEqual(loaded_result.media_type, result.media_type)
        self.assertEqual(loaded_result.filename, result.filename)
        
        # Delete result
        deleted = handler.delete_result(result.id)
        self.assertTrue(deleted)
        
        # Verify result is deleted
        self.assertIsNone(handler.load_result(result.id))
    
    @patch("detectors.detector_factory.DetectorFactory.create_detector")
    def test_processing_workflow(self, mock_create_detector):
        """Test integrated processing workflow."""
        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = DetectionResult.create_new(
            MediaType.IMAGE, "test_image.jpg"
        )
        mock_create_detector.return_value = mock_detector
        
        # Create workflow
        workflow = ProcessingWorkflow(self.config)
        
        try:
            # Process file synchronously
            result = workflow.process_file(
                file_path=self.test_files["image"],
                media_type=MediaType.IMAGE,
                wait=True
            )
            
            # Verify result
            self.assertIsInstance(result, DetectionResult)
            self.assertEqual(result.media_type, MediaType.IMAGE)
            self.assertEqual(result.filename, "test_image.jpg")
            
            # Process file asynchronously
            task = workflow.process_file(
                file_path=self.test_files["image"],
                media_type=MediaType.IMAGE,
                wait=False
            )
            
            # Verify task
            self.assertIsInstance(task, ProcessingTask)
            
            # Wait for task to complete
            max_wait = 5.0  # seconds
            start_time = time.time()
            
            while task.status != ProcessingStatus.COMPLETED:
                time.sleep(0.1)
                
                # Check for timeout
                if time.time() - start_time > max_wait:
                    self.fail("Task processing timed out")
            
            # Verify task result
            self.assertEqual(task.status, ProcessingStatus.COMPLETED)
            self.assertIsNotNone(task.result)
            
            # Get workflow status
            status = workflow.get_workflow_status()
            
            # Verify status
            self.assertIn("queue", status)
            self.assertIn("results", status)
            self.assertIn("media_types", status)
            
        finally:
            # Shut down workflow
            workflow.shutdown()


if __name__ == '__main__':
    unittest.main()