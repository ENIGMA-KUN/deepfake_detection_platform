"""
Unit tests for the Detection Manager.
Tests the integration of detection components with the core processing pipeline.
"""

import os
import unittest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.detection_manager import DetectionManager
from detectors.detection_result import DetectionResult


class TestDetectionManager(unittest.TestCase):
    """Test case for the Detection Manager."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.uploads_dir = os.path.join(self.temp_dir, "uploads")
        self.results_dir = os.path.join(self.temp_dir, "results")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create sample files for testing
        self.image_path = os.path.join(self.uploads_dir, "test_image.jpg")
        self.audio_path = os.path.join(self.uploads_dir, "test_audio.mp3")
        self.video_path = os.path.join(self.uploads_dir, "test_video.mp4")
        
        # Create empty files
        open(self.image_path, "w").close()
        open(self.audio_path, "w").close()
        open(self.video_path, "w").close()
        
        # Configuration for the detection manager
        self.config = {
            "upload_dir": self.uploads_dir,
            "results_dir": self.results_dir,
            "model_cache_dir": self.cache_dir,
            "worker_count": 1,
            "allowed_extensions": {
                "image": [".jpg", ".jpeg", ".png", ".bmp", ".gif"],
                "audio": [".wav", ".mp3", ".flac"],
                "video": [".mp4", ".avi", ".mov"]
            },
            "detectors": {
                "image": {
                    "model_name": "vit-detector",
                    "confidence_threshold": 0.5
                },
                "audio": {
                    "model_name": "wav2vec-detector",
                    "confidence_threshold": 0.5
                },
                "video": {
                    "model_name": "genconvit-detector",
                    "confidence_threshold": 0.5
                }
            },
            "preprocessing": {
                "image": {
                    "target_size": [224, 224],
                    "normalize": True
                },
                "audio": {
                    "sample_rate": 16000,
                    "normalize": True
                },
                "video": {
                    "target_fps": 30,
                    "frame_interval": 10
                }
            }
        }
    
    def tearDown(self):
        """Clean up temporary files after tests."""
        shutil.rmtree(self.temp_dir)
    
    @patch("detectors.detector_factory.DetectorFactory")
    @patch("models.model_loader.ModelLoader")
    @patch("data.preprocessing.preprocessing_pipeline.PreprocessingPipeline")
    @patch("app.core.processor.MediaProcessor")
    @patch("app.core.queue_manager.QueueManager")
    @patch("app.core.result_handler.ResultHandler")
    @patch("app.utils.file_handler.FileHandler")
    def test_initialization(self, mock_file_handler, mock_result_handler, mock_queue_manager,
                          mock_processor, mock_preprocessing, mock_model_loader, mock_detector_factory):
        """Test initialization of the Detection Manager."""
        # Initialize the detection manager
        detection_manager = DetectionManager(self.config)
        
        # Verify that all components were initialized
        mock_model_loader.assert_called_once()
        mock_detector_factory.assert_called_once()
        mock_preprocessing.assert_called_once()
        mock_processor.assert_called_once()
        mock_queue_manager.assert_called_once()
        mock_result_handler.assert_called_once()
        mock_file_handler.assert_called_once()
        
        # Verify that tasks were registered
        queue_manager_instance = mock_queue_manager.return_value
        self.assertEqual(queue_manager_instance.register_task_type.call_count, 4)
    
    @patch("detectors.detector_factory.DetectorFactory")
    @patch("models.model_loader.ModelLoader")
    @patch("data.preprocessing.preprocessing_pipeline.PreprocessingPipeline")
    @patch("app.core.processor.MediaProcessor")
    @patch("app.core.queue_manager.QueueManager")
    @patch("app.core.result_handler.ResultHandler")
    @patch("app.utils.file_handler.FileHandler")
    def test_submit_detection_task(self, mock_file_handler, mock_result_handler, mock_queue_manager,
                                 mock_processor, mock_preprocessing, mock_model_loader, mock_detector_factory):
        """Test submitting a detection task."""
        # Configure mocks
        file_handler_instance = mock_file_handler.return_value
        file_handler_instance.validate_file.return_value = True
        file_handler_instance.get_media_type.return_value = "image"
        
        queue_manager_instance = mock_queue_manager.return_value
        queue_manager_instance.submit_task.return_value = "task-123"
        
        # Initialize the detection manager
        detection_manager = DetectionManager(self.config)
        
        # Submit a detection task
        task_id = detection_manager.submit_detection_task(self.image_path)
        
        # Verify the task was submitted
        self.assertEqual(task_id, "task-123")
        file_handler_instance.validate_file.assert_called_once_with(self.image_path)
        file_handler_instance.get_media_type.assert_called_once_with(self.image_path)
        queue_manager_instance.submit_task.assert_called_once()
    
    @patch("detectors.detector_factory.DetectorFactory")
    @patch("models.model_loader.ModelLoader")
    @patch("data.preprocessing.preprocessing_pipeline.PreprocessingPipeline")
    @patch("app.core.processor.MediaProcessor")
    @patch("app.core.queue_manager.QueueManager")
    @patch("app.core.result_handler.ResultHandler")
    @patch("app.utils.file_handler.FileHandler")
    def test_batch_detection_task(self, mock_file_handler, mock_result_handler, mock_queue_manager,
                               mock_processor, mock_preprocessing, mock_model_loader, mock_detector_factory):
        """Test submitting a batch detection task."""
        # Configure mocks
        file_handler_instance = mock_file_handler.return_value
        file_handler_instance.validate_file.return_value = True
        file_handler_instance.get_media_type.side_effect = ["image", "audio", "video"]
        
        queue_manager_instance = mock_queue_manager.return_value
        queue_manager_instance.submit_task.return_value = "batch-123"
        
        # Initialize the detection manager
        detection_manager = DetectionManager(self.config)
        
        # Submit a batch detection task
        task_id = detection_manager.submit_batch_detection_task([
            self.image_path, self.audio_path, self.video_path
        ])
        
        # Verify the batch task was submitted
        self.assertEqual(task_id, "batch-123")
        self.assertEqual(file_handler_instance.validate_file.call_count, 3)
        self.assertEqual(file_handler_instance.get_media_type.call_count, 3)
        queue_manager_instance.submit_task.assert_called_once()
    
    @patch("detectors.detector_factory.DetectorFactory")
    @patch("models.model_loader.ModelLoader")
    @patch("data.preprocessing.preprocessing_pipeline.PreprocessingPipeline")
    @patch("app.core.processor.MediaProcessor")
    @patch("app.core.queue_manager.QueueManager")
    @patch("app.core.result_handler.ResultHandler")
    @patch("app.utils.file_handler.FileHandler")
    def test_process_image_detection(self, mock_file_handler, mock_result_handler, mock_queue_manager,
                                  mock_processor, mock_preprocessing, mock_model_loader, mock_detector_factory):
        """Test processing an image detection task."""
        # Configure mocks
        preprocessor_instance = mock_preprocessing.return_value
        preprocessor_instance.preprocess_image.return_value = {"processed_data": "test"}
        
        detector_factory_instance = mock_detector_factory.return_value
        mock_detector = MagicMock()
        mock_detector.detect.return_value = DetectionResult(
            file_path=self.image_path,
            media_type="image",
            detector_name="ViT Detector",
            is_deepfake=True,
            confidence_score=0.85
        )
        detector_factory_instance.create_detector.return_value = mock_detector
        
        result_handler_instance = mock_result_handler.return_value
        result_handler_instance.save_result.return_value = "result-123"
        
        # Initialize the detection manager
        detection_manager = DetectionManager(self.config)
        
        # Process an image detection task
        task_data = {"file_path": self.image_path}
        result = detection_manager._process_image_detection(task_data)
        
        # Verify the task was processed correctly
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_id"], "result-123")
        self.assertTrue(result["is_deepfake"])
        self.assertEqual(result["confidence_score"], 0.85)
        
        preprocessor_instance.preprocess_image.assert_called_once_with(self.image_path)
        detector_factory_instance.create_detector.assert_called_once_with("image")
        mock_detector.detect.assert_called_once()
        result_handler_instance.save_result.assert_called_once()
    
    @patch("detectors.detector_factory.DetectorFactory")
    @patch("models.model_loader.ModelLoader")
    @patch("data.preprocessing.preprocessing_pipeline.PreprocessingPipeline")
    @patch("app.core.processor.MediaProcessor")
    @patch("app.core.queue_manager.QueueManager")
    @patch("app.core.result_handler.ResultHandler")
    @patch("app.utils.file_handler.FileHandler")
    def test_get_task_status(self, mock_file_handler, mock_result_handler, mock_queue_manager,
                          mock_processor, mock_preprocessing, mock_model_loader, mock_detector_factory):
        """Test getting the status of a task."""
        # Configure mocks
        queue_manager_instance = mock_queue_manager.return_value
        queue_manager_instance.get_task_status.return_value = {
            "status": "completed",
            "progress": 100,
            "result_id": "result-123"
        }
        
        # Initialize the detection manager
        detection_manager = DetectionManager(self.config)
        
        # Get task status
        status = detection_manager.get_task_status("task-123")
        
        # Verify the status was retrieved
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["progress"], 100)
        self.assertEqual(status["result_id"], "result-123")
        queue_manager_instance.get_task_status.assert_called_once_with("task-123")
    
    @patch("detectors.detector_factory.DetectorFactory")
    @patch("models.model_loader.ModelLoader")
    @patch("data.preprocessing.preprocessing_pipeline.PreprocessingPipeline")
    @patch("app.core.processor.MediaProcessor")
    @patch("app.core.queue_manager.QueueManager")
    @patch("app.core.result_handler.ResultHandler")
    @patch("app.utils.file_handler.FileHandler")
    def test_start_stop(self, mock_file_handler, mock_result_handler, mock_queue_manager,
                      mock_processor, mock_preprocessing, mock_model_loader, mock_detector_factory):
        """Test starting and stopping the detection manager."""
        # Configure mocks
        queue_manager_instance = mock_queue_manager.return_value
        
        # Initialize the detection manager
        detection_manager = DetectionManager(self.config)
        
        # Start and stop the detection manager
        detection_manager.start()
        detection_manager.stop()
        
        # Verify the queue manager was started and stopped
        queue_manager_instance.start.assert_called_once()
        queue_manager_instance.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()