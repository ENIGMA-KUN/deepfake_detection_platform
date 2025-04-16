"""
Unit tests for the video detector module.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import cv2

from detectors.video_detector.genconvit import GenConViTVideoDetector
from detectors.base_detector import BaseDetector
from models.model_loader import ModelLoader
from data.preprocessing.video_prep import VideoPreprocessor


class TestGenConViTVideoDetector(unittest.TestCase):
    """Test cases for the GenConViT-based video detector."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the model loader to avoid actual model loading
        self.model_loader_patcher = patch('models.model_loader.ModelLoader')
        self.mock_model_loader = self.model_loader_patcher.start()

        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.return_value = torch.tensor([0.3])  # Simulate a model prediction
        self.mock_model_loader.return_value.load_model.return_value = self.mock_model

        # Create a mock video processor
        self.processor_patcher = patch('data.preprocessing.video_prep.VideoPreprocessor')
        self.mock_processor = self.processor_patcher.start()
        
        # Setup mock for video preprocessing
        frames = [np.random.rand(224, 224, 3).astype(np.uint8) for _ in range(10)]
        audio = np.random.rand(16000)
        
        self.mock_processor.return_value.extract_frames.return_value = frames
        self.mock_processor.return_value.extract_audio.return_value = (audio, 16000)
        self.mock_processor.return_value.preprocess_frame.return_value = torch.rand(1, 3, 224, 224)
        self.mock_processor.return_value.get_video_info.return_value = {
            'width': 1280,
            'height': 720,
            'fps': 30.0,
            'frame_count': 300,
            'duration': 10.0
        }
        
        # Create detector with mocked dependencies
        self.detector = GenConViTVideoDetector(
            model_name="custom/genconvit-model",
            confidence_threshold=0.5
        )
        
        # Create a dummy test video file
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_video_path = os.path.join(self.test_dir, 'test_video.mp4')
        
        # Create a simple test video file if it doesn't exist
        if not os.path.exists(self.test_video_path):
            # Create a blank video with 30 frames
            height, width = 720, 1280
            fps = 30
            seconds = 1
            
            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.test_video_path, fourcc, fps, (width, height))
            
            # Generate frames with simple patterns
            for i in range(fps * seconds):
                # Create a blank frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add some text and a moving circle
                cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cx = int(width/2 + (width/4) * np.sin(i * 2 * np.pi / (fps * seconds)))
                cy = int(height/2)
                cv2.circle(frame, (cx, cy), 50, (0, 0, 255), -1)
                
                # Write the frame
                out.write(frame)
            
            # Release the writer
            out.release()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.model_loader_patcher.stop()
        self.processor_patcher.stop()

    def test_detector_initialization(self):
        """Test that the detector initializes correctly."""
        self.assertIsInstance(self.detector, BaseDetector)
        self.assertEqual(self.detector.model_name, "custom/genconvit-model")
        self.assertEqual(self.detector.confidence_threshold, 0.5)

    def test_detect_method_returns_valid_result(self):
        """Test that the detect method returns a valid detection result."""
        # Mock the frame_level_analysis method to return fake analysis
        with patch.object(GenConViTVideoDetector, 'frame_level_analysis', return_value={
            'frame_scores': [0.2, 0.3, 0.6, 0.7, 0.3],
            'detection_frames': [False, False, True, True, False]
        }):
            # Mock the temporal_analysis method to return fake analysis
            with patch.object(GenConViTVideoDetector, 'temporal_analysis', return_value=0.4):
                # Mock the analyze_av_sync method
                with patch.object(GenConViTVideoDetector, 'analyze_av_sync', return_value=0.25):
                    result = self.detector.detect(self.test_video_path)
                    
                    self.assertIsInstance(result, dict)
                    self.assertIn('is_deepfake', result)
                    self.assertIn('confidence', result)
                    self.assertIn('metadata', result)
                    
                    # Check metadata contents
                    metadata = result['metadata']
                    self.assertIn('video_info', metadata)
                    self.assertIn('frames_analyzed', metadata)
                    self.assertIn('frame_scores', metadata)
                    self.assertIn('temporal_inconsistency', metadata)
                    self.assertIn('av_sync_score', metadata)
                    self.assertIn('processing_time', metadata)

    def test_frame_level_analysis(self):
        """Test the frame level analysis functionality."""
        # Create mock frames
        frames = [np.random.rand(224, 224, 3).astype(np.uint8) for _ in range(5)]
        
        # Create mock model outputs
        mock_outputs = [torch.tensor([0.2]), torch.tensor([0.3]), 
                       torch.tensor([0.6]), torch.tensor([0.7]), 
                       torch.tensor([0.3])]
        
        with patch.object(self.detector.model, '__call__', side_effect=mock_outputs):
            # Mock the preprocessing step
            processed_frames = [torch.rand(1, 3, 224, 224) for _ in range(5)]
            self.mock_processor.return_value.preprocess_frame.side_effect = processed_frames
            
            # Run frame-level analysis
            analysis = self.detector.frame_level_analysis(frames)
            
            # Verify analysis results
            self.assertIn('frame_scores', analysis)
            self.assertIn('detection_frames', analysis)
            
            # Check computed values
            self.assertEqual(len(analysis['frame_scores']), 5)
            self.assertEqual(len(analysis['detection_frames']), 5)
            
            # Test detection logic based on threshold
            self.assertEqual(analysis['detection_frames'], [False, False, True, True, False])

    def test_temporal_analysis(self):
        """Test the temporal analysis functionality."""
        # Test with various score patterns
        
        # Case 1: Consistent scores (low inconsistency)
        consistent_scores = [0.3, 0.32, 0.29, 0.31, 0.28]
        inconsistency1 = self.detector.temporal_analysis(consistent_scores)
        
        # Case 2: Variable scores (high inconsistency)
        variable_scores = [0.1, 0.9, 0.2, 0.8, 0.3]
        inconsistency2 = self.detector.temporal_analysis(variable_scores)
        
        # Verify that inconsistency is higher when scores are more variable
        self.assertLess(inconsistency1, inconsistency2)
        
        # Verify bounds
        self.assertGreaterEqual(inconsistency1, 0.0)
        self.assertLessEqual(inconsistency1, 1.0)
        self.assertGreaterEqual(inconsistency2, 0.0)
        self.assertLessEqual(inconsistency2, 1.0)

    def test_analyze_av_sync(self):
        """Test the audio-video synchronization analysis."""
        # Create mock frames and audio
        frames = [np.random.rand(224, 224, 3).astype(np.uint8) for _ in range(10)]
        audio = np.random.rand(16000)
        
        # Mock the feature extraction
        with patch.object(self.detector, '_extract_audio_features', return_value=torch.rand(128)):
            with patch.object(self.detector, '_extract_visual_features', return_value=torch.rand(128)):
                # Run A/V sync analysis
                sync_score = self.detector.analyze_av_sync(frames, audio, 16000)
                
                # Verify score is within bounds
                self.assertGreaterEqual(sync_score, 0.0)
                self.assertLessEqual(sync_score, 1.0)

    def test_normalize_confidence(self):
        """Test the confidence normalization function."""
        # Test various raw scores
        self.assertAlmostEqual(self.detector.normalize_confidence(0.0), 0.0)
        self.assertAlmostEqual(self.detector.normalize_confidence(0.5), 0.5)
        self.assertAlmostEqual(self.detector.normalize_confidence(1.0), 1.0)
        
        # Test clamping of values outside [0,1]
        self.assertAlmostEqual(self.detector.normalize_confidence(-0.5), 0.0)
        self.assertAlmostEqual(self.detector.normalize_confidence(1.5), 1.0)

    def test_format_result(self):
        """Test the result formatting function."""
        metadata = {
            'video_info': {
                'width': 1280,
                'height': 720,
                'fps': 30.0,
                'frame_count': 300,
                'duration': 10.0
            },
            'frames_analyzed': 10,
            'frame_scores': [0.2, 0.3, 0.6, 0.7, 0.3, 0.2, 0.4, 0.5, 0.6, 0.3],
            'temporal_inconsistency': 0.4,
            'av_sync_score': 0.25,
            'processing_time': 1.5
        }
        
        # Test deepfake result
        result = self.detector.format_result(True, 0.7, metadata)
        self.assertTrue(result['is_deepfake'])
        self.assertEqual(result['confidence'], 0.7)
        self.assertEqual(result['metadata'], metadata)
        
        # Test authentic result
        result = self.detector.format_result(False, 0.3, metadata)
        self.assertFalse(result['is_deepfake'])
        self.assertEqual(result['confidence'], 0.3)


if __name__ == '__main__':
    unittest.main()
