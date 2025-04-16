"""
Unit tests for the audio detector module.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import librosa

from detectors.audio_detector.wav2vec_detector import Wav2VecAudioDetector
from detectors.base_detector import BaseDetector
from models.model_loader import ModelLoader
from data.preprocessing.audio_prep import AudioPreprocessor


class TestWav2VecAudioDetector(unittest.TestCase):
    """Test cases for the Wav2Vec2-based audio detector."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the model loader to avoid actual model loading
        self.model_loader_patcher = patch('models.model_loader.ModelLoader')
        self.mock_model_loader = self.model_loader_patcher.start()

        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.return_value = torch.tensor([0.3])  # Simulate a model prediction
        self.mock_model_loader.return_value.load_model.return_value = self.mock_model

        # Create a mock audio processor
        self.processor_patcher = patch('data.preprocessing.audio_prep.AudioPreprocessor')
        self.mock_processor = self.processor_patcher.start()
        
        # Setup mock for audio preprocessing
        self.mock_processor.return_value.load_audio.return_value = (np.random.rand(16000), 16000)
        self.mock_processor.return_value.preprocess_audio.return_value = torch.rand(1, 16000)
        self.mock_processor.return_value.segment_audio.return_value = [torch.rand(1, 16000) for _ in range(3)]
        self.mock_processor.return_value.generate_spectrogram.return_value = np.random.rand(128, 128)
        
        # Create detector with mocked dependencies
        self.detector = Wav2VecAudioDetector(
            model_name="facebook/wav2vec2-large-960h",
            confidence_threshold=0.5
        )
        
        # Create a dummy test audio file
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_audio_path = os.path.join(self.test_dir, 'test_audio.wav')
        
        # Create a simple test audio file if it doesn't exist
        if not os.path.exists(self.test_audio_path):
            # Generate a simple sine wave
            sample_rate = 16000
            duration = 2.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            # Save as WAV file
            try:
                librosa.output.write_wav(self.test_audio_path, audio_data, sample_rate)
            except AttributeError:
                # For newer versions of librosa
                import soundfile as sf
                sf.write(self.test_audio_path, audio_data, sample_rate)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.model_loader_patcher.stop()
        self.processor_patcher.stop()

    def test_detector_initialization(self):
        """Test that the detector initializes correctly."""
        self.assertIsInstance(self.detector, BaseDetector)
        self.assertEqual(self.detector.model_name, "facebook/wav2vec2-large-960h")
        self.assertEqual(self.detector.confidence_threshold, 0.5)

    def test_detect_method_returns_valid_result(self):
        """Test that the detect method returns a valid detection result."""
        # Mock the temporal_analysis method to avoid actual computation
        with patch.object(Wav2VecAudioDetector, 'temporal_analysis', return_value={
            'inconsistency_index': 0.3,
            'segment_scores': [0.2, 0.3, 0.4],
            'segment_times': [0.0, 1.0, 2.0]
        }):
            result = self.detector.detect(self.test_audio_path)
            
            self.assertIsInstance(result, dict)
            self.assertIn('is_deepfake', result)
            self.assertIn('confidence', result)
            self.assertIn('metadata', result)
            
            # Check metadata contents
            metadata = result['metadata']
            self.assertIn('duration', metadata)
            self.assertIn('original_sample_rate', metadata)
            self.assertIn('temporal_analysis', metadata)
            self.assertIn('processing_time', metadata)

    def test_temporal_analysis(self):
        """Test the temporal analysis functionality."""
        # Mock segment scores for the temporal analysis
        segment_scores = torch.tensor([0.2, 0.6, 0.8])
        
        # Create a mock model output that returns the segment scores
        mock_outputs = [score.unsqueeze(0) for score in segment_scores]
        
        with patch.object(self.detector.model, '__call__', side_effect=mock_outputs):
            # Mock the preprocessing step
            audio_segments = [torch.rand(1, 16000) for _ in range(3)]
            
            # Run temporal analysis
            analysis = self.detector.temporal_analysis(audio_segments, [0.0, 1.0, 2.0])
            
            # Verify analysis results
            self.assertIn('inconsistency_index', analysis)
            self.assertIn('segment_scores', analysis)
            self.assertIn('segment_times', analysis)
            
            # Check computed values
            self.assertEqual(len(analysis['segment_scores']), 3)
            self.assertEqual(len(analysis['segment_times']), 3)
            
            # Inconsistency index should increase with variance in scores
            self.assertGreater(analysis['inconsistency_index'], 0)

    def test_process_wav2vec_output(self):
        """Test the processing of Wav2Vec2 model output."""
        # Create mock model output
        mock_output = torch.tensor([[0.3, 0.7]])  # Higher score for the second class (deepfake)
        
        with patch.object(self.detector, 'normalize_confidence', return_value=0.7):
            # Process the output
            is_deepfake, confidence = self.detector._process_wav2vec_output(mock_output)
            
            # Verify results
            self.assertTrue(is_deepfake)
            self.assertEqual(confidence, 0.7)
            
            # Test with opposite scores
            mock_output = torch.tensor([[0.8, 0.2]])  # Higher score for the first class (authentic)
            with patch.object(self.detector, 'normalize_confidence', return_value=0.2):
                is_deepfake, confidence = self.detector._process_wav2vec_output(mock_output)
                self.assertFalse(is_deepfake)
                self.assertEqual(confidence, 0.2)

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
            'duration': 2.0,
            'original_sample_rate': 16000,
            'temporal_analysis': {
                'inconsistency_index': 0.3,
                'segment_scores': [0.2, 0.3, 0.4],
                'segment_times': [0.0, 1.0, 2.0]
            },
            'processing_time': 0.5
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
