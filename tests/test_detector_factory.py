#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for the detector factory module.
"""

import unittest
from unittest.mock import MagicMock, patch

from detectors.base_detector import BaseDetector, MediaType
from detectors.detector_factory import (DetectorFactory, DetectorRegistry,
                                      EnsembleDetectorFactory)


# Create mock detector classes for testing
class MockImageDetector(BaseDetector):
    def _initialize(self): pass
    def detect(self, media_path): pass
    def preprocess(self, media_path): pass
    def postprocess(self, model_output): pass
    def get_media_type(self): return MediaType.IMAGE


class MockAudioDetector(BaseDetector):
    def _initialize(self): pass
    def detect(self, media_path): pass
    def preprocess(self, media_path): pass
    def postprocess(self, model_output): pass
    def get_media_type(self): return MediaType.AUDIO


class MockVideoDetector(BaseDetector):
    def _initialize(self): pass
    def detect(self, media_path): pass
    def preprocess(self, media_path): pass
    def postprocess(self, model_output): pass
    def get_media_type(self): return MediaType.VIDEO


class TestDetectorFactory(unittest.TestCase):
    """Test cases for DetectorFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear the registry before each test
        DetectorRegistry._registry = {}
        
        # Register mock detectors
        DetectorRegistry.register(MediaType.IMAGE)(MockImageDetector)
        DetectorRegistry.register(MediaType.AUDIO)(MockAudioDetector)
        DetectorRegistry.register(MediaType.VIDEO)(MockVideoDetector)
        
        self.config = {"model_name": "test_model", "confidence_threshold": 0.7}
    
    def test_detector_registration(self):
        """Test detector registration."""
        self.assertIn(MediaType.IMAGE, DetectorRegistry._registry)
        self.assertIn(MediaType.AUDIO, DetectorRegistry._registry)
        self.assertIn(MediaType.VIDEO, DetectorRegistry._registry)
        
        self.assertEqual(DetectorRegistry._registry[MediaType.IMAGE], MockImageDetector)
        self.assertEqual(DetectorRegistry._registry[MediaType.AUDIO], MockAudioDetector)
        self.assertEqual(DetectorRegistry._registry[MediaType.VIDEO], MockVideoDetector)
    
    def test_get_detector_class(self):
        """Test getting detector class from registry."""
        image_detector_class = DetectorRegistry.get_detector_class(MediaType.IMAGE)
        self.assertEqual(image_detector_class, MockImageDetector)
        
        # Test with unregistered media type
        DetectorRegistry._registry = {}  # Clear registry
        self.assertIsNone(DetectorRegistry.get_detector_class(MediaType.IMAGE))
    
    def test_create_detector(self):
        """Test creating detector instances."""
        # Test creating detectors for each media type
        image_detector = DetectorFactory.create_detector(MediaType.IMAGE, self.config)
        self.assertIsInstance(image_detector, MockImageDetector)
        
        audio_detector = DetectorFactory.create_detector(MediaType.AUDIO, self.config)
        self.assertIsInstance(audio_detector, MockAudioDetector)
        
        video_detector = DetectorFactory.create_detector(MediaType.VIDEO, self.config)
        self.assertIsInstance(video_detector, MockVideoDetector)
    
    def test_create_detector_with_model_path(self):
        """Test creating detector with model path."""
        model_path = "path/to/model"
        detector = DetectorFactory.create_detector(MediaType.IMAGE, self.config, model_path)
        
        self.assertIsInstance(detector, MockImageDetector)
        self.assertEqual(detector.model_path, model_path)
    
    def test_create_detector_with_model_path_from_config(self):
        """Test creating detector with model path from config."""
        config_with_path = self.config.copy()
        config_with_path["model_path"] = "path/from/config"
        
        detector = DetectorFactory.create_detector(MediaType.IMAGE, config_with_path)
        
        self.assertIsInstance(detector, MockImageDetector)
        self.assertEqual(detector.model_path, "path/from/config")
    
    def test_create_detector_unregistered_type(self):
        """Test creating detector for unregistered media type."""
        DetectorRegistry._registry = {}  # Clear registry
        
        with self.assertRaises(ValueError):
            DetectorFactory.create_detector(MediaType.IMAGE, self.config)
    
    @patch('detectors.image_detector.ensemble.ImageEnsembleDetector')
    def test_create_ensemble_detector(self, mock_ensemble_class):
        """Test creating ensemble detector."""
        # Create a mock instance
        mock_ensemble = MagicMock()
        mock_ensemble_class.return_value = mock_ensemble
        
        model_paths = {
            "model1": "path/to/model1",
            "model2": "path/to/model2"
        }
        
        # Create ensemble detector
        detector = EnsembleDetectorFactory.create_ensemble_detector(
            MediaType.IMAGE, self.config, model_paths
        )
        
        # Verify the ensemble class was called with correct parameters
        mock_ensemble_class.assert_called_once_with(self.config, model_paths)
        self.assertEqual(detector, mock_ensemble)
    
    def test_create_ensemble_detector_unsupported_type(self):
        """Test creating ensemble detector for unsupported media type."""
        with self.assertRaises(NotImplementedError):
            EnsembleDetectorFactory.create_ensemble_detector(
                MediaType.AUDIO, self.config
            )
        
        with self.assertRaises(NotImplementedError):
            EnsembleDetectorFactory.create_ensemble_detector(
                MediaType.VIDEO, self.config
            )


if __name__ == '__main__':
    unittest.main()