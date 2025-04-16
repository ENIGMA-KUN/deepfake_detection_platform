#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for the base detector module.
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from detectors.base_detector import (BaseDetector, DetectionResult,
                                    DetectionStatus, MediaType, Region)


class MockDetector(BaseDetector):
    """Mock detector implementation for testing BaseDetector abstract class."""
    
    def _initialize(self):
        self.model = MagicMock()
        
    def detect(self, media_path):
        if not self.validate_media(media_path):
            return DetectionResult(
                id="test_id",
                media_type=MediaType.IMAGE,
                filename=media_path,
                timestamp=datetime.now(),
                is_deepfake=False,
                confidence_score=0.0,
                regions=[],
                metadata={},
                execution_time=0.0,
                model_used="mock_model",
                status=DetectionStatus.FAILED
            )
            
        preprocessed = self.preprocess(media_path)
        model_output = self.model(preprocessed)
        is_deepfake, confidence, regions = self.postprocess(model_output)
        
        return DetectionResult(
            id="test_id",
            media_type=MediaType.IMAGE,
            filename=media_path,
            timestamp=datetime.now(),
            is_deepfake=is_deepfake,
            confidence_score=confidence,
            regions=regions,
            metadata={"test": True},
            execution_time=0.1,
            model_used="mock_model",
            status=DetectionStatus.COMPLETED
        )
    
    def preprocess(self, media_path):
        return np.zeros((224, 224, 3))
    
    def postprocess(self, model_output):
        confidence = 0.75
        is_deepfake = self.is_deepfake(confidence)
        regions = [
            Region(x=0.1, y=0.1, width=0.5, height=0.5, confidence=0.8, label="fake_region")
        ]
        return is_deepfake, confidence, regions
    
    def get_media_type(self):
        return MediaType.IMAGE


class TestBaseDetector(unittest.TestCase):
    """Test cases for BaseDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "confidence_threshold": 0.7,
            "model_name": "test_model"
        }
        self.detector = MockDetector(self.config)
        
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.confidence_threshold, 0.7)
        self.assertIsNotNone(self.detector.model)
        
    @patch('os.path.exists', return_value=True)
    def test_detect(self, mock_exists):
        """Test detection process."""
        result = self.detector.detect("test_image.jpg")
        
        self.assertEqual(result.media_type, MediaType.IMAGE)
        self.assertEqual(result.filename, "test_image.jpg")
        self.assertTrue(result.is_deepfake)
        self.assertEqual(result.confidence_score, 0.75)
        self.assertEqual(len(result.regions), 1)
        self.assertEqual(result.regions[0].label, "fake_region")
        self.assertEqual(result.status, DetectionStatus.COMPLETED)
        
    def test_normalize_confidence(self):
        """Test confidence score normalization."""
        self.assertEqual(self.detector.normalize_confidence(0.5), 0.5)
        self.assertEqual(self.detector.normalize_confidence(1.5), 1.0)
        self.assertEqual(self.detector.normalize_confidence(-0.5), 0.0)
        
    def test_is_deepfake(self):
        """Test deepfake classification."""
        self.assertTrue(self.detector.is_deepfake(0.8))
        self.assertTrue(self.detector.is_deepfake(0.7))  # Threshold
        self.assertFalse(self.detector.is_deepfake(0.6))
        
    @patch('os.path.exists', return_value=False)
    def test_validate_media_nonexistent(self, mock_exists):
        """Test media validation for non-existent file."""
        self.assertFalse(self.detector.validate_media("nonexistent.jpg"))
        
    @patch('os.path.exists', return_value=True)
    def test_validate_media_exists(self, mock_exists):
        """Test media validation for existing file."""
        self.assertTrue(self.detector.validate_media("exists.jpg"))
        
    def test_detection_result_to_dict(self):
        """Test conversion of DetectionResult to dictionary."""
        region = Region(x=0.1, y=0.1, width=0.5, height=0.5, confidence=0.8, label="fake_region")
        result = DetectionResult(
            id="test_id",
            media_type=MediaType.IMAGE,
            filename="test.jpg",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            is_deepfake=True,
            confidence_score=0.8,
            regions=[region],
            metadata={"test": True},
            execution_time=0.1,
            model_used="test_model",
            status=DetectionStatus.COMPLETED
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["id"], "test_id")
        self.assertEqual(result_dict["media_type"], "image")
        self.assertEqual(result_dict["filename"], "test.jpg")
        self.assertEqual(result_dict["is_deepfake"], True)
        self.assertEqual(result_dict["confidence_score"], 0.8)
        self.assertEqual(result_dict["regions"][0]["x"], 0.1)
        self.assertEqual(result_dict["regions"][0]["label"], "fake_region")
        self.assertEqual(result_dict["status"], "completed")


if __name__ == '__main__':
    unittest.main()