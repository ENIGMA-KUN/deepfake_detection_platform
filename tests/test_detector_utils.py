#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for the detector utilities module.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                      DetectionStatus, MediaType, Region,
                                      TimeSegment)
from detectors.detector_utils import (calculate_ensemble_confidence,
                                     identify_deepfake_category,
                                     measure_execution_time,
                                     normalize_confidence_score)


class TestDetectorUtils(unittest.TestCase):
    """Test cases for detector utilities."""
    
    def test_normalize_confidence_score(self):
        """Test normalization of confidence scores."""
        # Test within range
        self.assertEqual(normalize_confidence_score(0.5), 0.5)
        
        # Test above range
        self.assertEqual(normalize_confidence_score(1.5), 1.0)
        
        # Test below range
        self.assertEqual(normalize_confidence_score(-0.5), 0.0)
        
        # Test edge cases
        self.assertEqual(normalize_confidence_score(0.0), 0.0)
        self.assertEqual(normalize_confidence_score(1.0), 1.0)
    
    def test_calculate_ensemble_confidence_equal_weights(self):
        """Test ensemble confidence calculation with equal weights."""
        scores = [0.7, 0.8, 0.9]
        confidence = calculate_ensemble_confidence(scores)
        
        # Expected: average of scores
        expected = (0.7 + 0.8 + 0.9) / 3
        self.assertAlmostEqual(confidence, expected, places=6)
    
    def test_calculate_ensemble_confidence_custom_weights(self):
        """Test ensemble confidence calculation with custom weights."""
        scores = [0.7, 0.8, 0.9]
        weights = [0.5, 0.3, 0.2]
        confidence = calculate_ensemble_confidence(scores, weights)
        
        # Expected: weighted average of scores
        expected = 0.7 * 0.5 + 0.8 * 0.3 + 0.9 * 0.2
        self.assertAlmostEqual(confidence, expected, places=6)
    
    def test_calculate_ensemble_confidence_empty(self):
        """Test ensemble confidence calculation with empty scores."""
        self.assertEqual(calculate_ensemble_confidence([]), 0.0)
    
    def test_calculate_ensemble_confidence_weight_normalization(self):
        """Test normalization of weights in ensemble confidence calculation."""
        scores = [0.7, 0.8, 0.9]
        weights = [5, 3, 2]  # Sum = 10, should be normalized to [0.5, 0.3, 0.2]
        confidence = calculate_ensemble_confidence(scores, weights)
        
        # Expected: weighted average with normalized weights
        expected = 0.7 * 0.5 + 0.8 * 0.3 + 0.9 * 0.2
        self.assertAlmostEqual(confidence, expected, places=6)
    
    def test_calculate_ensemble_confidence_unequal_lengths(self):
        """Test ensemble confidence calculation with unequal scores and weights."""
        scores = [0.7, 0.8, 0.9]
        weights = [0.5, 0.5]  # Length mismatch
        
        with self.assertRaises(ValueError):
            calculate_ensemble_confidence(scores, weights)
    
    def test_identify_deepfake_category_below_threshold(self):
        """Test deepfake category identification below confidence threshold."""
        categories = identify_deepfake_category(MediaType.IMAGE, 0.4)
        self.assertEqual(categories, [])  # Below threshold, no categories
    
    def test_identify_deepfake_category_image_default(self):
        """Test default deepfake category for images."""
        categories = identify_deepfake_category(MediaType.IMAGE, 0.7)
        self.assertEqual(categories, [DeepfakeCategory.GAN_GENERATED])
    
    def test_identify_deepfake_category_audio_default(self):
        """Test default deepfake category for audio."""
        categories = identify_deepfake_category(MediaType.AUDIO, 0.7)
        self.assertEqual(categories, [DeepfakeCategory.AUDIO_SYNTHESIS])
    
    def test_identify_deepfake_category_video_default(self):
        """Test default deepfake category for video."""
        categories = identify_deepfake_category(MediaType.VIDEO, 0.7)
        self.assertEqual(categories, [DeepfakeCategory.VIDEO_MANIPULATION])
    
    def test_identify_deepfake_category_from_regions(self):
        """Test deepfake category identification from regions."""
        regions = [
            Region(x=0.1, y=0.1, width=0.1, height=0.1, confidence=0.8, 
                  label="fake_face", category=DeepfakeCategory.FACE_SWAP),
            Region(x=0.2, y=0.2, width=0.1, height=0.1, confidence=0.9, 
                  label="fake_eyes", category=DeepfakeCategory.FACE_MANIPULATION)
        ]
        
        categories = identify_deepfake_category(MediaType.IMAGE, 0.7, regions=regions)
        
        # Both categories from regions should be identified
        self.assertEqual(set(categories), {DeepfakeCategory.FACE_SWAP, 
                                          DeepfakeCategory.FACE_MANIPULATION})
    
    def test_identify_deepfake_category_from_time_segments(self):
        """Test deepfake category identification from time segments."""
        segments = [
            TimeSegment(start_time=1.0, end_time=2.0, confidence=0.8, 
                      label="voice_clone", category=DeepfakeCategory.VOICE_CLONING),
            TimeSegment(start_time=3.0, end_time=4.0, confidence=0.9, 
                      label="audio_synth", category=DeepfakeCategory.AUDIO_SYNTHESIS)
        ]
        
        categories = identify_deepfake_category(MediaType.AUDIO, 0.7, 
                                              time_segments=segments)
        
        # Both categories from time segments should be identified
        self.assertEqual(set(categories), {DeepfakeCategory.VOICE_CLONING, 
                                          DeepfakeCategory.AUDIO_SYNTHESIS})
    
    def test_measure_execution_time_decorator(self):
        """Test execution time measurement decorator."""
        # Create a test function
        @measure_execution_time
        def test_function():
            return "test_result"
        
        # Execute the function and check result
        result = test_function()
        self.assertEqual(result, "test_result")
    
    def test_measure_execution_time_with_detection_result(self):
        """Test execution time measurement with DetectionResult."""
        # Create a test function that returns a DetectionResult
        @measure_execution_time
        def test_detection():
            return DetectionResult.create_new(MediaType.IMAGE, "test.jpg")
        
        # Execute the function
        result = test_detection()
        
        # Check that execution_time was set
        self.assertGreater(result.execution_time, 0.0)
        self.assertIsInstance(result, DetectionResult)


if __name__ == '__main__':
    unittest.main()