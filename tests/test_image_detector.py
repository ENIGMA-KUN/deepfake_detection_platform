"""
Unit tests for the image detector module.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from PIL import Image

from detectors.image_detector.vit_detector import ViTImageDetector
from detectors.base_detector import BaseDetector
from models.model_loader import ModelLoader
from data.preprocessing.image_prep import ImagePreprocessor


class TestViTImageDetector(unittest.TestCase):
    """Test cases for the ViT-based image detector."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the model loader to avoid actual model loading
        self.model_loader_patcher = patch('models.model_loader.ModelLoader')
        self.mock_model_loader = self.model_loader_patcher.start()

        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.return_value = torch.tensor([0.3])  # Simulate a model prediction
        self.mock_model_loader.return_value.load_model.return_value = self.mock_model

        # Create a mock image processor
        self.processor_patcher = patch('data.preprocessing.image_prep.ImagePreprocessor')
        self.mock_processor = self.processor_patcher.start()
        
        # Setup mock for face detection
        self.mock_processor.return_value.detect_faces.return_value = [
            {
                'box': [10, 10, 100, 100],
                'confidence': 0.99
            }
        ]
        
        # Setup mock for preprocessing
        self.mock_processor.return_value.preprocess_image.return_value = torch.rand(1, 3, 224, 224)
        
        # Create detector with mocked dependencies
        self.detector = ViTImageDetector(
            model_name="google/vit-base-patch16-224",
            confidence_threshold=0.5
        )
        
        # Create a dummy test image
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_image_path = os.path.join(self.test_dir, 'test_image.jpg')
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(self.test_image_path):
            img = Image.new('RGB', (224, 224), color='white')
            img.save(self.test_image_path)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.model_loader_patcher.stop()
        self.processor_patcher.stop()

    def test_detector_initialization(self):
        """Test that the detector initializes correctly."""
        self.assertIsInstance(self.detector, BaseDetector)
        self.assertEqual(self.detector.model_name, "google/vit-base-patch16-224")
        self.assertEqual(self.detector.confidence_threshold, 0.5)

    def test_detect_method_returns_valid_result(self):
        """Test that the detect method returns a valid detection result."""
        # Patch the extract_attention_map method to avoid actual computation
        with patch.object(ViTImageDetector, 'extract_attention_map', return_value=np.zeros((224, 224))):
            result = self.detector.detect(self.test_image_path)
            
            self.assertIsInstance(result, dict)
            self.assertIn('is_deepfake', result)
            self.assertIn('confidence', result)
            self.assertIn('metadata', result)
            
            # Check metadata contents
            metadata = result['metadata']
            self.assertIn('faces_detected', metadata)
            self.assertIn('face_results', metadata)
            self.assertIn('processing_time', metadata)

    def test_face_detection_integration(self):
        """Test face detection integration within the detector."""
        # Setup mock face detection to return multiple faces
        mock_faces = [
            {'box': [10, 10, 100, 100], 'confidence': 0.99},
            {'box': [150, 150, 100, 100], 'confidence': 0.95}
        ]
        self.mock_processor.return_value.detect_faces.return_value = mock_faces
        
        # Patch the extract_attention_map method
        with patch.object(ViTImageDetector, 'extract_attention_map', return_value=np.zeros((224, 224))):
            result = self.detector.detect(self.test_image_path)
            
            # Verify correct number of faces detected
            self.assertEqual(result['metadata']['faces_detected'], 2)
            self.assertEqual(len(result['metadata']['face_results']), 2)

    def test_no_faces_detected_fallback(self):
        """Test fallback to whole image analysis when no faces are detected."""
        # Setup mock to return no faces
        self.mock_processor.return_value.detect_faces.return_value = []
        
        # Patch the extract_attention_map method
        with patch.object(ViTImageDetector, 'extract_attention_map', return_value=np.zeros((224, 224))):
            result = self.detector.detect(self.test_image_path)
            
            # Verify whole image analysis was performed
            self.assertEqual(result['metadata']['faces_detected'], 0)
            self.assertIn('whole_image_score', result['metadata'])

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
            'faces_detected': 1,
            'face_results': [{'confidence': 0.7, 'box': [10, 10, 100, 100]}],
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
