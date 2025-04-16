# tests/test_image_prep.py
import unittest
import os
import tempfile
import numpy as np
from PIL import Image
import torch

from data.preprocessing.image_prep import ImagePreprocessor

class TestImagePreprocessor(unittest.TestCase):
    """Tests for the ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test images
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test image
        self.test_image_path = os.path.join(self.temp_dir.name, "test_image.jpg")
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(self.test_image_path)
        
        # Initialize preprocessor with default settings
        self.preprocessor = ImagePreprocessor()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_preprocess_tensor_output(self):
        """Test preprocessing with tensor output."""
        # Preprocess image with tensor output
        processed_image, meta_info = self.preprocessor.preprocess(self.test_image_path, return_tensor=True)
        
        # Check that output is a tensor
        self.assertIsInstance(processed_image, torch.Tensor)
        
        # Check tensor dimensions
        self.assertEqual(processed_image.shape[0], 3)  # RGB channels
        self.assertEqual(processed_image.shape[1], 224)  # Height
        self.assertEqual(processed_image.shape[2], 224)  # Width
        
        # Check meta info
        self.assertEqual(meta_info['original_size'], (100, 100))
        self.assertIn('load_image', meta_info['preprocessing_steps'])
        self.assertIn('transform_to_tensor', meta_info['preprocessing_steps'])
    
    def test_preprocess_numpy_output(self):
        """Test preprocessing with NumPy output."""
        # Preprocess image with NumPy output
        processed_image, meta_info = self.preprocessor.preprocess(self.test_image_path, return_tensor=False)
        
        # Check that output is a NumPy array
        self.assertIsInstance(processed_image, np.ndarray)
        
        # Check array dimensions
        self.assertEqual(processed_image.shape[0], 224)  # Height
        self.assertEqual(processed_image.shape[1], 224)  # Width
        self.assertEqual(processed_image.shape[2], 3)    # RGB channels
        
        # Check meta info
        self.assertEqual(meta_info['original_size'], (100, 100))
        self.assertIn('load_image', meta_info['preprocessing_steps'])
        self.assertIn('resize', meta_info['preprocessing_steps'])
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        # Create a second test image
        test_image2_path = os.path.join(self.temp_dir.name, "test_image2.jpg")
        test_image2 = Image.new('RGB', (200, 150), color='blue')
        test_image2.save(test_image2_path)
        
        # Batch preprocess
        image_paths = [self.test_image_path, test_image2_path]
        processed_images, meta_infos = self.preprocessor.batch_preprocess(image_paths)
        
        # Check batch tensor
        self.assertIsInstance(processed_images, torch.Tensor)
        self.assertEqual(processed_images.shape[0], 2)  # Batch size
        self.assertEqual(processed_images.shape[1], 3)  # RGB channels
        self.assertEqual(processed_images.shape[2], 224)  # Height
        self.assertEqual(processed_images.shape[3], 224)  # Width
        
        # Check meta info for each image
        self.assertEqual(meta_infos[0]['original_size'], (100, 100))
        self.assertEqual(meta_infos[1]['original_size'], (200, 150))
    
    def test_face_detection(self):
        """Test face detection functionality."""
        # Initialize preprocessor with face detection enabled
        face_preprocessor = ImagePreprocessor({
            'face_detection': True
        })
        
        # Since we don't have a real face in test image, just verify it returns an image
        processed_image, meta_info = face_preprocessor.preprocess(self.test_image_path)
        
        # Check that output is a tensor
        self.assertIsInstance(processed_image, torch.Tensor)
        
        # Check meta info
        self.assertIn('faces_detected', meta_info)
        self.assertEqual(meta_info['faces_detected'], 0)  # No faces in our test image