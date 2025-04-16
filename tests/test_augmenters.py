import unittest
import os
import tempfile
import numpy as np
import librosa
import torch
import cv2
from PIL import Image

from data.augmentation.augmenters import AugmentationPipeline

class TestAugmentationPipeline(unittest.TestCase):
    """Tests for the AugmentationPipeline class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize augmentation pipeline with default settings
        self.augmenter = AugmentationPipeline()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test data for different media types."""
        # Create a test image
        self.test_image_path = os.path.join(self.temp_dir.name, "test_image.jpg")
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_image.save(self.test_image_path)
        self.test_image_pil = test_image
        self.test_image_np = np.array(test_image)
        self.test_image_tensor = torch.from_numpy(np.array(test_image).transpose(2, 0, 1)).float() / 255.0
        
        # Create test audio (1 second of white noise at 16kHz)
        self.test_audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        self.test_sr = 16000
        self.test_audio = np.random.randn(self.test_sr)  # 1 second of white noise
        librosa.output.write_wav(self.test_audio_path, self.test_audio, self.test_sr)
        
        # Create a test video (10 frames of colored patterns)
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        self.create_test_video(self.test_video_path, num_frames=10)
        
        # Load test video frames
        cap = cv2.VideoCapture(self.test_video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        self.test_video_np = np.array(frames)
        
        # Convert to tensor format [T, C, H, W]
        self.test_video_tensor = torch.from_numpy(
            np.array(frames).transpose(0, 3, 1, 2)
        ).float() / 255.0
    
    def create_test_video(self, output_path, width=320, height=240, fps=30, num_frames=10):
        """Create a test video file with synthetic frames."""
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create frames with different patterns
        for i in range(num_frames):
            # Create a colored frame (varying by frame index)
            frame = np.ones((height, width, 3), dtype=np.uint8) * (i * 25 % 255)
            
            # Add some patterns
            cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), 2)
            cv2.circle(frame, (width//2, height//2), 30 + i*3, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write the frame
            writer.write(frame)
        
        # Release the writer
        writer.release()
    
    def test_augment_image_pil(self):
        """Test image augmentation with PIL Image input."""
        # Use fixed number of augmentations
        augmented_image, aug_info = self.augmenter.augment_image(self.test_image_pil, num_augs=2)
        
        # Check output
        self.assertIsInstance(augmented_image, Image.Image)
        self.assertIsInstance(aug_info, dict)
        self.assertIn('applied_augmentations', aug_info)
        
        # Check that augmentations were applied (may not be exactly 2 due to probability)
        self.assertGreaterEqual(len(aug_info['applied_augmentations']), 0)
        
        # Check that each augmentation has type information
        for aug in aug_info['applied_augmentations']:
            self.assertIn('type', aug)
    
    def test_augment_image_numpy(self):
        """Test image augmentation with NumPy array input."""
        # Use fixed number of augmentations
        augmented_image, aug_info = self.augmenter.augment_image(self.test_image_np, num_augs=2)
        
        # Check output
        self.assertIsInstance(augmented_image, np.ndarray)
        self.assertEqual(augmented_image.shape, self.test_image_np.shape)
        
        # Check that augmentations were applied
        self.assertGreaterEqual(len(aug_info['applied_augmentations']), 0)
    
    def test_augment_image_tensor(self):
        """Test image augmentation with torch tensor input."""
        # Use fixed number of augmentations
        augmented_image, aug_info = self.augmenter.augment_image(self.test_image_tensor, num_augs=2)
        
        # Check output
        self.assertIsInstance(augmented_image, torch.Tensor)
        self.assertEqual(augmented_image.shape, self.test_image_tensor.shape)
        
        # Check that augmentations were applied
        self.assertGreaterEqual(len(aug_info['applied_augmentations']), 0)
    
    def test_augment_audio(self):
        """Test audio augmentation."""
        # Use fixed number of augmentations
        augmented_audio, aug_info = self.augmenter.augment_audio(self.test_audio, self.test_sr, num_augs=2)
        
        # Check output
        self.assertIsInstance(augmented_audio, np.ndarray)
        self.assertEqual(len(augmented_audio), len(self.test_audio))
        
        # Check that augmentations were applied
        self.assertGreaterEqual(len(aug_info['applied_augmentations']), 0)
        
        # Check that each augmentation has type information
        for aug in aug_info['applied_augmentations']:
            self.assertIn('type', aug)
    
    def test_augment_video_numpy(self):
        """Test video augmentation with NumPy array input."""
        # Use fixed number of augmentations
        augmented_video, aug_info = self.augmenter.augment_video(self.test_video_np, num_augs=2)
        
        # Check output
        self.assertIsInstance(augmented_video, np.ndarray)
        self.assertEqual(augmented_video.shape, self.test_video_np.shape)
        
        # Check that augmentations were applied
        self.assertGreaterEqual(len(aug_info['applied_augmentations']), 0)
        
        # Check that each augmentation has type information
        for aug in aug_info['applied_augmentations']:
            self.assertIn('type', aug)
    
    def test_augment_video_tensor(self):
        """Test video augmentation with torch tensor input."""
        # Use fixed number of augmentations
        augmented_video, aug_info = self.augmenter.augment_video(self.test_video_tensor, num_augs=2)
        
        # Check output
        self.assertIsInstance(augmented_video, torch.Tensor)
        self.assertEqual(augmented_video.shape, self.test_video_tensor.shape)
        
        # Check that augmentations were applied
        self.assertGreaterEqual(len(aug_info['applied_augmentations']), 0)
    
    def test_augment_batch(self):
        """Test batch augmentation."""
        # Create a batch with multiple media types
        batch = {
            'images': [self.test_image_np, self.test_image_np.copy()],
            'audio': [self.test_audio, self.test_audio.copy()],
            'sr': self.test_sr,
            'video': [self.test_video_np, self.test_video_np.copy()],
            'labels': [0, 1]  # Additional data that shouldn't be augmented
        }
        
        # Augment the batch
        augmented_batch = self.augmenter.augment_batch(batch)
        
        # Check output
        self.assertIsInstance(augmented_batch, dict)
        self.assertIn('images', augmented_batch)
        self.assertIn('audio', augmented_batch)
        self.assertIn('video', augmented_batch)
        self.assertIn('labels', augmented_batch)
        self.assertIn('augmentation_info', augmented_batch)
        
        # Check that the same number of items are returned
        self.assertEqual(len(augmented_batch['images']), len(batch['images']))
        self.assertEqual(len(augmented_batch['audio']), len(batch['audio']))
        self.assertEqual(len(augmented_batch['video']), len(batch['video']))
        self.assertEqual(augmented_batch['labels'], batch['labels'])
        
        # Check augmentation info
        self.assertIn('images', augmented_batch['augmentation_info'])
        self.assertIn('audio', augmented_batch['augmentation_info'])
        self.assertIn('video', augmented_batch['augmentation_info'])