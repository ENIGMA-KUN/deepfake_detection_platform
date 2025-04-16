import unittest
import os
import tempfile
import numpy as np
import librosa
import torch
import cv2
from PIL import Image

from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline

class TestPreprocessingPipeline(unittest.TestCase):
    """Tests for the PreprocessingPipeline class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize preprocessing pipeline with default settings
        self.pipeline = PreprocessingPipeline()
        
        # Create test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def create_test_files(self):
        """Create test files for different media types."""
        # Create a test image
        self.test_image_path = os.path.join(self.temp_dir.name, "test_image.jpg")
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_image.save(self.test_image_path)
        
        # Create test audio (1 second of white noise at 16kHz)
        self.test_audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        self.test_sr = 16000
        self.test_audio = np.random.randn(self.test_sr)  # 1 second of white noise
        librosa.output.write_wav(self.test_audio_path, self.test_audio, self.test_sr)
        
        # Create a test video (10 frames of colored patterns)
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        self.create_test_video(self.test_video_path, num_frames=10)
    
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
    
    def test_determine_media_type(self):
        """Test media type determination."""
        # Test image file
        self.assertEqual(self.pipeline.determine_media_type(self.test_image_path), 'image')
        
        # Test audio file
        self.assertEqual(self.pipeline.determine_media_type(self.test_audio_path), 'audio')
        
        # Test video file
        self.assertEqual(self.pipeline.determine_media_type(self.test_video_path), 'video')
        
        # Test unknown file
        unknown_file_path = os.path.join(self.temp_dir.name, "unknown.xyz")
        with open(unknown_file_path, 'w') as f:
            f.write("test")
        self.assertEqual(self.pipeline.determine_media_type(unknown_file_path), 'unknown')
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Preprocess image
        preprocessed_image, meta_info = self.pipeline.preprocess(self.test_image_path)
        
        # Check output
        self.assertIsInstance(preprocessed_image, torch.Tensor)
        self.assertEqual(preprocessed_image.shape[0], 3)  # RGB channels
        self.assertEqual(preprocessed_image.shape[1], 224)  # Height
        self.assertEqual(preprocessed_image.shape[2], 224)  # Width
        
        # Check meta info
        self.assertEqual(meta_info['media_type'], 'image')
        self.assertEqual(meta_info['file_path'], self.test_image_path)
    
    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        # Preprocess audio
        preprocessed_audio, meta_info = self.pipeline.preprocess(self.test_audio_path)
        
        # Check output
        self.assertIsInstance(preprocessed_audio, torch.Tensor)
        
        # Check meta info
        self.assertEqual(meta_info['media_type'], 'audio')
        self.assertEqual(meta_info['file_path'], self.test_audio_path)
        self.assertEqual(meta_info['original_sr'], 16000)
    
    def test_preprocess_video(self):
        """Test video preprocessing."""
        # Preprocess video
        preprocessed_video, meta_info = self.pipeline.preprocess(self.test_video_path)
        
        # Check output
        self.assertIsInstance(preprocessed_video, torch.Tensor)
        self.assertEqual(preprocessed_video.shape[0], 10)  # 10 frames
        self.assertEqual(preprocessed_video.shape[1], 3)   # RGB channels
        
        # Check meta info
        self.assertEqual(meta_info['media_type'], 'video')
        self.assertEqual(meta_info['file_path'], self.test_video_path)
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        # Create batch of files
        file_paths = [self.test_image_path, self.test_audio_path, self.test_video_path]
        
        # Batch preprocess
        preprocessed_data, meta_info = self.pipeline.batch_preprocess(file_paths)
        
        # Check output structure
        self.assertIn('images', preprocessed_data)
        self.assertIn('audio', preprocessed_data)
        self.assertIn('video', preprocessed_data)
        
        self.assertIn('images', meta_info)
        self.assertIn('audio', meta_info)
        self.assertIn('video', meta_info)
        
        # Check that each media type has correct data
        self.assertEqual(len(preprocessed_data['images']), 1)
        self.assertEqual(len(preprocessed_data['audio']), 1)
        self.assertEqual(len(preprocessed_data['video']), 1)
    
    def test_extract_features_audio(self):
        """Test feature extraction for audio."""
        # Preprocess audio
        preprocessed_audio, meta_info = self.pipeline.preprocess(self.test_audio_path)
        
        # Extract features
        features = self.pipeline.extract_features(preprocessed_audio, 'audio', meta_info)
        
        # Check output
        self.assertIsInstance(features, dict)
        self.assertIn('mel_spectrogram', features)
        self.assertIn('mfcc', features)
    
    def test_extract_features_video(self):
        """Test feature extraction for video."""
        # Preprocess video
        preprocessed_video, meta_info = self.pipeline.preprocess(self.test_video_path)
        
        # Extract features
        features = self.pipeline.extract_features(preprocessed_video, 'video', meta_info)
        
        # Check output
        self.assertIsInstance(features, dict)
        self.assertIn('mean_motion', features)
        self.assertIn('std_motion', features)
    
    def test_cache(self):
        """Test preprocessing cache."""
        # Enable cache
        self.pipeline.cache_enabled = True
        
        # Preprocess the same file twice
        preprocessed_image1, _ = self.pipeline.preprocess(self.test_image_path)
        
        # Check cache size
        self.assertEqual(self.pipeline.get_cache_size(), 1)
        
        # Preprocess again (should use cache)
        preprocessed_image2, _ = self.pipeline.preprocess(self.test_image_path)
        
        # Check that both outputs are the same object (from cache)
        self.assertIs(preprocessed_image1, preprocessed_image2)
        
        # Clear cache
        self.pipeline.clear_cache()
        
        # Check cache size
        self.assertEqual(self.pipeline.get_cache_size(), 0)
    
    def test_augmentation(self):
        """Test preprocessing with augmentation."""
        # Enable augmentation
        pipeline_with_aug = PreprocessingPipeline({
            'apply_augmentation': True,
            'augmentation_config': {
                'image_aug_prob': 1.0,  # Always apply augmentation
                'num_image_augs': 1     # Apply 1 augmentation
            }
        })
        
        # Preprocess with augmentation
        preprocessed_image, meta_info = pipeline_with_aug.preprocess(self.test_image_path)
        
        # Check that augmentation was applied
        self.assertIn('augmentation', meta_info)
        self.assertIn('applied_augmentations', meta_info['augmentation'])
    
    def test_create_preprocessing_config(self):
        """Test creating preprocessing configuration."""
        # Create config
        config = self.pipeline.create_preprocessing_config()
        
        # Check config structure
        self.assertIn('image_config', config)
        self.assertIn('audio_config', config)
        self.assertIn('video_config', config)
        self.assertIn('apply_augmentation', config)
        self.assertIn('enable_cache', config)
        
        # Enable augmentation and check config again
        self.pipeline.apply_augmentation = True
        self.pipeline.augmenter = pipeline = PreprocessingPipeline({
            'apply_augmentation': True
        }).augmenter
        
        config = self.pipeline.create_preprocessing_config()
        self.assertIn('augmentation_config', config)