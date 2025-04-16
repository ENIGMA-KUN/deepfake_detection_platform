import unittest
import os
import tempfile
import numpy as np
import librosa
import torch

from data.preprocessing.audio_prep import AudioPreprocessor

class TestAudioPreprocessor(unittest.TestCase):
    """Tests for the AudioPreprocessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test audio file (1 second of white noise at 16kHz)
        self.test_audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        sr = 16000
        y = np.random.randn(sr)  # 1 second of white noise
        librosa.output.write_wav(self.test_audio_path, y, sr)
        
        # Initialize preprocessor with default settings
        self.preprocessor = AudioPreprocessor()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_preprocess_tensor_output(self):
        """Test preprocessing with tensor output."""
        # Preprocess audio with tensor output
        processed_audio, meta_info = self.preprocessor.preprocess(self.test_audio_path, return_tensor=True)
        
        # Check that output is a tensor
        self.assertIsInstance(processed_audio, torch.Tensor)
        
        # Check tensor dimensions (should be target_duration * target_sr = 5.0 * 16000 = 80000 samples)
        self.assertEqual(processed_audio.shape[0], int(self.preprocessor.target_duration * self.preprocessor.target_sr))
        
        # Check meta info
        self.assertEqual(meta_info['original_sr'], 16000)
        self.assertIn('load_audio', meta_info['preprocessing_steps'])
        self.assertIn('pad', meta_info['preprocessing_steps'])  # Should pad 1s to 5s
        self.assertIn('normalize', meta_info['preprocessing_steps'])
        self.assertIn('to_tensor', meta_info['preprocessing_steps'])
    
    def test_preprocess_numpy_output(self):
        """Test preprocessing with NumPy output."""
        # Preprocess audio with NumPy output
        processed_audio, meta_info = self.preprocessor.preprocess(self.test_audio_path, return_tensor=False)
        
        # Check that output is a NumPy array
        self.assertIsInstance(processed_audio, np.ndarray)
        
        # Check array dimensions
        self.assertEqual(processed_audio.shape[0], int(self.preprocessor.target_duration * self.preprocessor.target_sr))
        
        # Check meta info
        self.assertEqual(meta_info['original_sr'], 16000)
        self.assertIn('load_audio', meta_info['preprocessing_steps'])
        self.assertIn('pad', meta_info['preprocessing_steps'])
        self.assertIn('normalize', meta_info['preprocessing_steps'])
        self.assertNotIn('to_tensor', meta_info['preprocessing_steps'])
    
    def test_extract_features(self):
        """Test feature extraction."""
        # Load the test audio file
        y, sr = librosa.load(self.test_audio_path, sr=None)
        
        # Extract features
        features = self.preprocessor.extract_features(y, sr)
        
        # Check features
        self.assertIn('mel_spectrogram', features)
        self.assertIn('mfcc', features)
        self.assertIn('chroma', features)
        self.assertIn('contrast', features)
        self.assertIn('tempogram', features)
        self.assertIn('zcr', features)
        
        # Check feature dimensions
        self.assertEqual(features['mfcc'].shape[0], 20)  # 20 MFCC coefficients
        self.assertEqual(features['mel_spectrogram'].shape[0], self.preprocessor.n_mels)
    
    def test_compute_silence_stats(self):
        """Test silence statistics computation."""
        # Create a signal with known silence
        sr = 16000
        y = np.random.randn(sr)  # 1 second of white noise
        y[sr//4:sr//2] = 0  # 0.25 seconds of silence (25%)
        
        # Compute silence statistics
        silence_stats = self.preprocessor.compute_silence_stats(y)
        
        # Check statistics
        self.assertIn('silence_percentage', silence_stats)
        self.assertIn('max_silence_length', silence_stats)
        self.assertIn('mean_silence_length', silence_stats)
        self.assertIn('num_silence_segments', silence_stats)
        
        # Test accuracy (with some tolerance)
        self.assertGreater(silence_stats['silence_percentage'], 20)
        self.assertLess(silence_stats['silence_percentage'], 30)
        self.assertAlmostEqual(silence_stats['max_silence_length'], 0.25, delta=0.05)
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        # Create a second test audio file
        test_audio2_path = os.path.join(self.temp_dir.name, "test_audio2.wav")
        sr = 22050  # Different sample rate
        y = np.random.randn(sr * 2)  # 2 seconds of white noise
        librosa.output.write_wav(test_audio2_path, y, sr)
        
        # Batch preprocess
        audio_paths = [self.test_audio_path, test_audio2_path]
        processed_audios, meta_infos = self.preprocessor.batch_preprocess(audio_paths)
        
        # Check batch results
        self.assertEqual(len(processed_audios), 2)
        self.assertEqual(len(meta_infos), 2)
        
        # Check sample rates were handled correctly
        self.assertEqual(meta_infos[0]['original_sr'], 16000)
        self.assertEqual(meta_infos[1]['original_sr'], 22050)
        self.assertIn('resample', meta_infos[1]['preprocessing_steps'])
        
        # Check all are tensors
        for audio in processed_audios:
            self.assertIsInstance(audio, torch.Tensor)
    
    def test_augmentation(self):
        """Test augmentation."""
        # Initialize preprocessor with augmentation enabled
        aug_preprocessor = AudioPreprocessor({
            'apply_augmentation': True
        })
        
        # Preprocess with augmentation
        processed_audio, meta_info = aug_preprocessor.preprocess(self.test_audio_path)
        
        # Check meta info
        self.assertIn('augmentation', meta_info['preprocessing_steps'])
        
        # Preprocess again to test a different augmentation (due to randomness)
        processed_audio2, meta_info2 = aug_preprocessor.preprocess(self.test_audio_path)
        
        # The two augmented versions should be different (very unlikely to be the same)
        self.assertFalse(torch.allclose(processed_audio, processed_audio2))
    
    def test_create_spectrogram(self):
        """Test spectrogram creation."""
        # Create a temporary file for the spectrogram
        spectrogram_path = os.path.join(self.temp_dir.name, "spectrogram.png")
        
        # Create spectrogram
        spectrogram = self.preprocessor.create_spectrogram(self.test_audio_path, spectrogram_path)
        
        # Check spectrogram dimensions
        self.assertIsInstance(spectrogram, np.ndarray)
        self.assertEqual(spectrogram.ndim, 2)  # 2D spectrogram
        
        # Check if the file was created
        self.assertTrue(os.path.exists(spectrogram_path))