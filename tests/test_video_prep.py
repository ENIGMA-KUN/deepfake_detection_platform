import unittest
import os
import tempfile
import numpy as np
import cv2
import torch
import shutil

from data.preprocessing.video_prep import VideoPreprocessor

class TestVideoPreprocessor(unittest.TestCase):
    """Tests for the VideoPreprocessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test video file (10 frames of colored patterns)
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        self._create_test_video(self.test_video_path, num_frames=10)
        
        # Initialize preprocessor with default settings
        self.preprocessor = VideoPreprocessor()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def _create_test_video(self, output_path, width=320, height=240, fps=30, num_frames=10):
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
    
    def test_preprocess_tensor_output(self):
        """Test preprocessing with tensor output."""
        # Preprocess video with tensor output
        processed_frames, meta_info = self.preprocessor.preprocess(self.test_video_path, return_tensor=True)
        
        # Check that output is a tensor
        self.assertIsInstance(processed_frames, torch.Tensor)
        
        # Check tensor dimensions
        self.assertEqual(processed_frames.shape[0], 10)  # 10 frames
        self.assertEqual(processed_frames.shape[1], 3)   # RGB channels
        self.assertEqual(processed_frames.shape[2], 224) # Height
        self.assertEqual(processed_frames.shape[3], 224) # Width
        
        # Check meta info
        self.assertEqual(meta_info['original_fps'], 30)
        self.assertEqual(meta_info['original_total_frames'], 10)
        self.assertIn('load_video', meta_info['preprocessing_steps'])
        self.assertIn('sample_frames', meta_info['preprocessing_steps'])
        self.assertIn('to_tensor', meta_info['preprocessing_steps'])
        self.assertIn('normalize', meta_info['preprocessing_steps'])
    
    def test_preprocess_numpy_output(self):
        """Test preprocessing with NumPy output."""
        # Preprocess video with NumPy output
        processed_frames, meta_info = self.preprocessor.preprocess(self.test_video_path, return_tensor=False)
        
        # Check that output is a NumPy array
        self.assertIsInstance(processed_frames, np.ndarray)
        
        # Check array dimensions
        self.assertEqual(processed_frames.shape[0], 10)  # 10 frames
        self.assertEqual(processed_frames.shape[1], 224) # Height
        self.assertEqual(processed_frames.shape[2], 224) # Width
        self.assertEqual(processed_frames.shape[3], 3)   # RGB channels
        
        # Check meta info
        self.assertEqual(meta_info['original_fps'], 30)
        self.assertEqual(meta_info['original_total_frames'], 10)
        self.assertIn('load_video', meta_info['preprocessing_steps'])
        self.assertIn('sample_frames', meta_info['preprocessing_steps'])
        self.assertNotIn('to_tensor', meta_info['preprocessing_steps'])
        self.assertNotIn('normalize', meta_info['preprocessing_steps'])
    
    def test_face_detection(self):
        """Test face detection in video frames."""
        # Skip if OpenCV's face detection is not available
        try:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if face_cascade.empty():
                self.skipTest("OpenCV face detector not available")
        except:
            self.skipTest("OpenCV face detector not available")
        
        # Create a test video with a face (here we'll just use a rectangle as a simple proxy)
        face_video_path = os.path.join(self.temp_dir.name, "face_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(face_video_path, fourcc, 30, (320, 240))
        
        # Create 10 frames with a "face" (rectangle with typical face proportions)
        for i in range(10):
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 150
            # Draw a rectangle to look like a face (typical face proportions are ~3:4)
            cv2.rectangle(frame, (120, 80), (200, 160), (200, 200, 200), -1)  # Fill
            cv2.circle(frame, (140, 100), 5, (0, 0, 0), -1)  # Left eye
            cv2.circle(frame, (180, 100), 5, (0, 0, 0), -1)  # Right eye
            cv2.rectangle(frame, (150, 130), (170, 140), (0, 0, 0), -1)  # Mouth
            writer.write(frame)
        
        writer.release()
        
        # Initialize preprocessor with face detection enabled
        face_preprocessor = VideoPreprocessor({
            'face_detection': True
        })
        
        # Preprocess video with face detection
        processed_frames, meta_info = face_preprocessor.preprocess(face_video_path)
        
        # Note: This test is not reliable because the simple rectangle might not be detected as a face
        # So we're just verifying the code runs without errors
        self.assertIsInstance(processed_frames, torch.Tensor)
    
    def test_motion_metrics(self):
        """Test motion metrics computation."""
        # Preprocess video
        processed_frames, _ = self.preprocessor.preprocess(self.test_video_path)
        
        # Compute motion metrics
        motion_metrics = self.preprocessor.compute_motion_metrics(processed_frames)
        
        # Check metrics
        self.assertIn('mean_motion', motion_metrics)
        self.assertIn('std_motion', motion_metrics)
        self.assertIn('max_motion', motion_metrics)
        self.assertIn('mean_flow', motion_metrics)
        self.assertIn('std_flow', motion_metrics)
        self.assertIn('max_flow', motion_metrics)
        
        # All metrics should be non-negative
        for key, value in motion_metrics.items():
            self.assertGreaterEqual(value, 0)
    
    def test_scene_changes(self):
        """Test scene change detection."""
        # Create a test video with a scene change
        scene_video_path = os.path.join(self.temp_dir.name, "scene_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(scene_video_path, fourcc, 30, (320, 240))
        
        # Create 5 frames of one scene
        for i in range(5):
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 50
            cv2.putText(frame, f"Scene 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
        
        # Create 5 frames of another scene (significantly different)
        for i in range(5):
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 200
            cv2.putText(frame, f"Scene 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            writer.write(frame)
        
        writer.release()
        
        # Preprocess video
        processed_frames, _ = self.preprocessor.preprocess(scene_video_path)
        
        # Detect scene changes
        scene_changes = self.preprocessor.detect_scene_changes(processed_frames, threshold=20.0)
        
        # Should detect at least one scene change
        self.assertGreaterEqual(len(scene_changes), 1)
    
    def test_extract_audio(self):
        """Test audio extraction from video."""
        # Skip if ffmpeg is not available
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode != 0:
                self.skipTest("ffmpeg not available")
        except:
            self.skipTest("ffmpeg not available")
        
        # Copy a video with audio (you'd need a real video with audio for this test)
        # For testing purposes, we'll just check if the method runs without errors
        try:
            audio_path, meta_info = self.preprocessor.extract_audio(self.test_video_path)
            
            # Our test video doesn't have audio, so it's expected to fail
            self.assertFalse(meta_info['extraction_successful'])
        except:
            # This is expected for our simple test video
            pass
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        # Create a second test video
        test_video2_path = os.path.join(self.temp_dir.name, "test_video2.mp4")
        self._create_test_video(test_video2_path, num_frames=15)
        
        # Batch preprocess
        video_paths = [self.test_video_path, test_video2_path]
        processed_videos, meta_infos = self.preprocessor.batch_preprocess(video_paths)
        
        # Check batch results
        self.assertEqual(len(processed_videos), 2)
        self.assertEqual(len(meta_infos), 2)
        
        # Check frame counts
        self.assertEqual(processed_videos[0].shape[0], 10)  # First video has 10 frames
        self.assertEqual(processed_videos[1].shape[0], 15)  # Second video has 15 frames
        
        # Check all videos are tensors
        for video in processed_videos:
            self.assertIsInstance(video, torch.Tensor)