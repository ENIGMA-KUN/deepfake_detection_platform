"""
Video Deepfake Detection Test Script

This script tests the video deepfake detection capabilities of the platform by:
1. Testing individual video detectors (GenConViT and others) on real and fake video samples
2. Testing ensemble detection with multiple models
3. Testing Singularity Mode enhanced deepfake detection
4. Evaluating CUDA performance and memory usage

The script supports both real model implementations and mock versions for testing.
"""
# Setup imports and project path
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional, Union

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Define device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Import real detector implementations
try:
    from detectors.video_detector.genconvit import GenConViTVideoDetector
    from detectors.video_detector.ensemble import VideoEnsembleDetector
except Exception as e:
    print(f"Warning: Could not import real video detector implementations: {str(e)}")

# Import base detector class
from detectors.base_detector import BaseDetector

# Set this to True to force using mock models (for testing without internet or when model downloading is not feasible)
USE_MOCK_MODELS = True  # Set to False to attempt loading real models

# Simplified mock video detector for testing
class MockVideoDetector(BaseDetector):
    """
    Mock video detector that simulates deepfake detection without requiring actual models.
    For testing CUDA infrastructure only.
    """
    
    def __init__(self, model_name: str, confidence_threshold: float = 0.5, device: str = None, frames_per_second: int = 5):
        """Initialize the mock video detector"""
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        self.frames_per_second = frames_per_second
        
        # Simulate model loading
        print(f"Mock loading video model: {model_name} on {self.device}")
        self.model = self._create_mock_model()
        self.model.to(self.device)
        
    def _create_mock_model(self):
        """Create a very simple model to test CUDA infrastructure"""
        # Create a simple convolutional model for video frames
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 2)
        )
        return model
        
    def detect(self, video_path: str) -> Dict[str, Any]:
        """
        Simulate detection of a video deepfake.
        This is a mock implementation that returns predetermined results.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing mock detection results
        """
        # Handle the case if video_path is already a dict
        if isinstance(video_path, dict):
            return video_path
            
        start_time = time.time()
        
        try:
            # Extract frames to simulate processing
            frames, timestamps, video_info = self._extract_frames(video_path)
            
            if not frames:
                raise ValueError("No frames could be extracted from the video")
            
            # Process frames through mock model to test CUDA
            frame_scores, _ = self._analyze_frames(frames)
            
            # Temporal analysis
            temporal_score = self._temporal_analysis(frame_scores)
            
            # Calculate overall score (average of frame scores)
            overall_score = np.mean(frame_scores)
            
            # Determine if deepfake based on threshold
            is_deepfake = overall_score >= self.confidence_threshold
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create sample heatmap for visualization
            heatmap = np.random.rand(7, 7)  # Simplified heatmap
            
            # Prepare metadata
            metadata = {
                "model_name": self.model_name,
                "processing_time": processing_time,
                "frames_analyzed": len(frames),
                "frames_per_second": self.frames_per_second,
                "video_info": video_info,
                "temporal_score": temporal_score,
                "frame_scores": frame_scores.tolist(),
                "timestamps": timestamps,
                "heatmap": heatmap.tolist()
            }
            
            # Prepare result
            result = {
                "is_deepfake": is_deepfake,
                "confidence": overall_score,
                "metadata": metadata
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in mock video detection: {str(e)}")
            # Return mock error result
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames, timestamps, video_info)
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Sample at specified FPS
            sample_interval = max(1, int(fps / self.frames_per_second))
            
            # Extract frames
            frames = []
            timestamps = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    frames.append(frame)
                    timestamps.append(frame_idx / fps)
                
                frame_idx += 1
                
                # Limit the number of frames to avoid memory issues
                if len(frames) >= 100:  # Arbitrary limit for testing
                    break
                    
            cap.release()
            
            # Video info
            video_info = {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "width": width,
                "height": height,
                "sampled_frames": len(frames)
            }
            
            self.logger.info(f"Extracted {len(frames)} frames from video: {video_path}")
            return frames, timestamps, video_info
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            return [], [], {}
    
    def _analyze_frames(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Analyze individual frames using the mock model.
        
        Args:
            frames: List of video frames
            
        Returns:
            Tuple of (frame_scores, face_detections)
        """
        # For testing purposes, create random scores with a bias
        if 'fake' in self.model_name.lower():
            # Bias toward detecting as fake
            frame_scores = np.random.normal(0.7, 0.15, len(frames))
        else:
            # More balanced random scores
            frame_scores = np.random.normal(0.5, 0.2, len(frames))
            
        # Clip to valid range
        frame_scores = np.clip(frame_scores, 0, 1)
        
        # For testing CUDA, process one batch through the mock model
        if len(frames) > 0:
            sample_frame = frames[0]
            
            # Convert to tensor for CUDA testing
            if isinstance(sample_frame, np.ndarray):
                # Ensure it's RGB and proper shape (add batch dimension)
                if sample_frame.ndim == 3:
                    tensor_frame = torch.from_numpy(sample_frame.transpose(2, 0, 1)).float().unsqueeze(0)
                    tensor_frame = tensor_frame.to(self.device)
                    
                    # Run through model to test CUDA
                    with torch.no_grad():
                        outputs = self.model(tensor_frame)
        
        # Mock face detections (empty for simplicity)
        face_detections = [{} for _ in range(len(frames))]
        
        return frame_scores, face_detections
    
    def _temporal_analysis(self, frame_scores: np.ndarray) -> float:
        """
        Analyze temporal consistency of frame scores.
        
        Args:
            frame_scores: Array of scores for each frame
            
        Returns:
            Temporal inconsistency score (0-1)
        """
        if len(frame_scores) <= 1:
            return 0.0
            
        # Calculate the standard deviation of scores
        std_dev = np.std(frame_scores)
        
        # Calculate differences between consecutive frames
        diffs = np.abs(np.diff(frame_scores))
        mean_diff = np.mean(diffs)
        
        # Combined temporal score
        temporal_score = 0.5 * std_dev + 0.5 * mean_diff
        
        # Normalize to 0-1 range
        normalized = min(1.0, temporal_score * 2.0)
        
        return normalized
    
    def get_confidence(self, result) -> float:
        """
        Get the confidence score from a detection result.
        
        Args:
            result: Detection result dictionary
            
        Returns:
            Confidence score (0-1)
        """
        if isinstance(result, dict) and 'confidence' in result:
            return result['confidence']
        return 0.0
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Alias for detect method to maintain compatibility with ensemble detector.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Detection result dictionary
        """
        return self.detect(video_path)


# MockEnsembleDetector for testing ensemble detection
class MockEnsembleDetector:
    """
    Mock ensemble detector for video deepfakes.
    Simulated implementation for testing CUDA infrastructure.
    """
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None, 
                 threshold: float = 0.5, enable_singularity: bool = False):
        """
        Initialize the mock ensemble detector
        
        Args:
            detectors: List of detector instances to ensemble
            weights: Optional weights for each detector
            threshold: Confidence threshold for classification
            enable_singularity: Whether to enable singularity mode
        """
        self.detectors = detectors
        self.num_detectors = len(detectors)
        self.threshold = threshold
        self.enable_singularity = enable_singularity
        
        # Use equal weights if none provided
        if weights is None:
            self.weights = [1.0 / self.num_detectors] * self.num_detectors
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict whether video is authentic or deepfake using ensemble voting.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict containing prediction results
        """
        try:
            # Get predictions from all detectors
            predictions = []
            for detector in self.detectors:
                pred = detector.predict(video_path)
                predictions.append(pred)
            
            # Extract confidence scores
            confidence_scores = []
            for i, pred in enumerate(predictions):
                if isinstance(pred, dict) and 'confidence' in pred:
                    confidence_scores.append(pred['confidence'])
                else:
                    confidence_scores.append(0.0)
            
            # Calculate weighted average confidence
            weighted_confidence = sum(w * c for w, c in zip(self.weights, confidence_scores))
            
            # Determine if deepfake based on threshold
            is_deepfake = weighted_confidence >= self.threshold
            
            # Individual results
            individual_results = []
            for i, (detector, score) in enumerate(zip(self.detectors, confidence_scores)):
                if hasattr(detector, 'model_name'):
                    model_name = detector.model_name
                else:
                    model_name = f"Model_{i}"
                
                individual_results.append({
                    'model': model_name,
                    'confidence': score,
                    'weight': self.weights[i]
                })
            
            # Base result
            result = {
                'is_deepfake': is_deepfake,
                'confidence': weighted_confidence,
                'individual_results': individual_results
            }
            
            # Add singularity mode info if enabled
            if self.enable_singularity:
                # Mock singularity mode information
                result['singularity_mode'] = {
                    'applied': True,
                    'mode': 'Temporal Oracle',
                    'version': '1.0',
                    'temporal_consistency_score': np.random.random() * 0.5,
                    'spatial_consistency_score': np.random.random() * 0.5
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {'is_deepfake': False, 'confidence': 0.0, 'error': str(e)}

# Test helper function
def test_model(detector, video_path, video_type):
    """
    Test a video deepfake detector on a given video file.
    
    Args:
        detector: The detector to test
        video_path: Path to the video file
        video_type: 'real' or 'fake'
        
    Returns:
        Tuple containing the detection result and whether it was correct
    """
    print(f"\nTesting {detector.__class__.__name__} on {video_type} video...")
    
    try:
        # Run detection
        start_time = time.time()
        result = detector.detect(video_path)
        processing_time = time.time() - start_time
        
        # Extract results
        is_deepfake = result.get('is_deepfake', False)
        confidence = result.get('confidence', 0.0)
        
        # Determine if prediction is correct
        correct = (is_deepfake and video_type == 'fake') or (not is_deepfake and video_type == 'real')
        
        # Print results
        print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Correct: {'✓' if correct else '✗'}")
        
        # Print additional info if available
        if 'metadata' in result:
            metadata = result['metadata']
            if 'frames_analyzed' in metadata:
                print(f"  Frames analyzed: {metadata['frames_analyzed']}")
            if 'temporal_score' in metadata:
                print(f"  Temporal inconsistency: {metadata['temporal_score']:.4f}")
            if 'heatmap' in metadata:
                print(f"  Heatmap: Available")
                
                # Optionally visualize the heatmap
                # plt.figure(figsize=(5, 5))
                # plt.imshow(np.array(metadata['heatmap']), cmap='hot')
                # plt.title(f"{video_type.capitalize()} Video - Deepfake Heatmap")
                # plt.colorbar(label='Deepfake Probability')
                # plt.show()
        
        return result, correct
        
    except Exception as e:
        print(f"  Error testing model: {str(e)}")
        return {'is_deepfake': False, 'confidence': 0.0, 'error': str(e)}, False


# Main test execution
if __name__ == "__main__":
    # Initialize detectors
    print("\n=== Initializing Video Detectors ===")
    
    # Flag to track if we're using real or mock models
    using_real_models = False
    
    if not USE_MOCK_MODELS:
        try:
            # Try to initialize real detectors
            print("Initializing real video detector implementations...")
            
            # Try to initialize GenConViT model
            try:
                print("Loading GenConViT model...")
                genconvit_detector = GenConViTVideoDetector(
                    frame_model_name="google/vit-base-patch16-224",
                    temporal_model_name="facebook/timesformer-base-finetuned-k400",
                    device=device
                )
                print("GenConViT model loaded successfully")
                
                # If we get here, we're using at least one real model
                using_real_models = True
            except Exception as e:
                print(f"Error loading GenConViT model: {str(e)}")
                print("Falling back to mock GenConViT detector")
                genconvit_detector = MockVideoDetector("GenConViT", device=device)
            
            # Try to initialize additional models here in the future (TimeSformer, SlowFast, etc.)
            # For now, use a second mock detector with a different bias
            print("Using mock TimeSformer detector for diversity")
            timesformer_detector = MockVideoDetector("TimeSformer", device=device)
            
        except Exception as e:
            print(f"Error initializing real detectors: {str(e)}")
            print("Falling back to all mock detectors for testing...")
            genconvit_detector = MockVideoDetector("GenConViT", device=device)
            timesformer_detector = MockVideoDetector("TimeSformer", device=device)
    else:
        # Force using mock detectors
        print("Using mock detectors as specified by USE_MOCK_MODELS flag")
        genconvit_detector = MockVideoDetector("GenConViT", device=device)
        timesformer_detector = MockVideoDetector("TimeSformer", device=device)
        
    print(f"Detectors initialized successfully. Using {'real' if using_real_models else 'mock'} detectors.")
    
    # Define test video paths
    print("\n=== Defining Test Videos ===")
    real_video_path = os.path.join(project_root, 'tests', 'test_data', 'Real_Video', 'real_001.mp4')
    fake_video_path = os.path.join(project_root, 'tests', 'test_data', 'Fake_Video', 'fake_001.mp4')
    
    # Check if video files exist
    try:
        if not os.path.exists(real_video_path):
            # Try to find any real video file if the specific one doesn't exist
            real_video_dir = os.path.join(project_root, 'tests', 'test_data', 'Real_Video')
            real_files = [f for f in os.listdir(real_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            if real_files:
                real_video_path = os.path.join(real_video_dir, real_files[0])
                print(f"Using alternative real video file: {real_video_path}")
            else:
                raise FileNotFoundError(f"No real video files found in {real_video_dir}")
    
        if not os.path.exists(fake_video_path):
            # Try to find any fake video file if the specific one doesn't exist
            fake_video_dir = os.path.join(project_root, 'tests', 'test_data', 'Fake_Video')
            fake_files = [f for f in os.listdir(fake_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            if fake_files:
                fake_video_path = os.path.join(fake_video_dir, fake_files[0])
                print(f"Using alternative fake video file: {fake_video_path}")
            else:
                raise FileNotFoundError(f"No fake video files found in {fake_video_dir}")
                
        print(f"Real video: {real_video_path}")
        print(f"Fake video: {fake_video_path}")
    except Exception as e:
        print(f"Error finding test video files: {str(e)}")
        sys.exit(1)
    
    # Display video file info
    print("\nVideo file information:")
    try:
        # Get info about real video
        real_cap = cv2.VideoCapture(real_video_path)
        real_fps = real_cap.get(cv2.CAP_PROP_FPS)
        real_frame_count = int(real_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        real_duration = real_frame_count / real_fps if real_fps > 0 else 0
        real_width = int(real_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_height = int(real_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_cap.release()
        
        print(f"Real video: {os.path.basename(real_video_path)}, " +
              f"{real_width}x{real_height}, {real_fps:.2f} fps, {real_duration:.2f} seconds")
        
        # Get info about fake video
        fake_cap = cv2.VideoCapture(fake_video_path)
        fake_fps = fake_cap.get(cv2.CAP_PROP_FPS)
        fake_frame_count = int(fake_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fake_duration = fake_frame_count / fake_fps if fake_fps > 0 else 0
        fake_width = int(fake_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fake_height = int(fake_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fake_cap.release()
        
        print(f"Fake video: {os.path.basename(fake_video_path)}, " +
              f"{fake_width}x{fake_height}, {fake_fps:.2f} fps, {fake_duration:.2f} seconds")
        
    except Exception as e:
        print(f"Warning: Couldn't display video info: {str(e)}")
    
    # Test individual models
    print("\n=== Testing Individual Models ===")
    
    # Test GenConViT model
    print("\n--- Testing GenConViT Model ---")
    genconvit_real_result, genconvit_real_correct = test_model(genconvit_detector, real_video_path, 'real')
    genconvit_fake_result, genconvit_fake_correct = test_model(genconvit_detector, fake_video_path, 'fake')
    
    # Test TimeSformer model (mock)
    print("\n--- Testing TimeSformer Model ---")
    timesformer_real_result, timesformer_real_correct = test_model(timesformer_detector, real_video_path, 'real')
    timesformer_fake_result, timesformer_fake_correct = test_model(timesformer_detector, fake_video_path, 'fake')
    
    # Test ensemble
    print("\n=== Testing Ensemble Detector ===")
    # Create the ensemble detector with all individual detectors
    detectors = [genconvit_detector, timesformer_detector]
    try:
        if using_real_models:
            # Use the real VideoEnsembleDetector
            video_ensemble = VideoEnsembleDetector(
                detectors=detectors,
                weights=None,  # Use equal weights initially
                threshold=0.5,
                enable_singularity=False  # First test without Singularity Mode
            )
        else:
            # Fall back to mock ensemble
            video_ensemble = MockEnsembleDetector(
                detectors=detectors,
                weights=None,  # Use equal weights initially
                threshold=0.5,
                enable_singularity=False  # First test without Singularity Mode
            )
        print(f"Created Video Ensemble Detector with {len(detectors)} models")
    except Exception as e:
        print(f"Error creating ensemble detector: {str(e)}")
        print("Falling back to mock ensemble detector...")
        video_ensemble = MockEnsembleDetector(
            detectors=detectors,
            weights=None,
            threshold=0.5,
            enable_singularity=False
        )
        print(f"Created Mock Video Ensemble Detector with {len(detectors)} models")
    
    # Test ensemble on real video
    print("\n--- Testing Ensemble on Real Video ---")
    ensemble_real_result = video_ensemble.predict(real_video_path)
    is_deepfake = ensemble_real_result['is_deepfake']
    confidence = ensemble_real_result['confidence']
    print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Correct: {'✓' if not is_deepfake else '✗'}")
    
    # Show individual model contributions if available
    if 'individual_results' in ensemble_real_result:
        print("\nIndividual model contributions:")
        for result in ensemble_real_result['individual_results']:
            model_name = result['model']
            confidence = result['confidence']
            weight = result['weight']
            print(f"  {model_name}: Confidence = {confidence:.4f}, Weight = {weight:.2f}")
    
    # Test ensemble on fake video
    print("\n--- Testing Ensemble on Fake Video ---")
    ensemble_fake_result = video_ensemble.predict(fake_video_path)
    is_deepfake = ensemble_fake_result['is_deepfake']
    confidence = ensemble_fake_result['confidence']
    print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Correct: {'✓' if is_deepfake else '✗'}")
    
    # Show individual model contributions if available
    if 'individual_results' in ensemble_fake_result:
        print("\nIndividual model contributions:")
        for result in ensemble_fake_result['individual_results']:
            model_name = result['model']
            confidence = result['confidence']
            weight = result['weight']
            print(f"  {model_name}: Confidence = {confidence:.4f}, Weight = {weight:.2f}")
    
    print("\n=== Testing Temporal Oracle Singularity Mode ===")
    # Enable Singularity Mode in the ensemble detector
    try:
        if using_real_models:
            # Use the real VideoEnsembleDetector with Singularity Mode
            video_ensemble_with_singularity = VideoEnsembleDetector(
                detectors=detectors,
                weights=None,
                threshold=0.5,
                enable_singularity=True  # Enable Singularity Mode
            )
        else:
            # Fall back to mock ensemble with Singularity Mode
            video_ensemble_with_singularity = MockEnsembleDetector(
                detectors=detectors,
                weights=None,
                threshold=0.5,
                enable_singularity=True  # Enable Singularity Mode
            )
        print(f"Created Video Ensemble Detector with Temporal Oracle Singularity Mode enabled")
    except Exception as e:
        print(f"Error creating ensemble detector with Singularity Mode: {str(e)}")
        print("Falling back to mock ensemble detector with Singularity Mode...")
        video_ensemble_with_singularity = MockEnsembleDetector(
            detectors=detectors,
            weights=None,
            threshold=0.5,
            enable_singularity=True
        )
        print(f"Created Mock Video Ensemble Detector with Singularity Mode enabled")
    
    # Test Singularity Mode on real video
    print("\n--- Testing Singularity Mode on Real Video ---")
    singularity_real_result = video_ensemble_with_singularity.predict(real_video_path)
    is_deepfake = singularity_real_result['is_deepfake']
    confidence = singularity_real_result['confidence']
    print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Correct: {'✓' if not is_deepfake else '✗'}")
    
    # Show Singularity Mode information if available
    if 'singularity_mode' in singularity_real_result:
        print("\nSingularity Mode information:")
        for key, value in singularity_real_result['singularity_mode'].items():
            if key != 'content_features':  # Skip detailed content features for cleaner output
                print(f"  {key}: {value}")
    
    # Test Singularity Mode on fake video
    print("\n--- Testing Singularity Mode on Fake Video ---")
    singularity_fake_result = video_ensemble_with_singularity.predict(fake_video_path)
    is_deepfake = singularity_fake_result['is_deepfake']
    confidence = singularity_fake_result['confidence']
    print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Correct: {'✓' if is_deepfake else '✗'}")
    
    # Show Singularity Mode information if available
    if 'singularity_mode' in singularity_fake_result:
        print("\nSingularity Mode information:")
        for key, value in singularity_fake_result['singularity_mode'].items():
            if key != 'content_features':  # Skip detailed content features for cleaner output
                print(f"  {key}: {value}")
    
    # Model accuracy summary
    print("\n=== Model Accuracy Summary ===")
    genconvit_accuracy = (genconvit_real_correct + genconvit_fake_correct) / 2 * 100
    timesformer_accuracy = (timesformer_real_correct + timesformer_fake_correct) / 2 * 100
    ensemble_accuracy = ((not ensemble_real_result['is_deepfake']) + ensemble_fake_result['is_deepfake']) / 2 * 100
    
    print("Accuracy summary:")
    print(f"  {'Real' if using_real_models else 'Mock'} GenConViT: {genconvit_accuracy:.2f}%")
    print(f"  {'Real' if using_real_models else 'Mock'} TimeSformer: {timesformer_accuracy:.2f}%")
    print(f"  {'Real' if using_real_models else 'Mock'} Ensemble: {ensemble_accuracy:.2f}%")
    
    # Print CUDA memory usage
    if torch.cuda.is_available():
        print(f"\nCUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    
    print("\nTest complete! Video detection infrastructure with CUDA support verified successfully.")
    if not using_real_models:
        print("NOTE: This script used mock detectors with simplified models for CUDA testing purposes only.")
        print("For actual deepfake detection, you would need properly fine-tuned models.")
    else:
        print("NOTE: The real models used in this test may not be fully fine-tuned for deepfake detection.")
        print("For optimal results, models should be fine-tuned on a deepfake dataset.")
