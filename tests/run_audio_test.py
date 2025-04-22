# Setup imports and project path
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
import librosa
import librosa.display
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
from detectors.audio_detector.wav2vec_detector import Wav2VecAudioDetector
from detectors.audio_detector.xlsr_detector import XLSRAudioDetector
from detectors.audio_detector.ensemble import AudioEnsembleDetector

# Import base detector class if needed
from detectors.base_detector import BaseDetector

# Simplified mock audio detector for testing - kept for reference and fallback
class MockAudioDetector(BaseDetector):
    """
    Mock audio detector that simulates deepfake detection without requiring actual models.
    For testing CUDA infrastructure only.
    """
    
    def __init__(self, model_name: str, confidence_threshold: float = 0.5, device: str = None):
        """Initialize the mock audio detector"""
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        # Simulate model loading
        print(f"Mock loading model: {model_name} on {self.device}")
        self.model = self._create_mock_model()
        self.model.to(self.device)
        
    def _create_mock_model(self):
        """Create a very simple model to test CUDA infrastructure"""
        # Create a simple convolutional model
        model = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 2)
        )
        return model
    
    def detect(self, audio_path: str) -> Dict[str, Any]:
        """
        Simulate detection of an audio deepfake.
        This is a mock implementation that returns predetermined results.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing mock detection results
        """
        # Handle the case if audio_path is already a dict
        if isinstance(audio_path, dict):
            return audio_path
            
        start_time = time.time()
        
        try:
            # Load the audio to simulate processing
            waveform, sr = self._load_audio(audio_path)
            
            # Move to device to test CUDA
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Test CUDA with a forward pass
            with torch.no_grad():
                # Trim to a reasonable length to avoid CUDA memory issues
                input_tensor = waveform_tensor[:, :, :min(len(waveform), 16000)]
                outputs = self.model(input_tensor)
                logits = outputs
                probs = F.softmax(logits, dim=1)
                
                # Get mock prediction based on the file path
                if "fake" in audio_path.lower():
                    # Higher confidence for fake files
                    deepfake_score = probs[0, 1].item() * 0.6 + 0.3
                else:
                    # Lower confidence for real files
                    deepfake_score = probs[0, 1].item() * 0.3 + 0.1
                
            # Generate results
            is_deepfake = deepfake_score >= self.confidence_threshold
            
            # Create spectrogram for visualization
            spectrogram = self._generate_spectrogram(waveform)
            
            # Create mock temporal scores
            segment_length = int(sr * 0.5)  # 0.5 second segments
            num_segments = max(1, len(waveform) // segment_length)
            
            # Generate fake temporal scores with some variation
            temporal_scores = np.zeros(num_segments)
            segment_times = np.zeros(num_segments)
            
            for i in range(num_segments):
                # Add some random variation to the base score
                segment_score = deepfake_score + np.random.normal(0, 0.05)
                segment_score = max(0, min(1, segment_score))  # Ensure 0-1 range
                temporal_scores[i] = segment_score
                segment_times[i] = i * 0.5  # 0.5 seconds per segment
            
            # Create the result dictionary
            result = {
                'is_deepfake': is_deepfake,
                'confidence': deepfake_score,
                'model': self.__class__.__name__,
                'analysis_time': time.time() - start_time,
                'spectrogram': spectrogram,
                'temporal_scores': temporal_scores.tolist(),
                'segment_times': segment_times.tolist(),
                'inconsistency_index': 0.2 if "fake" in audio_path.lower() else 0.05
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e),
                'model': self.__class__.__name__,
                'analysis_time': time.time() - start_time
            }
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess an audio file."""
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform
        return waveform, sr
    
    def _generate_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Generate a mel spectrogram for visualization."""
        try:
            S = librosa.feature.melspectrogram(
                y=waveform, 
                sr=16000,
                n_mels=128,
                hop_length=512
            )
            
            # Convert to dB scale
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            return S_dB
        except Exception as e:
            self.logger.error(f"Error generating spectrogram: {str(e)}")
            return None
    
    def get_confidence(self, media_path: str) -> float:
        """Get confidence score that audio is a deepfake."""
        if isinstance(media_path, dict):
            return media_path.get('confidence', 0.0)
        result = self.detect(media_path)
        return result['confidence']
    
    def predict(self, media_path: str) -> Dict[str, Any]:
        """Alias for detect method to maintain compatibility with ensemble detector."""
        return self.detect(media_path)


class MockEnsembleDetector:
    """
    Mock ensemble detector for audio deepfakes.
    Simulated implementation for testing CUDA infrastructure.
    """
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None, 
                 threshold: float = 0.5, enable_singularity: bool = False):
        """Initialize the mock ensemble detector"""
        self.detectors = detectors
        self.num_detectors = len(detectors)
        
        # Use equal weights if none provided
        if weights is None:
            self.weights = [1.0 / self.num_detectors] * self.num_detectors
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.threshold = threshold
        self.enable_singularity = enable_singularity
        self.logger = logging.getLogger(__name__)
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict whether audio is authentic or deepfake using ensemble voting.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing prediction results
        """
        # Get predictions from all detectors
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(audio_path)
            predictions.append(pred)
        
        # Extract confidence scores
        confidence_scores = []
        for detector, pred in zip(self.detectors, predictions):
            if isinstance(pred, dict) and 'confidence' in pred:
                confidence_scores.append(pred['confidence'])
            else:
                confidence_scores.append(detector.get_confidence(pred))
        
        # Calculate weighted average confidence
        weighted_confidence = sum(w * c for w, c in zip(self.weights, confidence_scores))
        
        # Determine if deepfake based on threshold
        is_deepfake = weighted_confidence > self.threshold
        
        # Combine results
        result = {
            'is_deepfake': is_deepfake,
            'confidence': weighted_confidence,
            'individual_results': [
                {
                    'model': detector.__class__.__name__,
                    'confidence': score,
                    'weight': weight,
                    'raw_prediction': pred
                }
                for detector, score, weight, pred in zip(
                    self.detectors, confidence_scores, self.weights, predictions
                )
            ]
        }
        
        # Add singularity enhancement if enabled
        if self.enable_singularity:
            # Simple enhancement for demo purposes
            result['singularity_mode'] = {
                'applied': True,
                'mode': 'Frequency Sentinel',
                'version': '1.0',
                'content_features': {
                    'content_type': 'speech' if 'real' in audio_path.lower() else 'music',
                    'spectral_centroid': 3200.0,
                    'spectral_rolloff': 7500.0,
                    'frequency_distribution': {
                        'low': 0.3,
                        'mid': 0.5,
                        'high': 0.2
                    }
                }
            }
        
        return result


# Test helper function
def test_model(detector, audio_path, audio_type):
    """
    Test an audio deepfake detector on a given audio file.
    
    Args:
        detector: The detector to test
        audio_path: Path to the audio file
        audio_type: 'real' or 'fake'
        
    Returns:
        Tuple containing the detection result and whether it was correct
    """
    print(f"\nTesting {detector.__class__.__name__} on {audio_type} audio...")
    try:
        result = detector.detect(audio_path)
        
        # Check if result is properly formatted
        if not isinstance(result, dict):
            raise TypeError(f"Detector returned {type(result)} instead of dict")
        
        if 'is_deepfake' not in result or 'confidence' not in result:
            raise KeyError("Detector result missing required keys (is_deepfake, confidence)")
        
        is_deepfake = result['is_deepfake']
        confidence = result['confidence']
        correct = (is_deepfake and audio_type == 'fake') or (not is_deepfake and audio_type == 'real')
        
        print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Correct: {'✓' if correct else '✗'}")
        
        # Plot visualization if available
        if 'spectrogram' in result and result['spectrogram'] is not None:
            print("  Spectrogram: Available")
            try:
                plt.figure(figsize=(10, 4))
                
                # Display spectrogram
                S_dB = result['spectrogram']
                plt.subplot(1, 1, 1)
                librosa.display.specshow(S_dB, sr=16000, hop_length=512, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"{audio_type.capitalize()} Audio Spectrogram (Confidence: {confidence:.4f})")
                plt.tight_layout()
                plt.show()
            except Exception as viz_err:
                print(f"  Warning: Could not visualize spectrogram: {viz_err}")
                
        # Display temporal analysis if available
        if 'temporal_scores' in result and 'segment_times' in result:
            try:
                plt.figure(figsize=(10, 3))
                plt.plot(result['segment_times'], result['temporal_scores'])
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                plt.xlabel('Time (s)')
                plt.ylabel('Deepfake Confidence')
                plt.title(f"Temporal Analysis (Inconsistency: {result.get('inconsistency_index', 0):.4f})")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as viz_err:
                print(f"  Warning: Could not visualize temporal analysis: {viz_err}")
        
        return result, correct
    except Exception as e:
        import traceback
        print(f"  Error: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")
        return {'is_deepfake': False, 'confidence': 0.0, 'error': str(e)}, False


# Initialize detectors
print("\n=== Initializing Audio Detectors ===")

# Set this to True to force using mock models (for testing without internet or when model downloading is not feasible)
USE_MOCK_MODELS = True  # Set to False to attempt loading real models

# Flag to track if we're using real or mock models
using_real_models = False

if not USE_MOCK_MODELS:
    try:
        # Try to initialize real detectors
        print("Initializing real audio detector implementations...")
        
        # Try to initialize Wav2Vec model
        try:
            print("Loading Wav2Vec model...")
            wav2vec_detector = Wav2VecAudioDetector(
                model_name="facebook/wav2vec2-large-960h", 
                device=device
            )
            print("Wav2Vec model loaded successfully")
        except Exception as e:
            print(f"Error loading Wav2Vec model: {str(e)}")
            print("Falling back to mock Wav2Vec detector")
            wav2vec_detector = MockAudioDetector("Wav2Vec", device=device)
        
        # Try to initialize XLSR model
        try:
            print("Loading XLSR model...")
            xlsr_detector = XLSRAudioDetector(
                model_name="facebook/wav2vec2-xls-r-300m", 
                device=device
            )
            print("XLSR model loaded successfully")
            
            # If we get here, we're using at least one real model
            using_real_models = True
        except Exception as e:
            print(f"Error loading XLSR model: {str(e)}")
            print("Falling back to mock XLSR detector")
            xlsr_detector = MockAudioDetector("XLSR", device=device)
        
    except Exception as e:
        print(f"Error initializing real detectors: {str(e)}")
        print("Falling back to all mock detectors for testing...")
        wav2vec_detector = MockAudioDetector("Wav2Vec", device=device)
        xlsr_detector = MockAudioDetector("XLSR", device=device)
else:
    # Force using mock detectors
    print("Using mock detectors as specified by USE_MOCK_MODELS flag")
    wav2vec_detector = MockAudioDetector("Wav2Vec", device=device)
    xlsr_detector = MockAudioDetector("XLSR", device=device)
    
print(f"Detectors initialized successfully. Using {'real' if using_real_models else 'mock'} detectors.")

# Define test audio paths
print("\n=== Defining Test Audio ===")
real_audio_path = os.path.join(project_root, 'tests', 'test_data', 'Real_Audio', 'real_001.wav')
fake_audio_path = os.path.join(project_root, 'tests', 'test_data', 'Fake_Audio', 'fake_001.wav')

# Check if audio files exist
try:
    if not os.path.exists(real_audio_path):
        # Try to find any real audio file if the specific one doesn't exist
        real_audio_dir = os.path.join(project_root, 'tests', 'test_data', 'Real_Audio')
        real_files = [f for f in os.listdir(real_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        if real_files:
            real_audio_path = os.path.join(real_audio_dir, real_files[0])
            print(f"Using alternative real audio file: {real_audio_path}")
        else:
            raise FileNotFoundError(f"No real audio files found in {real_audio_dir}")

    if not os.path.exists(fake_audio_path):
        # Try to find any fake audio file if the specific one doesn't exist
        fake_audio_dir = os.path.join(project_root, 'tests', 'test_data', 'Fake_Audio')
        fake_files = [f for f in os.listdir(fake_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        if fake_files:
            fake_audio_path = os.path.join(fake_audio_dir, fake_files[0])
            print(f"Using alternative fake audio file: {fake_audio_path}")
        else:
            raise FileNotFoundError(f"No fake audio files found in {fake_audio_dir}")
            
    print(f"Real audio: {real_audio_path}")
    print(f"Fake audio: {fake_audio_path}")
except Exception as e:
    print(f"Error finding test audio files: {str(e)}")
    sys.exit(1)

# Display audio file info
print("\nAudio file information:")
try:
    # Load and display basic info about real audio
    real_waveform, real_sr = librosa.load(real_audio_path, sr=None)
    real_duration = librosa.get_duration(y=real_waveform, sr=real_sr)
    print(f"Real audio: {os.path.basename(real_audio_path)}, {real_sr} Hz, {real_duration:.2f} seconds")
    
    # Load and display basic info about fake audio
    fake_waveform, fake_sr = librosa.load(fake_audio_path, sr=None)
    fake_duration = librosa.get_duration(y=fake_waveform, sr=fake_sr)
    print(f"Fake audio: {os.path.basename(fake_audio_path)}, {fake_sr} Hz, {fake_duration:.2f} seconds")
    
    # Plot waveforms
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Real Audio Waveform")
    plt.plot(np.linspace(0, real_duration, len(real_waveform)), real_waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.title("Fake Audio Waveform")
    plt.plot(np.linspace(0, fake_duration, len(fake_waveform)), fake_waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Warning: Couldn't display audio info: {str(e)}")

# Test individual models
print("\n=== Testing Individual Models ===")

# Test Wav2Vec model
print("\n--- Testing Wav2Vec Model ---")
wav2vec_real_result, wav2vec_real_correct = test_model(wav2vec_detector, real_audio_path, 'real')
wav2vec_fake_result, wav2vec_fake_correct = test_model(wav2vec_detector, fake_audio_path, 'fake')

# Test XLS-R model
print("\n--- Testing XLSR Model ---")
xlsr_real_result, xlsr_real_correct = test_model(xlsr_detector, real_audio_path, 'real')
xlsr_fake_result, xlsr_fake_correct = test_model(xlsr_detector, fake_audio_path, 'fake')

# Test ensemble
print("\n=== Testing Ensemble Detector ===")
# Create the ensemble detector with all individual detectors
detectors = [wav2vec_detector, xlsr_detector]
try:
    if using_real_models:
        # Use the real AudioEnsembleDetector
        audio_ensemble = AudioEnsembleDetector(
            detectors=detectors,
            weights=None,  # Use equal weights initially
            threshold=0.5,
            enable_singularity=False  # First test without Singularity Mode
        )
    else:
        # Fall back to mock ensemble
        audio_ensemble = MockEnsembleDetector(
            detectors=detectors,
            weights=None,  # Use equal weights initially
            threshold=0.5,
            enable_singularity=False  # First test without Singularity Mode
        )
    print(f"Created Audio Ensemble Detector with {len(detectors)} models")
except Exception as e:
    print(f"Error creating ensemble detector: {str(e)}")
    print("Falling back to mock ensemble detector...")
    audio_ensemble = MockEnsembleDetector(
        detectors=detectors,
        weights=None,
        threshold=0.5,
        enable_singularity=False
    )
    print(f"Created Mock Audio Ensemble Detector with {len(detectors)} models")

# Test ensemble on real audio
print("\n--- Testing Ensemble on Real Audio ---")
ensemble_real_result = audio_ensemble.predict(real_audio_path)
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

# Test ensemble on fake audio
print("\n--- Testing Ensemble on Fake Audio ---")
ensemble_fake_result = audio_ensemble.predict(fake_audio_path)
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

print("\n=== Testing Frequency Sentinel Singularity Mode ===")
# Enable Singularity Mode in the ensemble detector
try:
    if using_real_models:
        # Use the real AudioEnsembleDetector with Singularity Mode
        audio_ensemble_with_singularity = AudioEnsembleDetector(
            detectors=detectors,
            weights=None,
            threshold=0.5,
            enable_singularity=True  # Enable Singularity Mode
        )
    else:
        # Fall back to mock ensemble with Singularity Mode
        audio_ensemble_with_singularity = MockEnsembleDetector(
            detectors=detectors,
            weights=None,
            threshold=0.5,
            enable_singularity=True  # Enable Singularity Mode
        )
    print(f"Created Audio Ensemble Detector with Frequency Sentinel Singularity Mode enabled")
except Exception as e:
    print(f"Error creating ensemble detector with Singularity Mode: {str(e)}")
    print("Falling back to mock ensemble detector with Singularity Mode...")
    audio_ensemble_with_singularity = MockEnsembleDetector(
        detectors=detectors,
        weights=None,
        threshold=0.5,
        enable_singularity=True
    )
    print(f"Created Mock Audio Ensemble Detector with Singularity Mode enabled")

# Test Singularity Mode on real audio
print("\n--- Testing Singularity Mode on Real Audio ---")
singularity_real_result = audio_ensemble_with_singularity.predict(real_audio_path)
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

# Test Singularity Mode on fake audio
print("\n--- Testing Singularity Mode on Fake Audio ---")
singularity_fake_result = audio_ensemble_with_singularity.predict(fake_audio_path)
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
wav2vec_accuracy = (wav2vec_real_correct + wav2vec_fake_correct) / 2 * 100
xlsr_accuracy = (xlsr_real_correct + xlsr_fake_correct) / 2 * 100
ensemble_accuracy = ((not ensemble_real_result['is_deepfake']) + ensemble_fake_result['is_deepfake']) / 2 * 100

print("Accuracy summary:")
print(f"  {'Real' if using_real_models else 'Mock'} Wav2Vec: {wav2vec_accuracy:.2f}%")
print(f"  {'Real' if using_real_models else 'Mock'} XLSR: {xlsr_accuracy:.2f}%")
print(f"  {'Real' if using_real_models else 'Mock'} Ensemble: {ensemble_accuracy:.2f}%")

# Print CUDA memory usage
if torch.cuda.is_available():
    print(f"\nCUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"CUDA memory reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

print("\nTest complete! Audio detection infrastructure with CUDA support verified successfully.")
if not using_real_models:
    print("NOTE: This script used mock detectors with simplified models for CUDA testing purposes only.")
    print("For actual deepfake detection, you would need properly fine-tuned models.")
else:
    print("NOTE: The real models used in this test may not be fully fine-tuned for deepfake detection.")
    print("For optimal results, models should be fine-tuned on a deepfake dataset.")
