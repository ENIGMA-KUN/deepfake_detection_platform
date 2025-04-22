# Setup imports and project path
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import time
from typing import Dict, Any, List, Tuple, Optional, Union

# Add the project root to the path - fix the path issue
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for the image detector modules
# Temporarily patch the transformers imports in detector modules
import importlib
from detectors.base_detector import BaseDetector
from detectors.ensemble_detector import EnsembleDetector

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Define our patched detector versions to avoid import issues
class ViTImageDetector(BaseDetector):
    """Vision Transformer (ViT) based detector for image deepfakes."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", 
                 confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize the ViT-based image detector.
        
        Args:
            model_name: Name or path of the pretrained ViT model to use
            confidence_threshold: Threshold for classifying as deepfake
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.feature_extractor = None
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the ViT model for deepfake detection."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            
            self.logger.info(f"Loading ViT model: {self.model_name}")
            
            # Load the feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move model to the specified device
            self.model.to(self.device)
            self.logger.info("ViT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading ViT model: {str(e)}")
            raise
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image is a deepfake using the ViT model.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dict containing detection results including:
            - is_deepfake: Boolean indicating whether the image is classified as a deepfake
            - confidence: Confidence score for the prediction
            - heatmap: Attention visualization if available
        """
        start_time = time.time()
        
        # If image_path is already a dict (result from another detector), return it
        if isinstance(image_path, dict):
            return image_path
            
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            
            # Extract features
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
                # Assume label 1 is for deepfake, 0 for real
                deepfake_confidence = probs[0, 1].item()
                is_deepfake = deepfake_confidence >= self.confidence_threshold
            
            # Create result dictionary
            result = {
                'is_deepfake': is_deepfake,
                'confidence': deepfake_confidence,
                'model': "ViTImageDetector",
                'analysis_time': time.time() - start_time
            }
            
            # Get attention heatmap if available
            try:
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Create a simple visualization from attention maps
                    attn = outputs.attentions[-1].mean(dim=1).mean(dim=0)  # Average over batch and heads
                    attn_map = attn.detach().cpu().numpy()
                    
                    # Reshape to square image
                    heatmap_size = int(np.sqrt(attn_map.shape[0]))
                    heatmap = attn_map.reshape(heatmap_size, heatmap_size)
                    
                    # Normalize for visualization
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    
                    # Add to result
                    result['heatmap'] = heatmap
            except Exception as e:
                self.logger.warning(f"Could not generate heatmap: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e),
                'model': "ViTImageDetector",
                'analysis_time': time.time() - start_time
            }
    
    def get_confidence(self, image_path: str) -> float:
        """Get confidence score that an image is a deepfake."""
        result = self.detect(image_path)
        return result['confidence']
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Alias for detect method to maintain compatibility with ensemble detector."""
        return self.detect(image_path)


class BEITImageDetector(BaseDetector):
    """BEIT-based detector for image deepfakes."""
    
    def __init__(self, model_name: str = "microsoft/beit-base-patch16-224-pt22k-ft22k", 
                 confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize the BEIT-based image detector.
        
        Args:
            model_name: Name or path of the pretrained BEIT model to use
            confidence_threshold: Threshold for classifying as deepfake
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the BEIT model for deepfake detection."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            
            self.logger.info(f"Loading BEiT model: {self.model_name}")
            
            # Load BEiT model for image classification
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move model to the specified device
            self.model.to(self.device)
            self.logger.info("BEiT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading BEiT model: {str(e)}")
            raise
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image is a deepfake using the BEIT model.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dict containing detection results
        """
        start_time = time.time()
        
        # If image_path is already a dict (result from another detector), return it
        if isinstance(image_path, dict):
            return image_path
            
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            
            # Process image with BEIT processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
                # Assume label 1 is for deepfake, 0 for real
                deepfake_confidence = probs[0, 1].item()
                is_deepfake = deepfake_confidence >= self.confidence_threshold
            
            # Create result dictionary
            result = {
                'is_deepfake': is_deepfake,
                'confidence': deepfake_confidence,
                'model': "BEITImageDetector",
                'analysis_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e),
                'model': "BEITImageDetector",
                'analysis_time': time.time() - start_time
            }
    
    def get_confidence(self, image_path: str) -> float:
        """Get confidence score that an image is a deepfake."""
        result = self.detect(image_path)
        return result['confidence']
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Alias for detect method to maintain compatibility with ensemble detector."""
        return self.detect(image_path)


class DeiTImageDetector(BaseDetector):
    """DeiT-based detector for image deepfakes."""
    
    def __init__(self, model_name: str = "facebook/deit-base-distilled-patch16-224", 
                 confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize the DeiT-based image detector.
        
        Args:
            model_name: Name or path of the pretrained DeiT model to use
            confidence_threshold: Threshold for classifying as deepfake
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the DeiT model for deepfake detection."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            
            self.logger.info(f"Loading DeiT model: {self.model_name}")
            
            # Load DeiT model for image classification
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move model to the specified device
            self.model.to(self.device)
            self.logger.info("DeiT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading DeiT model: {str(e)}")
            raise
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image is a deepfake using the DeiT model.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dict containing detection results
        """
        start_time = time.time()
        
        # If image_path is already a dict (result from another detector), return it
        if isinstance(image_path, dict):
            return image_path
            
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            
            # Process image with DeiT processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
                # Assume label 1 is for deepfake, 0 for real
                deepfake_confidence = probs[0, 1].item()
                is_deepfake = deepfake_confidence >= self.confidence_threshold
            
            # Create result dictionary
            result = {
                'is_deepfake': is_deepfake,
                'confidence': deepfake_confidence,
                'model': "DeiTImageDetector",
                'analysis_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e),
                'model': "DeiTImageDetector",
                'analysis_time': time.time() - start_time
            }
    
    def get_confidence(self, image_path: str) -> float:
        """Get confidence score that an image is a deepfake."""
        result = self.detect(image_path)
        return result['confidence']
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Alias for detect method to maintain compatibility with ensemble detector."""
        return self.detect(image_path)


class SwinImageDetector(BaseDetector):
    """Swin Transformer-based detector for image deepfakes."""
    
    def __init__(self, model_name: str = "microsoft/swin-base-patch4-window7-224-in22k", 
                 confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize the Swin Transformer-based image detector.
        
        Args:
            model_name: Name or path of the pretrained Swin model to use
            confidence_threshold: Threshold for classifying as deepfake
            device: Device to run the model on (cuda or cpu)
        """
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the Swin model for deepfake detection."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            
            self.logger.info(f"Loading Swin model: {self.model_name}")
            
            # Load Swin model for image classification
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move model to the specified device
            self.model.to(self.device)
            self.logger.info("Swin model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Swin model: {str(e)}")
            raise
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image is a deepfake using the Swin model.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Dict containing detection results
        """
        start_time = time.time()
        
        # If image_path is already a dict (result from another detector), return it
        if isinstance(image_path, dict):
            return image_path
            
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            
            # Process image with Swin processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
                # Assume label 1 is for deepfake, 0 for real
                deepfake_confidence = probs[0, 1].item()
                is_deepfake = deepfake_confidence >= self.confidence_threshold
            
            # Create result dictionary
            result = {
                'is_deepfake': is_deepfake,
                'confidence': deepfake_confidence,
                'model': "SwinImageDetector",
                'analysis_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e),
                'model': "SwinImageDetector",
                'analysis_time': time.time() - start_time
            }
    
    def get_confidence(self, image_path: str) -> float:
        """Get confidence score that an image is a deepfake."""
        result = self.detect(image_path)
        return result['confidence']
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Alias for detect method to maintain compatibility with ensemble detector."""
        return self.detect(image_path)


class ImageEnsembleDetector(EnsembleDetector):
    """
    Specialized ensemble detector for image deepfakes.
    
    Combines multiple image deepfake detection models with content-aware
    adaptive weighting and supports Visual Sentinel Singularity Mode integration.
    """
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None, 
                 threshold: float = 0.5, enable_singularity: bool = False):
        """
        Initialize the image ensemble detector.
        
        Args:
            detectors: List of detector instances to ensemble
            weights: Optional weights for each detector (default: equal weights)
            threshold: Confidence threshold for classifying as deepfake
            enable_singularity: Whether to enable Singularity Mode enhancement
        """
        super().__init__(detectors, weights, threshold)
        self.enable_singularity = enable_singularity
        self.logger = logging.getLogger(__name__)
        
        # Map detector names to model types for adaptive weighting
        self.detector_types = {}
        for detector in detectors:
            if hasattr(detector, 'model_name'):
                model_name = detector.model_name.lower()
                if 'vit' in model_name:
                    self.detector_types[detector] = 'vit'
                elif 'beit' in model_name:
                    self.detector_types[detector] = 'beit'
                elif 'deit' in model_name:
                    self.detector_types[detector] = 'deit'
                elif 'swin' in model_name:
                    self.detector_types[detector] = 'swin'
                else:
                    self.detector_types[detector] = 'other'
    
    def predict(self, image):
        """
        Predict whether an image is authentic or deepfake using the ensemble.
        
        Args:
            image: The image to analyze (path or array)
            
        Returns:
            Dict containing prediction results
        """
        # Get base ensemble prediction
        result = super().predict(image)
        
        # Include the image data for potential Singularity Mode enhancement
        if isinstance(image, str):
            # If image is a path, we include it
            result['image_path'] = image
        
        # Apply Visual Sentinel enhancement if enabled
        if self.enable_singularity:
            try:
                enhanced_result = self._enhance_with_singularity(image, result)
                return enhanced_result
            except Exception as e:
                self.logger.warning(f"Error applying Singularity Mode: {str(e)}")
                # Fall back to standard ensemble result
                return result
        
        return result
    
    def _enhance_with_singularity(self, image, result):
        """
        Apply Visual Sentinel Singularity Mode enhancement to the result.
        Simple placeholder implementation for the notebook.
        """
        # This is a placeholder implementation
        result['singularity_mode'] = {
            'applied': True,
            'mode': 'Visual Sentinel',
            'version': '1.0',
            'confidence_boost': 0.1
        }
        
        # Boost confidence for demonstration
        if result['is_deepfake']:
            result['confidence'] = min(1.0, result['confidence'] + 0.1)
        else:
            result['confidence'] = max(0.0, result['confidence'] - 0.1)
            
        return result
    
    def _get_visualization_data(self, predictions):
        """
        Generate visualization data for image predictions.
        
        Args:
            predictions: List of prediction results from individual detectors
            
        Returns:
            Dict with visualization data
        """
        # Extract heatmaps from individual predictions if available
        heatmaps = {}
        for i, (detector, pred) in enumerate(zip(self.detectors, predictions)):
            if hasattr(detector, 'name'):
                detector_name = detector.name
            else:
                detector_name = f"Detector_{i}"
                
            # Extract heatmap if available
            if isinstance(pred, dict) and 'heatmap' in pred:
                heatmaps[detector_name] = pred['heatmap']
        
        # Combine heatmaps with weights if available
        if heatmaps:
            # Get first heatmap dimensions
            first_heatmap = next(iter(heatmaps.values()))
            combined_heatmap = np.zeros_like(first_heatmap)
            
            for det_name, heatmap in heatmaps.items():
                # Find detector index to get its weight
                for i, det in enumerate(self.detectors):
                    if hasattr(det, 'name') and det.name == det_name:
                        weight = self.weights[i]
                        break
                else:
                    weight = 1.0 / len(heatmaps)
                
                combined_heatmap += heatmap * weight
                
            # Normalize combined heatmap
            if np.max(combined_heatmap) > 0:
                combined_heatmap /= np.max(combined_heatmap)
                
            return {'heatmap': combined_heatmap, 'individual_heatmaps': heatmaps}
        
        return {}

# Define the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("\n=== Initializing Image Detectors ===")
# Initialize the image detectors with proper error handling
vit_detector = None
beit_detector = None
deit_detector = None
swin_detector = None

try:
    print("Loading ViT model...")
    vit_detector = ViTImageDetector(
        model_name="google/vit-base-patch16-224", 
        confidence_threshold=0.5,
        device=device
    )
    
    print("\nLoading BEIT model...")
    beit_detector = BEITImageDetector(
        model_name="microsoft/beit-base-patch16-224-pt22k-ft22k", 
        confidence_threshold=0.5,
        device=device
    )
    
    print("\nLoading DeiT model...")
    deit_detector = DeiTImageDetector(
        model_name="facebook/deit-base-distilled-patch16-224", 
        confidence_threshold=0.5,
        device=device
    )
    
    print("\nLoading Swin model...")
    swin_detector = SwinImageDetector(
        model_name="microsoft/swin-base-patch4-window7-224-in22k", 
        confidence_threshold=0.5,
        device=device
    )
    
    print("All models loaded successfully!")
    
except Exception as e:
    import traceback
    print(f"Error loading models: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)  # Exit if model loading fails

# Verify models are loaded
if not all([vit_detector, beit_detector, deit_detector, swin_detector]):
    print("Error: Not all models were successfully loaded")
    sys.exit(1)

print("\n=== Defining Test Images ===")
# Define test images
real_image_path = os.path.join(project_root, 'tests', 'test_data', 'Real_Image', 'real_993.jpg')
fake_image_path = os.path.join(project_root, 'tests', 'test_data', 'Fake_Image', 'fake_999.jpg')

# Check if images exist
if not os.path.exists(real_image_path):
    raise FileNotFoundError(f"Real image not found: {real_image_path}")
if not os.path.exists(fake_image_path):
    raise FileNotFoundError(f"Fake image not found: {fake_image_path}")

# Display the test images
print("Displaying test images...")
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(Image.open(real_image_path))
axs[0].set_title("Real Image")
axs[0].axis('off')
axs[1].imshow(Image.open(fake_image_path))
axs[1].set_title("Fake Image")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# Helper function to test models on images
def test_model(detector, image_path, image_type):
    print(f"\nTesting {detector.__class__.__name__} on {image_type} image...")
    try:
        result = detector.detect(image_path)
        
        # Check if result is properly formatted
        if not isinstance(result, dict):
            raise TypeError(f"Detector returned {type(result)} instead of dict")
        
        if 'is_deepfake' not in result or 'confidence' not in result:
            raise KeyError("Detector result missing required keys (is_deepfake, confidence)")
        
        is_deepfake = result['is_deepfake']
        confidence = result['confidence']
        correct = (is_deepfake and image_type == 'fake') or (not is_deepfake and image_type == 'real')
        
        print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Correct: {'✓' if correct else '✗'}")
        
        if 'heatmap' in result and result['heatmap'] is not None:
            print("  Heatmap: Available")
            try:
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(Image.open(image_path))
                plt.title(f"{image_type.capitalize()} Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(result['heatmap'], cmap='hot')
                plt.title(f"Heatmap (Confidence: {confidence:.4f})")
                plt.axis('off')
                plt.colorbar()
                plt.tight_layout()
                plt.show()
            except Exception as viz_err:
                print(f"  Warning: Could not visualize heatmap: {viz_err}")
        
        return result, correct
    except Exception as e:
        import traceback
        print(f"  Error: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")
        return {'is_deepfake': False, 'confidence': 0.0, 'error': str(e)}, False

print("\n=== Testing Individual Models ===")
# Test ViT model
print("\n--- Testing ViT Model ---")
vit_real_result, vit_real_correct = test_model(vit_detector, real_image_path, 'real')
vit_fake_result, vit_fake_correct = test_model(vit_detector, fake_image_path, 'fake')

# Test BEIT model
print("\n--- Testing BEIT Model ---")
beit_real_result, beit_real_correct = test_model(beit_detector, real_image_path, 'real')
beit_fake_result, beit_fake_correct = test_model(beit_detector, fake_image_path, 'fake')

# Test DeiT model
print("\n--- Testing DeiT Model ---")
deit_real_result, deit_real_correct = test_model(deit_detector, real_image_path, 'real')
deit_fake_result, deit_fake_correct = test_model(deit_detector, fake_image_path, 'fake')

# Test Swin model
print("\n--- Testing Swin Model ---")
swin_real_result, swin_real_correct = test_model(swin_detector, real_image_path, 'real')
swin_fake_result, swin_fake_correct = test_model(swin_detector, fake_image_path, 'fake')

print("\n=== Testing Ensemble Detector ===")
# Create the ensemble detector with all individual detectors
detectors = [vit_detector, beit_detector, deit_detector, swin_detector]
image_ensemble = ImageEnsembleDetector(
    detectors=detectors,
    weights=None,  # Use equal weights initially
    threshold=0.5,
    enable_singularity=False  # First test without Singularity Mode
)

print(f"Created Image Ensemble Detector with {len(detectors)} models")

# Test ensemble on real image
print("\n--- Testing Ensemble on Real Image ---")
ensemble_real_result = image_ensemble.predict(real_image_path)
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

# Test ensemble on fake image
print("\n--- Testing Ensemble on Fake Image ---")
ensemble_fake_result = image_ensemble.predict(fake_image_path)
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

print("\n=== Testing Visual Sentinel Singularity Mode ===")
# Enable Singularity Mode in the ensemble detector
image_ensemble_with_singularity = ImageEnsembleDetector(
    detectors=detectors,
    weights=None,
    threshold=0.5,
    enable_singularity=True  # Enable Singularity Mode
)

print(f"Created Image Ensemble Detector with Visual Sentinel Singularity Mode enabled")

# Test Singularity Mode on real image
print("\n--- Testing Singularity Mode on Real Image ---")
try:
    singularity_real_result = image_ensemble_with_singularity.predict(real_image_path)
    is_deepfake = singularity_real_result['is_deepfake']
    confidence = singularity_real_result['confidence']
    print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Correct: {'✓' if not is_deepfake else '✗'}")
    
    # Show Singularity Mode information if available
    if 'singularity_mode' in singularity_real_result:
        sm_info = singularity_real_result['singularity_mode']
        print("\nSingularity Mode information:")
        for key, value in sm_info.items():
            print(f"  {key}: {value}")
            
    # Display enhanced heatmap if available
    if 'heatmap' in singularity_real_result:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(real_image_path))
        plt.title("Real Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(singularity_real_result['heatmap'], cmap='hot')
        plt.title(f"Enhanced Heatmap (Confidence: {confidence:.4f})")
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"Error testing Visual Sentinel: {str(e)}")

# Test Singularity Mode on fake image
print("\n--- Testing Singularity Mode on Fake Image ---")
try:
    singularity_fake_result = image_ensemble_with_singularity.predict(fake_image_path)
    is_deepfake = singularity_fake_result['is_deepfake']
    confidence = singularity_fake_result['confidence']
    print(f"  Prediction: {'Deepfake' if is_deepfake else 'Real'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Correct: {'✓' if is_deepfake else '✗'}")
    
    # Show Singularity Mode information if available
    if 'singularity_mode' in singularity_fake_result:
        sm_info = singularity_fake_result['singularity_mode']
        print("\nSingularity Mode information:")
        for key, value in sm_info.items():
            print(f"  {key}: {value}")
            
    # Display enhanced heatmap if available
    if 'heatmap' in singularity_fake_result:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(fake_image_path))
        plt.title("Fake Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(singularity_fake_result['heatmap'], cmap='hot')
        plt.title(f"Enhanced Heatmap (Confidence: {confidence:.4f})")
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"Error testing Visual Sentinel: {str(e)}")

# Calculate model accuracy
print("\n=== Model Accuracy Summary ===")
vit_accuracy = (int(vit_real_correct) + int(vit_fake_correct)) / 2 * 100
beit_accuracy = (int(beit_real_correct) + int(beit_fake_correct)) / 2 * 100
deit_accuracy = (int(deit_real_correct) + int(deit_fake_correct)) / 2 * 100
swin_accuracy = (int(swin_real_correct) + int(swin_fake_correct)) / 2 * 100

# Calculate ensemble accuracy
ensemble_real_correct = not ensemble_real_result.get('is_deepfake', True)
ensemble_fake_correct = ensemble_fake_result.get('is_deepfake', False)
ensemble_accuracy = (int(ensemble_real_correct) + int(ensemble_fake_correct)) / 2 * 100

print("Accuracy summary:")
print(f"  ViT: {vit_accuracy:.2f}%")
print(f"  BEIT: {beit_accuracy:.2f}%")
print(f"  DeiT: {deit_accuracy:.2f}%")
print(f"  Swin: {swin_accuracy:.2f}%")
print(f"  Ensemble: {ensemble_accuracy:.2f}%")

# Print CUDA memory usage
if torch.cuda.is_available():
    print(f"\nCUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"CUDA memory reserved: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

print("\nTest complete! Images processed successfully.")

# Clean up temporary files
import os
temp_files = [
    'notebook_ensemble_patch.py',
    'fix_and_run_notebook.py',
    'image_model_test_fixed.ipynb'
]

print("\nRemoving temporary files...")
for file in temp_files:
    file_path = os.path.join(project_root, 'tests', file)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"  Removed {file}")
        except Exception as e:
            print(f"  Failed to remove {file}: {str(e)}")
