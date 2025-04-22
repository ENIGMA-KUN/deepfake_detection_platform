"""
BEIT-based deepfake detector for images.
"""
import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import torch.nn.functional as F
from transformers import (
    AutoModelForImageClassification,
    AutoFeatureExtractor
)

from detectors.base_detector import BaseDetector

class BEITImageDetector(BaseDetector):
    """
    BEiT detector for image deepfakes.
    Leverages the Microsoft BEiT model for image analysis.
    """
    
    def __init__(self, model_name: str = "microsoft/beit-base-patch16-224-pt22k-ft22k",
                 confidence_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize the BEiT image detector.
        
        Args:
            model_name: BEiT model to use
            confidence_threshold: Threshold for classification
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__(model_name, confidence_threshold)
        
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models as None (lazy loading)
        self.processor = None
        self.model = None
        
        # CUDA specific optimizations
        self.use_mixed_precision = torch.cuda.is_available() and 'cuda' in self.device
        self.use_compiled_model = hasattr(torch, 'compile') and torch.cuda.is_available()
        
    def load_model(self):
        """
        Load the BEiT model for image analysis.
        """
        if self.model is not None:
            return
            
        try:
            self.logger.info(f"Loading BEiT model: {self.model_name}")
            
            # Load BEiT model for image classification
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move model to the appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            # Apply model compilation for PyTorch 2.0+
            if self.use_compiled_model:
                try:
                    self.logger.info("Compiling model with torch.compile() for optimization")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    self.logger.warning(f"Could not compile model: {str(e)}")
            
            self.logger.info("BEiT model loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading BEiT model: {str(e)}")
            raise RuntimeError(f"Failed to load BEiT model: {str(e)}")
    
    def detect(self, media_path: str) -> Dict[str, Any]:
        """
        Detect if the image is a deepfake.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the file is not a valid image
        """
        # Validate the image file
        self._validate_media(media_path)
        
        # Load the model if not already loaded
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Load and preprocess the image
            image = Image.open(media_path).convert('RGB')
            
            # Process the image with the model
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                # Get logits and convert to probabilities
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
                # Get the deepfake probability (assuming index 1 is "fake")
                deepfake_score = probs[0, 1].item()
            
            # Generate attention heatmap
            heatmap = self._generate_heatmap(image)
            
            # Prepare metadata
            metadata = {
                "timestamp": time.time(),
                "media_type": "image",
                "analysis_time": time.time() - start_time,
                "heatmap": heatmap,
                "model_name": self.model_name
            }
            
            # Determine if the image is a deepfake based on confidence threshold
            is_deepfake = deepfake_score >= self.confidence_threshold
            
            # Format and return results
            return self.format_result(is_deepfake, deepfake_score, metadata)
        
        except Exception as e:
            self.logger.error(f"Error detecting deepfake in {media_path}: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def _generate_heatmap(self, image: Image.Image) -> np.ndarray:
        """
        Generate attention heatmap for visualization.
        In a real implementation, we would extract attention scores from the model.
        This is a simplified version that just returns a placeholder.
        
        Args:
            image: PIL Image to generate heatmap for
            
        Returns:
            Attention heatmap as numpy array
        """
        # Get image size
        width, height = image.size
        
        # In a real implementation, we would extract attention maps from the model
        # For this example, we're just creating a placeholder heatmap
        heatmap = np.zeros((height, width))
        
        # Create a gradient pattern (just for visualization)
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                # Distance from center (normalized)
                dist = np.sqrt(((x - center_x) / width) ** 2 + ((y - center_y) / height) ** 2)
                # Inverse distance (higher in center)
                heatmap[y, x] = max(0, 1 - dist * 2)
        
        return heatmap
    
    def normalize_confidence(self, raw_score: float) -> float:
        """
        Normalize the raw confidence score.
        
        Args:
            raw_score: Raw score from the model
            
        Returns:
            Normalized confidence score
        """
        # For this detector, the raw score is already in [0,1]
        return raw_score
