#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vision Transformer (ViT) based deepfake detector for images.
Utilizes the google/vit-base-patch16-224 pre-trained model for detection.
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region)
from detectors.detector_utils import (identify_deepfake_category,
                                     load_image, measure_execution_time,
                                     normalize_confidence_score,
                                     normalize_image, resize_image)

# Configure logger
logger = logging.getLogger(__name__)


class VitDetector(BaseDetector):
    """Vision Transformer (ViT) based deepfake detector for images."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the Vision Transformer detector.
        
        Args:
            config: Configuration dictionary
            model_path: Path to pre-trained model (optional)
        """
        self.model_name = config.get("model_name", "google/vit-base-patch16-224")
        self.device = "cuda" if config.get("use_gpu", True) else "cpu"
        
        # Initialize preprocessing parameters
        self.image_size = (
            config.get("preprocessing", {}).get("image", {}).get("resize_width", 224),
            config.get("preprocessing", {}).get("image", {}).get("resize_height", 224)
        )
        self.normalize = config.get("preprocessing", {}).get("image", {}).get("normalize", True)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the ViT model and resources."""
        try:
            # Import transformers here to avoid dependency if not needed
            from transformers import ViTForImageClassification, ViTImageProcessor
            
            logger.info(f"Loading ViT model: {self.model_name}")
            
            # Load image processor
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            
            # Load pre-trained model
            self.model = ViTForImageClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real vs deepfake
                ignore_mismatched_sizes=True  # Handle potential mismatch in final layer
            )
            
            # Move model to appropriate device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                    logger.info("Model loaded on GPU")
                else:
                    logger.warning("CUDA requested but not available, using CPU instead")
                    self.device = "cpu"
            
            logger.info("ViT detector initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("The 'transformers' library is required for ViT detector")
            
        except Exception as e:
            logger.error(f"Failed to initialize ViT detector: {e}")
            raise RuntimeError(f"Error initializing ViT detector: {e}")
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run deepfake detection on the provided image.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.IMAGE, os.path.basename(media_path))
        result.model_used = self.model_name
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid image file"
                return result
            
            # Preprocess image
            preprocessed = self.preprocess(media_path)
            
            # Run model inference
            import torch
            with torch.no_grad():
                outputs = self.model(**preprocessed)
            
            # Postprocess results
            is_deepfake, confidence, regions = self.postprocess(outputs)
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.regions = regions
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["image_size"] = f"{preprocessed['pixel_values'].shape[-2]}x{preprocessed['pixel_values'].shape[-1]}"
            
            # Identify deepfake categories
            result.categories = identify_deepfake_category(
                MediaType.IMAGE, confidence, regions
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> Dict[str, Any]:
        """Preprocess the image for ViT model.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            Dictionary with preprocessed inputs for the model
        """
        # Load image
        image = load_image(media_path)
        
        # Resize if needed
        if image.shape[0] != self.image_size[1] or image.shape[1] != self.image_size[0]:
            image = resize_image(image, self.image_size)
        
        # Preprocess using ViT processor
        inputs = self.processor(
            images=image,
            return_tensors="pt"  # PyTorch tensors
        )
        
        # Move inputs to device
        if self.device == "cuda":
            import torch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def postprocess(self, model_output: Any) -> Tuple[bool, float, List[Region]]:
        """Postprocess the model output to get detection results.
        
        Args:
            model_output: Raw output from the ViT model
            
        Returns:
            Tuple of (is_deepfake, confidence_score, regions)
        """
        import torch
        
        # Get logits from model output
        logits = model_output.logits
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get probabilities as numpy array
        probs_np = probs.cpu().numpy()[0]
        
        # Assuming index 1 corresponds to "deepfake" class
        deepfake_confidence = float(probs_np[1])
        
        # Normalize confidence
        confidence = normalize_confidence_score(deepfake_confidence)
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        # For ViT, we don't have specific region information from the base model
        # We'll create a full-image region for simplicity
        regions = []
        if is_deepfake:
            regions.append(Region(
                x=0.0,
                y=0.0,
                width=1.0,
                height=1.0,
                confidence=confidence,
                label="potential_deepfake",
                category=DeepfakeCategory.GAN_GENERATED
            ))
        
        return is_deepfake, confidence, regions
    
    def get_attention_map(self, model_output: Any) -> np.ndarray:
        """Extract attention map from model output for visualization.
        
        Args:
            model_output: Raw output from the ViT model with attention weights
            
        Returns:
            Attention map as numpy array
        """
        # This is a placeholder - full implementation would extract attention
        # weights from the ViT model for visualization
        
        # For now, return a dummy heatmap
        return np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
    
    def validate_media(self, media_path: str) -> bool:
        """Validate if the media file is a supported image.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            True if media is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(media_path):
            logger.error(f"File does not exist: {media_path}")
            return False
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        file_ext = os.path.splitext(media_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Unsupported image format: {file_ext}")
            return False
        
        # Basic file size check
        if os.path.getsize(media_path) == 0:
            logger.error(f"Empty file: {media_path}")
            return False
        
        # Try opening the image to ensure it's valid
        try:
            test_image = load_image(media_path)
            # Check image dimensions
            if len(test_image.shape) < 2 or len(test_image.shape) > 3:
                logger.error(f"Invalid image dimensions: {test_image.shape}")
                return False
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return False
        
        return True
    
    def get_media_type(self) -> MediaType:
        """Get the media type supported by this detector.
        
        Returns:
            MediaType.IMAGE
        """
        return MediaType.IMAGE