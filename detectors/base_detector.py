"""
Base detector module defining the abstract interface for all deepfake detectors.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseDetector(ABC):
    """
    Abstract base class for all deepfake detectors.
    All media-specific detectors (image, audio, video) must implement this interface.
    """
    
    def __init__(self, model_name: str, confidence_threshold: float = 0.5):
        """
        Initialize the base detector with model name and confidence threshold.
        
        Args:
            model_name: Name or path of the model to use for detection
            confidence_threshold: Threshold value for classifying media as deepfake
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.name = self.__class__.__name__  # Add name attribute for ensemble detector
        
    @abstractmethod
    def detect(self, media_path: str) -> Dict[str, Any]:
        """
        Detect if the given media is a deepfake.
        
        Args:
            media_path: Path to the media file to analyze
            
        Returns:
            Dictionary containing detection results, including:
            - is_deepfake: Boolean indicating if the media is detected as deepfake
            - confidence: Float value between 0 and 1 indicating detection confidence
            - details: Additional detection details specific to the media type
            
        Raises:
            FileNotFoundError: If the media file does not exist
            ValueError: If the media file is not valid or supported
        """
        pass
    
    def normalize_confidence(self, raw_score: float) -> float:
        """
        Normalize the raw confidence score to a value between 0 and 1.
        
        Args:
            raw_score: Raw score from the model
            
        Returns:
            Normalized confidence score between 0 and 1
        """
        # Simple min-max normalization, can be overridden by subclasses
        # for more sophisticated normalization
        return max(0.0, min(1.0, raw_score))
    
    def format_result(self, is_deepfake: bool, confidence: float, 
                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the detection result into a standardized dictionary.
        
        Args:
            is_deepfake: Boolean indicating if the media is detected as deepfake
            confidence: Detection confidence between 0 and 1
            metadata: Additional metadata about the detection
            
        Returns:
            Dictionary containing formatted detection results
        """
        return {
            "is_deepfake": is_deepfake,
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "model": self.model_name,
            "timestamp": metadata.get("timestamp"),
            "media_type": metadata.get("media_type"),
            "details": metadata.get("details", {}),
            "analysis_time": metadata.get("analysis_time")
        }
    
    def _validate_media(self, media_path: str) -> bool:
        """
        Validate if the media file exists and is readable.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            True if the media is valid, False otherwise
            
        Raises:
            FileNotFoundError: If the media file does not exist
        """
        import os
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        return True
    
    def predict(self, media_path: str) -> Dict[str, Any]:
        """
        Alias for detect method to maintain compatibility with ensemble detector.
        
        Args:
            media_path: Path to the media file to analyze
            
        Returns:
            Dictionary containing detection results
        """
        return self.detect(media_path)
    
    def get_confidence(self, result: Dict[str, Any]) -> float:
        """
        Extract confidence score from detection result.
        
        Args:
            result: Detection result dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        if isinstance(result, dict) and 'confidence' in result:
            return result['confidence']
        return 0.0
