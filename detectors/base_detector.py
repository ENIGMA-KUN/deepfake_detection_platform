#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base detector module for the Deepfake Detection Platform.
Defines abstract classes and interfaces for all detector implementations.
"""

import abc
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logger
logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Enum for supported media types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class DetectionStatus(Enum):
    """Enum for detection status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Region:
    """Data class for representing a region of interest in media."""
    x: float  # Normalized x-coordinate (0.0 to 1.0)
    y: float  # Normalized y-coordinate (0.0 to 1.0)
    width: float  # Normalized width (0.0 to 1.0)
    height: float  # Normalized height (0.0 to 1.0)
    confidence: float  # Confidence score (0.0 to 1.0)
    label: str  # Label for the region


@dataclass
class DetectionResult:
    """Data class for storing detection results."""
    id: str  # Unique identifier for the detection
    media_type: MediaType  # Type of media analyzed
    filename: str  # Original filename
    timestamp: datetime  # When the detection was performed
    is_deepfake: bool  # True if deepfake detected, False otherwise
    confidence_score: float  # Overall confidence score (0.0 to 1.0)
    regions: List[Region]  # List of regions of interest
    metadata: Dict[str, Any]  # Additional metadata
    execution_time: float  # Time taken for detection in seconds
    model_used: str  # Name/version of the model used
    status: DetectionStatus  # Current status of the detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert the detection result to a dictionary."""
        return {
            "id": self.id,
            "media_type": self.media_type.value,
            "filename": self.filename,
            "timestamp": self.timestamp.isoformat(),
            "is_deepfake": self.is_deepfake,
            "confidence_score": self.confidence_score,
            "regions": [
                {
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "confidence": r.confidence,
                    "label": r.label
                }
                for r in self.regions
            ],
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "model_used": self.model_used,
            "status": self.status.value
        }


class BaseDetector(abc.ABC):
    """Abstract base class for all deepfake detectors."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the detector.
        
        Args:
            config: Configuration dictionary
            model_path: Path to the model file or directory (optional)
        """
        self.config = config
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        # Initialize the detector
        self._initialize()
        
    @abc.abstractmethod
    def _initialize(self) -> None:
        """Initialize the detector model and resources.
        This method should be implemented by subclasses.
        """
        pass
    
    @abc.abstractmethod
    def detect(self, media_path: str) -> DetectionResult:
        """Run deepfake detection on the provided media.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            DetectionResult object with detection results
        """
        pass
    
    @abc.abstractmethod
    def preprocess(self, media_path: str) -> Any:
        """Preprocess the media file for detection.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            Preprocessed media in the format required by the model
        """
        pass
    
    @abc.abstractmethod
    def postprocess(self, model_output: Any) -> Tuple[bool, float, List[Region]]:
        """Postprocess the model output to get detection results.
        
        Args:
            model_output: Raw output from the model
            
        Returns:
            Tuple of (is_deepfake, confidence_score, regions)
        """
        pass
        
    def normalize_confidence(self, score: float) -> float:
        """Normalize raw confidence score to 0.0-1.0 range.
        
        Args:
            score: Raw confidence score
            
        Returns:
            Normalized confidence score between 0.0 and 1.0
        """
        # Simple min-max normalization, can be overridden by subclasses
        return max(0.0, min(float(score), 1.0))
    
    def is_deepfake(self, confidence: float) -> bool:
        """Determine if the media is a deepfake based on confidence score.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            True if the media is classified as deepfake, False otherwise
        """
        return confidence >= self.confidence_threshold
    
    def validate_media(self, media_path: str) -> bool:
        """Validate if the media file exists and is of the correct type.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            True if media is valid, False otherwise
        """
        if not os.path.exists(media_path):
            logger.error(f"Media file does not exist: {media_path}")
            return False
        
        # Additional validation can be implemented by subclasses
        return True
    
    def get_media_type(self) -> MediaType:
        """Get the media type supported by this detector.
        
        Returns:
            MediaType enum value
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_media_type()")


class DetectorFactory:
    """Factory class for creating appropriate detector instances."""
    
    @staticmethod
    def create_detector(media_type: MediaType, config: Dict[str, Any],
                       model_path: Optional[str] = None) -> BaseDetector:
        """Create an appropriate detector instance based on media type.
        
        Args:
            media_type: Type of media to detect
            config: Configuration dictionary
            model_path: Path to the model file or directory (optional)
            
        Returns:
            Instance of appropriate detector class
            
        Raises:
            ValueError: If media type is not supported
        """
        # This method will be updated as detector implementations are added
        if media_type == MediaType.IMAGE:
            # Import here to avoid circular imports
            from detectors.image_detector.vit_detector import VitDetector
            return VitDetector(config, model_path)
        elif media_type == MediaType.AUDIO:
            # Import here to avoid circular imports
            from detectors.audio_detector.wav2vec_detector import Wav2VecDetector
            return Wav2VecDetector(config, model_path)
        elif media_type == MediaType.VIDEO:
            # Import here to avoid circular imports
            from detectors.video_detector.genconvit import GenConVitDetector
            return GenConVitDetector(config, model_path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")