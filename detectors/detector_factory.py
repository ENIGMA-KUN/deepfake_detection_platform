#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Factory module for creating detector instances.
"""

import logging
import os
from typing import Any, Dict, Optional, Type

from detectors.base_detector import BaseDetector, MediaType

# Configure logger
logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Registry for detector classes."""
    _registry = {}
    
    @classmethod
    def register(cls, media_type: MediaType):
        """Decorator to register a detector class for a specific media type.
        
        Args:
            media_type: The media type this detector handles
            
        Returns:
            Decorator function
        """
        def decorator(detector_class):
            cls._registry[media_type] = detector_class
            return detector_class
        return decorator
    
    @classmethod
    def get_detector_class(cls, media_type: MediaType) -> Optional[Type[BaseDetector]]:
        """Get the detector class for a specific media type.
        
        Args:
            media_type: The media type
            
        Returns:
            Detector class or None if not registered
        """
        return cls._registry.get(media_type)


class DetectorFactory:
    """Factory for creating detector instances."""
    
    @staticmethod
    def create_detector(
        media_type: MediaType,
        config: Dict[str, Any],
        model_path: Optional[str] = None
    ) -> BaseDetector:
        """Create a detector instance for the specified media type.
        
        Args:
            media_type: Type of media to detect
            config: Configuration dictionary
            model_path: Path to the model file or directory (optional)
            
        Returns:
            Detector instance
            
        Raises:
            ValueError: If media type is not supported
        """
        detector_class = DetectorRegistry.get_detector_class(media_type)
        
        if detector_class is None:
            raise ValueError(f"No detector registered for media type: {media_type.value}")
        
        logger.info(f"Creating detector for media type: {media_type.value}")
        
        # If model_path is not provided, try to find it in the config
        if model_path is None and "model_path" in config:
            model_path = config["model_path"]
        
        # Validate model path if provided
        if model_path is not None and not os.path.exists(model_path):
            logger.warning(f"Model path does not exist: {model_path}")
        
        # Create and return detector instance
        return detector_class(config, model_path)


class EnsembleDetectorFactory:
    """Factory for creating ensemble detector instances."""
    
    @staticmethod
    def create_ensemble_detector(
        media_type: MediaType,
        config: Dict[str, Any],
        model_paths: Optional[Dict[str, str]] = None
    ) -> BaseDetector:
        """Create an ensemble detector for the specified media type.
        
        Args:
            media_type: Type of media to detect
            config: Configuration dictionary
            model_paths: Dictionary mapping detector names to model paths
            
        Returns:
            Ensemble detector instance
            
        Raises:
            ValueError: If media type is not supported for ensemble detection
        """
        # Import here to prevent circular imports
        if media_type == MediaType.IMAGE:
            from detectors.image_detector.ensemble import ImageEnsembleDetector
            return ImageEnsembleDetector(config, model_paths)
        elif media_type == MediaType.AUDIO:
            # This would be implemented when audio ensemble is added
            raise NotImplementedError("Audio ensemble detector not yet implemented")
        elif media_type == MediaType.VIDEO:
            # This would be implemented when video ensemble is added
            raise NotImplementedError("Video ensemble detector not yet implemented")
        else:
            raise ValueError(f"Unsupported media type for ensemble detection: {media_type.value}")


# Dynamically import and register all detector implementations
def _register_detector_implementations():
    """Register all detector implementations.
    This function is called at module initialization.
    """
    # Image detectors
    try:
        from detectors.image_detector.vit_detector import VitDetector
    except ImportError:
        logger.warning("VitDetector could not be imported")
    
    # Audio detectors
    try:
        from detectors.audio_detector.wav2vec_detector import Wav2VecDetector
    except ImportError:
        logger.warning("Wav2VecDetector could not be imported")
    
    # Video detectors
    try:
        from detectors.video_detector.genconvit import GenConVitDetector
    except ImportError:
        logger.warning("GenConVitDetector could not be imported")


# Register implementations when module is imported
_register_detector_implementations()