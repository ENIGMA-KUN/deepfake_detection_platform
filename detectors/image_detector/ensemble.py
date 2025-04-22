"""
Image Ensemble Detector module for combining multiple image-based deepfake detection models.

This module implements a specialized ensemble detector for images that combines
results from multiple models (ViT, BEIT, DeiT, Swin) with adaptive weighting,
and optionally applies the Visual Sentinel Singularity Mode for enhanced detection.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
import sys
import os

# Use absolute import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from detectors.ensemble_detector import EnsembleDetector

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
            # If image is a path, we should load it to provide to Singularity Mode
            # In a real implementation, we would load the image here
            result['image_data'] = image
        else:
            # If image is already loaded, include it directly
            result['image_data'] = image
        
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
    
    # Alias to keep interface consistent with BaseDetector
    detect = predict
    
    def _enhance_with_singularity(self, image, result):
        """
        Apply Visual Sentinel Singularity Mode enhancement to the result.
        
        Args:
            image: The image being analyzed
            result: The standard ensemble result
            
        Returns:
            Enhanced detection result
        """
        try:
            # Import here to avoid circular import issues
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from app.core.singularity_manager import SingularityManager
            
            # Create manager and apply Visual Sentinel
            manager = SingularityManager()
            enhanced_result = manager.apply_mode('image', 'Visual Sentinel', result)
            
            # Add enhancement metadata
            enhanced_result['enhancement'] = {
                'method': 'Visual Sentinel',
                'version': '1.0',
                'enabled': True
            }
            
            return enhanced_result
            
        except ImportError:
            self.logger.warning("SingularityManager not available, using standard ensemble result")
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
