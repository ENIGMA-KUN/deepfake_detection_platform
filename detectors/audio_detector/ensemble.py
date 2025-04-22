"""
Audio Ensemble Detector module for combining multiple audio-based deepfake detection models.

This module implements a specialized ensemble detector for audio that combines
results from multiple models (Wav2Vec2, XLSR+SLS, XLSR-Mamba, TCN-Add) with adaptive weighting,
and optionally applies the Acoustic Guardian Singularity Mode for enhanced detection.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

from detectors.ensemble_detector import EnsembleDetector

class AudioEnsembleDetector(EnsembleDetector):
    """
    Specialized ensemble detector for audio deepfakes.
    
    Combines multiple audio deepfake detection models with content-aware
    adaptive weighting and supports Acoustic Guardian Singularity Mode integration.
    """
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None, 
                 threshold: float = 0.5, enable_singularity: bool = False):
        """
        Initialize the audio ensemble detector.
        
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
                if 'wav2vec2' in model_name:
                    self.detector_types[detector] = 'wav2vec2'
                elif 'xlsr' in model_name and 'sls' in model_name:
                    self.detector_types[detector] = 'xlsr_sls'
                elif 'xlsr' in model_name and 'mamba' in model_name:
                    self.detector_types[detector] = 'xlsr_mamba'
                elif 'tcn' in model_name:
                    self.detector_types[detector] = 'tcn_add'
                else:
                    self.detector_types[detector] = 'other'
    
    def predict(self, audio):
        """
        Predict whether an audio is authentic or deepfake using the ensemble.
        
        Args:
            audio: The audio to analyze (path or array)
            
        Returns:
            Dict containing prediction results
        """
        # Get base ensemble prediction
        result = super().predict(audio)
        
        # Include the audio data for potential Singularity Mode enhancement
        if isinstance(audio, str):
            # If audio is a path, we should load it to provide to Singularity Mode
            # In a real implementation, we would load the audio here
            result['audio_data'] = audio
        else:
            # If audio is already loaded, include it directly
            result['audio_data'] = audio
        
        # Apply Acoustic Guardian enhancement if enabled
        if self.enable_singularity:
            try:
                enhanced_result = self._enhance_with_singularity(audio, result)
                return enhanced_result
            except Exception as e:
                self.logger.warning(f"Error applying Singularity Mode: {str(e)}")
                # Fall back to standard ensemble result
                return result
        
        return result
    
    def _enhance_with_singularity(self, audio, result):
        """
        Apply Acoustic Guardian Singularity Mode enhancement to the result.
        
        Args:
            audio: The audio being analyzed
            result: The standard ensemble result
            
        Returns:
            Enhanced detection result
        """
        try:
            # Import here to avoid circular import issues
            from app.core.singularity_manager import SingularityManager
            
            # Create manager and apply Acoustic Guardian
            manager = SingularityManager()
            enhanced_result = manager.apply_mode('audio', 'Acoustic Guardian', result)
            
            # Add enhancement metadata
            enhanced_result['enhancement'] = {
                'method': 'Acoustic Guardian',
                'version': '1.0',
                'enabled': True
            }
            
            return enhanced_result
            
        except ImportError:
            self.logger.warning("SingularityManager not available, using standard ensemble result")
            return result
    
    def _get_visualization_data(self, predictions):
        """
        Generate visualization data for audio predictions.
        
        Args:
            predictions: List of prediction results from individual detectors
            
        Returns:
            Dict with visualization data
        """
        # Extract spectrograms from individual predictions if available
        spectrograms = {}
        anomaly_regions = []
        
        for i, (detector, pred) in enumerate(zip(self.detectors, predictions)):
            if hasattr(detector, 'name'):
                detector_name = detector.name
            else:
                detector_name = f"Detector_{i}"
                
            # Extract spectrogram if available
            if isinstance(pred, dict):
                if 'spectrogram' in pred:
                    spectrograms[detector_name] = pred['spectrogram']
                if 'anomaly_regions' in pred:
                    for region in pred['anomaly_regions']:
                        region['model'] = detector_name
                        anomaly_regions.append(region)
        
        visualization_data = {}
        
        # Combine spectrograms with weights if available
        if spectrograms:
            # Get first spectrogram dimensions
            first_spec = next(iter(spectrograms.values()))
            combined_spec = np.zeros_like(first_spec)
            
            for det_name, spec in spectrograms.items():
                # Find detector index to get its weight
                for i, det in enumerate(self.detectors):
                    if hasattr(det, 'name') and det.name == det_name:
                        weight = self.weights[i]
                        break
                else:
                    weight = 1.0 / len(spectrograms)
                
                combined_spec += spec * weight
                
            # Normalize combined spectrogram
            if np.max(combined_spec) > 0:
                combined_spec /= np.max(combined_spec)
                
            visualization_data['spectrogram'] = combined_spec
            visualization_data['individual_spectrograms'] = spectrograms
        
        # Process anomaly regions (merge overlapping regions)
        if anomaly_regions:
            # Sort by start time
            anomaly_regions.sort(key=lambda x: x['start_time'])
            
            merged_regions = []
            if anomaly_regions:
                current = anomaly_regions[0].copy()
                
                for region in anomaly_regions[1:]:
                    if region['start_time'] <= current['end_time']:
                        # Extend current region if needed
                        current['end_time'] = max(current['end_time'], region['end_time'])
                        # Combine confidence using maximum
                        current['confidence'] = max(current['confidence'], region['confidence'])
                        # Keep track of contributing models
                        if 'models' not in current:
                            current['models'] = [current.pop('model')]
                        if region['model'] not in current['models']:
                            current['models'].append(region['model'])
                    else:
                        # No overlap, add current to results and start new one
                        if 'model' in current and 'models' not in current:
                            current['models'] = [current.pop('model')]
                        merged_regions.append(current)
                        current = region.copy()
                
                # Add the last region
                if 'model' in current and 'models' not in current:
                    current['models'] = [current.pop('model')]
                merged_regions.append(current)
                
            visualization_data['anomaly_regions'] = merged_regions
        
        return visualization_data
