"""
Video Ensemble Detector module for combining multiple video-based deepfake detection models.

This module implements a specialized ensemble detector for videos that combines
results from multiple models (GenConViT, TimeSformer, SlowFast, Video Swin, X3D) 
with adaptive weighting, and optionally applies the Temporal Oracle Singularity Mode 
for enhanced detection.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

from detectors.ensemble_detector import EnsembleDetector

class VideoEnsembleDetector(EnsembleDetector):
    """
    Specialized ensemble detector for video deepfakes.
    
    Combines multiple video deepfake detection models with content-aware
    adaptive weighting and supports Temporal Oracle Singularity Mode integration.
    """
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None, 
                 threshold: float = 0.5, enable_singularity: bool = False):
        """
        Initialize the video ensemble detector.
        
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
                if 'genconvit' in model_name:
                    self.detector_types[detector] = 'genconvit'
                elif 'timesformer' in model_name:
                    self.detector_types[detector] = 'timesformer'
                elif 'slowfast' in model_name:
                    self.detector_types[detector] = 'slowfast'
                elif 'swin' in model_name:
                    self.detector_types[detector] = 'video_swin'
                elif 'x3d' in model_name:
                    self.detector_types[detector] = 'x3d'
                else:
                    self.detector_types[detector] = 'other'
    
    def predict(self, video):
        """
        Predict whether a video is authentic or deepfake using the ensemble.
        
        Args:
            video: The video to analyze (path or array)
            
        Returns:
            Dict containing prediction results
        """
        # Get base ensemble prediction
        result = super().predict(video)
        
        # Include the video data for potential Singularity Mode enhancement
        if isinstance(video, str):
            # If video is a path, we should load it to provide to Singularity Mode
            # In a real implementation, we would load the video here
            result['video_data'] = video
        else:
            # If video is already loaded, include it directly
            result['video_data'] = video
        
        # Apply Temporal Oracle enhancement if enabled
        if self.enable_singularity:
            try:
                enhanced_result = self._enhance_with_singularity(video, result)
                return enhanced_result
            except Exception as e:
                self.logger.warning(f"Error applying Singularity Mode: {str(e)}")
                # Fall back to standard ensemble result
                return result
        
        return result
    
    # Alias to keep interface consistent with BaseDetector
    detect = predict
    
    def _enhance_with_singularity(self, video, result):
        """
        Apply Temporal Oracle Singularity Mode enhancement to the result.
        
        Args:
            video: The video being analyzed
            result: The standard ensemble result
            
        Returns:
            Enhanced detection result
        """
        try:
            # Import here to avoid circular import issues
            from app.core.singularity_manager import SingularityManager
            
            # Create manager and apply Temporal Oracle
            manager = SingularityManager()
            enhanced_result = manager.apply_mode('video', 'Temporal Oracle', result)
            
            # Add enhancement metadata
            enhanced_result['enhancement'] = {
                'method': 'Temporal Oracle',
                'version': '1.0',
                'enabled': True
            }
            
            return enhanced_result
            
        except ImportError:
            self.logger.warning("SingularityManager not available, using standard ensemble result")
            return result
    
    def _get_visualization_data(self, predictions):
        """
        Generate visualization data for video predictions.
        
        Args:
            predictions: List of prediction results from individual detectors
            
        Returns:
            Dict with visualization data
        """
        # Extract frame-level analysis, temporal analysis, and A/V sync data
        frame_analyses = {}
        temporal_analyses = {}
        av_sync_analyses = {}
        manipulation_timelines = {}
        
        for i, (detector, pred) in enumerate(zip(self.detectors, predictions)):
            if hasattr(detector, 'name'):
                detector_name = detector.name
            else:
                detector_name = f"Detector_{i}"
                
            # Extract visualization data if available
            if isinstance(pred, dict):
                if 'frame_analysis' in pred:
                    frame_analyses[detector_name] = pred['frame_analysis']
                if 'temporal_analysis' in pred:
                    temporal_analyses[detector_name] = pred['temporal_analysis']
                if 'av_sync_analysis' in pred:
                    av_sync_analyses[detector_name] = pred['av_sync_analysis']
                if 'manipulation_timeline' in pred:
                    manipulation_timelines[detector_name] = pred['manipulation_timeline']
        
        visualization_data = {}
        
        # Process frame analyses
        if frame_analyses:
            # In a real implementation, these would be combined more intelligently
            visualization_data['frame_analyses'] = frame_analyses
            
            # Create a combined heatmap for key frames if available
            combined_keyframes = {}
            for detector_name, frames in frame_analyses.items():
                for frame_idx, frame_data in frames.items():
                    if 'heatmap' in frame_data:
                        if frame_idx not in combined_keyframes:
                            combined_keyframes[frame_idx] = {
                                'heatmaps': {},
                                'combined_heatmap': None
                            }
                        combined_keyframes[frame_idx]['heatmaps'][detector_name] = frame_data['heatmap']
            
            # Combine heatmaps for each keyframe
            for frame_idx, frame_data in combined_keyframes.items():
                if frame_data['heatmaps']:
                    # Get first heatmap dimensions
                    first_heatmap = next(iter(frame_data['heatmaps'].values()))
                    combined_heatmap = np.zeros_like(first_heatmap)
                    
                    for det_name, heatmap in frame_data['heatmaps'].items():
                        # Find detector index to get its weight
                        for i, det in enumerate(self.detectors):
                            if hasattr(det, 'name') and det.name == det_name:
                                weight = self.weights[i]
                                break
                        else:
                            weight = 1.0 / len(frame_data['heatmaps'])
                        
                        if heatmap.shape == combined_heatmap.shape:
                            combined_heatmap += heatmap * weight
                    
                    # Normalize combined heatmap
                    if np.max(combined_heatmap) > 0:
                        combined_heatmap /= np.max(combined_heatmap)
                        
                    frame_data['combined_heatmap'] = combined_heatmap
            
            visualization_data['keyframes'] = combined_keyframes
        
        # Process temporal analyses
        if temporal_analyses:
            # Combine temporal consistency scores over time
            all_timestamps = set()
            for analysis in temporal_analyses.values():
                if 'timestamps' in analysis:
                    all_timestamps.update(analysis['timestamps'])
            
            if all_timestamps:
                timestamps = sorted(all_timestamps)
                combined_scores = np.zeros(len(timestamps))
                
                for detector_name, analysis in temporal_analyses.items():
                    if 'timestamps' in analysis and 'scores' in analysis:
                        # Find detector index to get its weight
                        for i, det in enumerate(self.detectors):
                            if hasattr(det, 'name') and det.name == detector_name:
                                weight = self.weights[i]
                                break
                        else:
                            weight = 1.0 / len(temporal_analyses)
                        
                        # Interpolate scores to match combined timestamps
                        det_timestamps = analysis['timestamps']
                        det_scores = analysis['scores']
                        
                        for i, ts in enumerate(timestamps):
                            if ts in det_timestamps:
                                idx = det_timestamps.index(ts)
                                combined_scores[i] += det_scores[idx] * weight
                
                visualization_data['temporal_analysis'] = {
                    'timestamps': timestamps,
                    'scores': combined_scores.tolist()
                }
        
        # Process A/V sync analyses
        if av_sync_analyses:
            # In a real implementation, would combine these more intelligently
            visualization_data['av_sync_analyses'] = av_sync_analyses
        
        # Process manipulation timelines
        if manipulation_timelines:
            # Merge manipulation timelines
            combined_timeline = {}
            
            for detector_name, timeline in manipulation_timelines.items():
                for segment_id, segment in timeline.items():
                    if segment_id not in combined_timeline:
                        combined_timeline[segment_id] = segment.copy()
                        combined_timeline[segment_id]['detectors'] = [detector_name]
                    else:
                        # Update existing segment
                        existing = combined_timeline[segment_id]
                        existing['confidence'] = max(existing['confidence'], segment['confidence'])
                        existing['detectors'].append(detector_name)
            
            visualization_data['manipulation_timeline'] = combined_timeline
        
        return visualization_data
