# app/core/singularity_modes.py
import numpy as np
from typing import List, Dict, Any, Optional

class SingularityMode:
    """
    Base class for Singularity Mode implementations.
    Singularity Modes are advanced ensemble approaches that dynamically combine
    multiple models for superior deepfake detection performance.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the detection result using the Singularity Mode's enhanced algorithms.
        To be implemented by subclasses.
        
        Args:
            result: Raw detection result from standard ensemble
            
        Returns:
            Enhanced detection result
        """
        raise NotImplementedError("Subclasses must implement process()")


class VisualSentinel(SingularityMode):
    """
    Visual Sentinel - Flagship image detection capability.
    Dynamically combines all image models with adaptive weighting
    based on content characteristics.
    """
    
    def __init__(self):
        super().__init__(
            name="Visual Sentinel",
            description="Advanced image deepfake detection using dynamic model weighting"
        )
        
        # Default content-based weights for different image characteristics
        # These will be dynamically adjusted based on content analysis
        self.content_weights = {
            "faces": {
                "vit": 0.2,
                "deit": 0.3,
                "beit": 0.2,
                "swin": 0.3
            },
            "landscapes": {
                "vit": 0.3,
                "deit": 0.2,
                "beit": 0.3,
                "swin": 0.2
            },
            "graphics": {
                "vit": 0.25,
                "deit": 0.25,
                "beit": 0.25,
                "swin": 0.25
            }
        }
    
    def analyze_content_characteristics(self, image_data: np.ndarray) -> str:
        """
        Analyze the image content to determine its primary characteristics.
        
        Args:
            image_data: Numpy array representing the image
            
        Returns:
            String identifying the content type
        """
        # In a real implementation, this would use a classifier to determine content type
        # For now, we'll use a simple placeholder that assumes most inputs are faces
        return "faces"
    
    def compute_adaptive_weights(self, image_data: np.ndarray, individual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute adaptive model weights based on image content and initial model confidences.
        
        Args:
            image_data: Numpy array representing the image
            individual_results: Results from individual detection models
            
        Returns:
            Dictionary of model names to weight values
        """
        # Determine content type
        content_type = self.analyze_content_characteristics(image_data)
        
        # Get base weights for this content type
        weights = self.content_weights[content_type].copy()
        
        # Adjust weights based on confidence separation (higher weight to more confident models)
        confidences = {result['model']: result['confidence'] for result in individual_results}
        confidence_mean = np.mean(list(confidences.values()))
        
        for model, confidence in confidences.items():
            model_key = model.split('/')[0].lower()  # Extract model family (vit, deit, etc.)
            if model_key in weights:
                # Increase weight for models with above-average confidence
                confidence_factor = (confidence / confidence_mean) if confidence_mean > 0 else 1.0
                weights[model_key] *= confidence_factor
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def generate_enhanced_heatmap(self, individual_heatmaps: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Generate an enhanced heatmap by combining individual model heatmaps with adaptive weighting.
        
        Args:
            individual_heatmaps: Dictionary of model names to heatmap arrays
            weights: Dictionary of model names to weight values
            
        Returns:
            Combined heatmap array
        """
        if not individual_heatmaps:
            return None
        
        # Get first heatmap dimensions
        first_heatmap = next(iter(individual_heatmaps.values()))
        combined_heatmap = np.zeros_like(first_heatmap)
        
        # Combine heatmaps with weights
        for model, heatmap in individual_heatmaps.items():
            model_key = model.split('/')[0].lower()
            if model_key in weights and heatmap is not None:
                # Ensure heatmap is the right shape
                if heatmap.shape != first_heatmap.shape:
                    # Resize heatmap to match first one
                    # In a real implementation, use proper resizing
                    continue
                
                combined_heatmap += heatmap * weights[model_key]
        
        # Normalize to [0, 1]
        if np.max(combined_heatmap) > 0:
            combined_heatmap = combined_heatmap / np.max(combined_heatmap)
        
        return combined_heatmap
    
    def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the detection result using the Visual Sentinel enhanced algorithms.
        
        Args:
            result: Raw detection result from standard ensemble
            
        Returns:
            Enhanced detection result with improved confidence and visualization
        """
        # Extract information from input result
        individual_results = result.get('individual_results', [])
        is_deepfake = result.get('is_deepfake', False)
        confidence = result.get('confidence', 0.5)
        image_data = result.get('image_data', None)
        metadata = result.get('metadata', {})
        
        if not individual_results or image_data is None:
            # Not enough data to enhance
            return result
        
        # Compute adaptive weights
        adaptive_weights = self.compute_adaptive_weights(image_data, individual_results)
        
        # Create enhanced version of result
        enhanced_result = result.copy()
        
        # Extract heatmaps from individual results
        individual_heatmaps = {}
        for ind_result in individual_results:
            model = ind_result['model']
            if 'raw_prediction' in ind_result and 'heatmap' in ind_result['raw_prediction']:
                individual_heatmaps[model] = ind_result['raw_prediction']['heatmap']
        
        # Generate enhanced heatmap
        enhanced_heatmap = self.generate_enhanced_heatmap(individual_heatmaps, adaptive_weights)
        if enhanced_heatmap is not None:
            enhanced_result['heatmap'] = enhanced_heatmap
        
        # Recalculate confidence using adaptive weights
        if len(individual_results) > 0:
            weighted_confidence = 0.0
            for ind_result in individual_results:
                model = ind_result['model']
                model_key = model.split('/')[0].lower()
                if model_key in adaptive_weights:
                    weighted_confidence += ind_result['confidence'] * adaptive_weights[model_key]
            
            enhanced_result['confidence'] = weighted_confidence
            enhanced_result['is_deepfake'] = weighted_confidence > 0.5
        
        # Add Singularity Mode metadata
        enhanced_result['singularity_mode'] = {
            'name': self.name,
            'description': self.description,
            'adaptive_weights': adaptive_weights,
            'confidence_improvement': weighted_confidence - confidence if 'weighted_confidence' in locals() else 0
        }
        
        return enhanced_result


class AcousticGuardian(SingularityMode):
    """
    Acoustic Guardian - Flagship audio detection capability.
    Combines all audio models with sophisticated weighted ensemble
    and confidence-based calibration.
    """
    
    def __init__(self):
        super().__init__(
            name="Acoustic Guardian",
            description="Advanced audio deepfake detection using spectral analysis and model fusion"
        )
        
        # Default weights for different audio characteristics
        self.audio_type_weights = {
            "speech": {
                "wav2vec2": 0.35,
                "xlsr_sls": 0.20,
                "xlsr_mamba": 0.30,
                "tcn_add": 0.15
            },
            "music": {
                "wav2vec2": 0.25,
                "xlsr_sls": 0.20,
                "xlsr_mamba": 0.35,
                "tcn_add": 0.20
            },
            "ambient": {
                "wav2vec2": 0.25,
                "xlsr_sls": 0.25,
                "xlsr_mamba": 0.25,
                "tcn_add": 0.25
            }
        }
        
        # Frequency band importance for different manipulation types
        self.frequency_band_weights = {
            "low": 0.3,    # 0-500 Hz
            "mid": 0.4,    # 500-2000 Hz
            "high": 0.3    # 2000+ Hz
        }
    
    def analyze_audio_type(self, audio_data: np.ndarray) -> str:
        """
        Analyze the audio content to determine its primary type.
        
        Args:
            audio_data: Numpy array representing the audio
            
        Returns:
            String identifying the audio type
        """
        # In a real implementation, this would use a classifier
        # For now, assume most inputs are speech
        return "speech"
    
    def analyze_frequency_bands(self, spectrograms: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze importance of different frequency bands in the spectrogram.
        
        Args:
            spectrograms: Dictionary of model names to spectrogram arrays
            
        Returns:
            Dictionary of frequency bands to importance values
        """
        # In a real implementation, this would analyze frequency content
        # For now, return default weights
        return self.frequency_band_weights.copy()
    
    def compute_adaptive_weights(self, audio_data: np.ndarray, individual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute adaptive model weights based on audio content and initial model confidences.
        
        Args:
            audio_data: Numpy array representing the audio
            individual_results: Results from individual detection models
            
        Returns:
            Dictionary of model names to weight values
        """
        # Determine audio type
        audio_type = self.analyze_audio_type(audio_data)
        
        # Get base weights for this audio type
        weights = self.audio_type_weights[audio_type].copy()
        
        # Adjust weights based on confidence separation
        confidences = {result['model']: result['confidence'] for result in individual_results}
        confidence_mean = np.mean(list(confidences.values()))
        
        for model, confidence in confidences.items():
            model_key = model.split('/')[0].lower().replace('-', '_')
            if model_key not in weights:
                # Try to match by partial name
                for key in weights.keys():
                    if key in model_key:
                        model_key = key
                        break
            
            if model_key in weights:
                # Increase weight for models with above-average confidence
                confidence_factor = (confidence / confidence_mean) if confidence_mean > 0 else 1.0
                weights[model_key] *= confidence_factor
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def generate_enhanced_spectrogram(self, spectrograms: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Generate an enhanced spectrogram by combining individual model spectrograms with adaptive weighting.
        
        Args:
            spectrograms: Dictionary of model names to spectrogram arrays
            weights: Dictionary of model names to weight values
            
        Returns:
            Combined spectrogram array
        """
        if not spectrograms:
            return None
        
        # Get first spectrogram dimensions
        first_spec = next(iter(spectrograms.values()))
        combined_spec = np.zeros_like(first_spec)
        
        # Combine spectrograms with weights
        for model, spec in spectrograms.items():
            model_key = model.split('/')[0].lower().replace('-', '_')
            if model_key not in weights:
                # Try to match by partial name
                for key in weights.keys():
                    if key in model_key:
                        model_key = key
                        break
            
            if model_key in weights and spec is not None:
                # Ensure spectrogram is the right shape
                if spec.shape != first_spec.shape:
                    # Resize spectrogram to match first one
                    # In a real implementation, use proper resizing
                    continue
                
                combined_spec += spec * weights[model_key]
        
        # Normalize to [0, 1]
        if np.max(combined_spec) > 0:
            combined_spec = combined_spec / np.max(combined_spec)
        
        return combined_spec
    
    def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the detection result using the Acoustic Guardian enhanced algorithms.
        
        Args:
            result: Raw detection result from standard ensemble
            
        Returns:
            Enhanced detection result with improved confidence and visualization
        """
        # Extract information from input result
        individual_results = result.get('individual_results', [])
        is_deepfake = result.get('is_deepfake', False)
        confidence = result.get('confidence', 0.5)
        audio_data = result.get('audio_data', None)
        metadata = result.get('metadata', {})
        
        if not individual_results or audio_data is None:
            # Not enough data to enhance
            return result
        
        # Compute adaptive weights
        adaptive_weights = self.compute_adaptive_weights(audio_data, individual_results)
        
        # Create enhanced version of result
        enhanced_result = result.copy()
        
        # Extract spectrograms from individual results
        spectrograms = {}
        anomaly_regions = []
        
        for ind_result in individual_results:
            model = ind_result['model']
            if 'raw_prediction' in ind_result:
                raw_pred = ind_result['raw_prediction']
                if 'spectrogram' in raw_pred:
                    spectrograms[model] = raw_pred['spectrogram']
                if 'anomaly_regions' in raw_pred:
                    for region in raw_pred['anomaly_regions']:
                        region['model'] = model
                        anomaly_regions.append(region)
        
        # Generate enhanced spectrogram
        enhanced_spectrogram = self.generate_enhanced_spectrogram(spectrograms, adaptive_weights)
        if enhanced_spectrogram is not None:
            enhanced_result['spectrogram'] = enhanced_spectrogram
        
        # Process anomaly regions (merge overlapping regions)
        if anomaly_regions:
            # Sort by start time
            anomaly_regions.sort(key=lambda x: x['start_time'])
            
            merged_regions = []
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
            
            enhanced_result['anomaly_regions'] = merged_regions
        
        # Recalculate confidence using adaptive weights
        if len(individual_results) > 0:
            weighted_confidence = 0.0
            for ind_result in individual_results:
                model = ind_result['model']
                model_key = model.split('/')[0].lower().replace('-', '_')
                if model_key not in adaptive_weights:
                    # Try to match by partial name
                    for key in adaptive_weights.keys():
                        if key in model_key:
                            model_key = key
                            break
                
                if model_key in adaptive_weights:
                    weighted_confidence += ind_result['confidence'] * adaptive_weights[model_key]
            
            enhanced_result['confidence'] = weighted_confidence
            enhanced_result['is_deepfake'] = weighted_confidence > 0.5
        
        # Add frequency band analysis
        if spectrograms:
            enhanced_result['frequency_analysis'] = self.analyze_frequency_bands(spectrograms)
        
        # Add Singularity Mode metadata
        enhanced_result['singularity_mode'] = {
            'name': self.name,
            'description': self.description,
            'adaptive_weights': adaptive_weights,
            'confidence_improvement': weighted_confidence - confidence if 'weighted_confidence' in locals() else 0
        }
        
        return enhanced_result


class TemporalOracle(SingularityMode):
    """
    Temporal Oracle - Flagship video detection capability.
    Multi-modal fusion of video, image, and audio analysis for comprehensive
    deepfake detection in videos.
    """
    
    def __init__(self):
        super().__init__(
            name="Temporal Oracle",
            description="Advanced video deepfake detection using multi-modal analysis and temporal consistency verification"
        )
        
        # Default component weights
        self.component_weights = {
            "frame_analysis": 0.25,
            "audio_analysis": 0.20,
            "temporal_analysis": 0.30,
            "av_sync_analysis": 0.15,
            "video_models": 0.10
        }
        
        # Video model weights
        self.video_model_weights = {
            "genconvit": 0.25,
            "timesformer": 0.25,
            "slowfast": 0.15,
            "video_swin": 0.25,
            "x3d": 0.10
        }
    
    def analyze_video_content(self, video_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze video content to determine optimal component weights.
        
        Args:
            video_metadata: Dictionary with video metadata
            
        Returns:
            Dictionary of component names to weight values
        """
        # In a real implementation, this would analyze video content
        # For now, return default weights with small random variations
        weights = self.component_weights.copy()
        
        # Add small variations based on video properties
        duration = video_metadata.get('duration', 0)
        if duration < 5:
            # Very short clips - emphasize frame analysis
            weights['frame_analysis'] += 0.05
            weights['temporal_analysis'] -= 0.05
        elif duration > 60:
            # Longer videos - emphasize temporal consistency
            weights['temporal_analysis'] += 0.05
            weights['frame_analysis'] -= 0.05
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def detect_manipulation_type(self, results: Dict[str, Any]) -> str:
        """
        Detect most likely manipulation type based on analysis results.
        
        Args:
            results: Dictionary with detection results
            
        Returns:
            String identifying the manipulation type
        """
        # Extract relevant information
        frame_analysis = results.get('frame_analysis', {})
        audio_analysis = results.get('audio_analysis', {})
        temporal_analysis = results.get('temporal_analysis', {})
        
        # Look for indicators of different manipulation types
        indicators = {
            "face_swap": 0,
            "face_reenactment": 0,
            "full_synthesis": 0,
            "audio_only": 0,
            "mouth_sync": 0
        }
        
        # Check face detection results
        if 'top_suspicious_frames' in frame_analysis:
            for frame in frame_analysis['top_suspicious_frames']:
                confidence = frame.get('confidence', 0)
                if confidence > 0.7:
                    indicators['face_swap'] += 1
                    indicators['face_reenactment'] += 0.5
        
        # Check temporal consistency
        if temporal_analysis:
            consistency = temporal_analysis.get('consistency_score', 1.0)
            if consistency < 0.6:
                indicators['face_reenactment'] += 2
                indicators['full_synthesis'] += 1
        
        # Check audio analysis
        if audio_analysis:
            confidence = audio_analysis.get('confidence', 0)
            if confidence > 0.7:
                indicators['audio_only'] += 2
                indicators['full_synthesis'] += 1
            
            if 'frequency_analysis' in audio_analysis:
                freq = audio_analysis['frequency_analysis']
                if 'harmonic_consistency' in freq and freq['harmonic_consistency'] < 0.6:
                    indicators['audio_only'] += 1
        
        # Check AV sync
        av_sync = results.get('av_sync_analysis', {})
        if av_sync:
            sync_score = av_sync.get('sync_score', 1.0)
            if sync_score < 0.7:
                indicators['mouth_sync'] += 2
                indicators['full_synthesis'] += 0.5
        
        # Return type with highest indicator score
        if max(indicators.values()) > 0:
            return max(indicators.items(), key=lambda x: x[1])[0]
        else:
            return "unknown"
    
    def process_temporal_analysis(self, temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance temporal analysis with additional metrics.
        
        Args:
            temporal_analysis: Original temporal analysis result
            
        Returns:
            Enhanced temporal analysis
        """
        if not temporal_analysis:
            return {}
        
        enhanced = temporal_analysis.copy()
        
        # Add temporal stability score based on confidence variance
        variance = temporal_analysis.get('confidence_variance', 0)
        stability = 1.0 / (1.0 + variance * 10)  # Transform variance to 0-1 range
        enhanced['temporal_stability'] = stability
        
        # Analyze significant jumps if available
        jumps = temporal_analysis.get('significant_jumps', [])
        if jumps:
            # Calculate average jump magnitude
            jump_magnitudes = [abs(j.get('confidence_change', 0)) for j in jumps]
            avg_magnitude = sum(jump_magnitudes) / len(jump_magnitudes) if jump_magnitudes else 0
            enhanced['avg_jump_magnitude'] = avg_magnitude
            
            # Count number of jumps per minute
            duration = temporal_analysis.get('duration', 60)  # Default 60s if not provided
            jumps_per_minute = len(jumps) / (duration / 60) if duration > 0 else 0
            enhanced['jumps_per_minute'] = jumps_per_minute
        
        return enhanced
    
    def generate_timeline(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced unified timeline from all analysis components.
        
        Args:
            results: Dictionary with detection results
            
        Returns:
            Enhanced timeline data
        """
        # Extract relevant timeline data
        frame_results = results.get('frame_analysis', {}).get('frame_results', [])
        audio_regions = results.get('audio_analysis', {}).get('anomaly_regions', [])
        temporal_jumps = results.get('temporal_analysis', {}).get('significant_jumps', [])
        av_issues = results.get('av_sync_analysis', {}).get('sync_issues', [])
        
        timeline_events = []
        
        # Add frame analysis events
        for frame in frame_results:
            timeline_events.append({
                'timestamp': frame.get('timestamp', 0),
                'type': 'frame',
                'confidence': frame.get('confidence', 0),
                'frame_idx': frame.get('frame_idx', 0)
            })
        
        # Add audio anomaly regions
        for region in audio_regions:
            timeline_events.append({
                'timestamp': region.get('start_time', 0),
                'end_time': region.get('end_time', 0),
                'type': 'audio',
                'confidence': region.get('confidence', 0)
            })
        
        # Add temporal jumps
        for jump in temporal_jumps:
            start_frame = jump.get('start_frame', {})
            end_frame = jump.get('end_frame', {})
            timeline_events.append({
                'timestamp': start_frame.get('timestamp', 0),
                'end_time': end_frame.get('timestamp', 0),
                'type': 'temporal_jump',
                'confidence': abs(jump.get('confidence_change', 0))
            })
        
        # Add AV sync issues
        for issue in av_issues:
            timeline_events.append({
                'timestamp': issue.get('timestamp', 0),
                'type': 'av_sync',
                'confidence': issue.get('magnitude', 0)
            })
        
        # Sort timeline by timestamp
        timeline_events.sort(key=lambda x: x['timestamp'])
        
        # Calculate overall confidence per second
        duration = results.get('metadata', {}).get('duration', 0)
        if duration <= 0 and timeline_events:
            # Estimate duration from timeline
            duration = max(e.get('end_time', e.get('timestamp', 0)) for e in timeline_events)
        
        # Create confidence timeline (1 second intervals)
        confidence_timeline = []
        if duration > 0:
            interval = 1.0  # 1 second intervals
            for t in np.arange(0, duration, interval):
                # Find events that overlap with this time point
                overlapping = [
                    e for e in timeline_events 
                    if e.get('timestamp', 0) <= t <= e.get('end_time', e.get('timestamp', 0) + 0.1)
                ]
                
                # Calculate average confidence
                avg_conf = sum(e.get('confidence', 0) for e in overlapping) / len(overlapping) if overlapping else 0
                confidence_timeline.append({
                    'time': t,
                    'confidence': avg_conf
                })
        
        return {
            'duration': duration,
            'events': timeline_events,
            'confidence_timeline': confidence_timeline
        }
    
    def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the detection result using the Temporal Oracle enhanced algorithms.
        
        Args:
            result: Raw detection result from standard ensemble
            
        Returns:
            Enhanced detection result with multi-modal analysis
        """
        # Extract information from input result
        is_deepfake = result.get('is_deepfake', False)
        confidence = result.get('confidence', 0.5)
        metadata = result.get('metadata', {})
        
        # Extract component results
        frame_analysis = result.get('frame_analysis', {})
        audio_analysis = result.get('audio_analysis', {})
        temporal_analysis = result.get('temporal_analysis', {})
        av_sync_analysis = result.get('av_sync_analysis', {})
        individual_video_models = result.get('individual_video_models', [])
        
        # Create enhanced version of result
        enhanced_result = result.copy()
        
        # Determine optimal component weights based on video content
        component_weights = self.analyze_video_content(metadata)
        
        # Enhance temporal analysis
        enhanced_temporal = self.process_temporal_analysis(temporal_analysis)
        enhanced_result['temporal_analysis'] = enhanced_temporal
        
        # Calculate video model confidence (if available)
        video_model_confidence = 0.0
        if individual_video_models:
            # Use default video model weights
            weights = self.video_model_weights.copy()
            
            # Calculate weighted confidence
            for model_result in individual_video_models:
                model = model_result.get('model', '').lower()
                conf = model_result.get('confidence', 0)
                
                # Find matching model weight
                model_key = None
                for key in weights:
                    if key in model:
                        model_key = key
                        break
                
                if model_key:
                    video_model_confidence += conf * weights[model_key]
                else:
                    # Equal weight for unknown models
                    video_model_confidence += conf / len(individual_video_models)
        
        # Collect all component confidences
        component_confidences = {
            'frame_analysis': frame_analysis.get('confidence', 0),
            'audio_analysis': audio_analysis.get('confidence', 0),
            'temporal_analysis': temporal_analysis.get('confidence', 0),
            'av_sync_analysis': av_sync_analysis.get('confidence', 0),
            'video_models': video_model_confidence
        }
        
        # Calculate final confidence using weighted components
        final_confidence = 0.0
        for component, weight in component_weights.items():
            final_confidence += component_confidences.get(component, 0) * weight
        
        # Create enhanced timeline
        enhanced_timeline = self.generate_timeline(result)
        
        # Detect most likely manipulation type
        manipulation_type = self.detect_manipulation_type(result)
        
        # Update result with enhanced values
        enhanced_result['confidence'] = final_confidence
        enhanced_result['is_deepfake'] = final_confidence > 0.5
        enhanced_result['timeline'] = enhanced_timeline
        enhanced_result['manipulation_type'] = manipulation_type
        
        # Add Singularity Mode metadata
        enhanced_result['singularity_mode'] = {
            'name': self.name,
            'description': self.description,
            'component_weights': component_weights,
            'confidence_improvement': final_confidence - confidence,
            'manipulation_type': manipulation_type,
            'confidence_by_component': component_confidences
        }
        
        return enhanced_result


# app/core/singularity_manager.py
from typing import Dict, Any, List, Optional
from .singularity_modes import SingularityMode, VisualSentinel, AcousticGuardian, TemporalOracle

class SingularityManager:
    """
    Manager class for Singularity Modes.
    Handles registration, selection, and application of Singularity Modes.
    """
    
    def __init__(self):
        self.modes = {}
        self.register_default_modes()
    
    def register_default_modes(self):
        """Register the default Singularity Modes."""
        self.register_mode("image", VisualSentinel())
        self.register_mode("audio", AcousticGuardian())
        self.register_mode("video", TemporalOracle())
    
    def register_mode(self, media_type: str, mode: SingularityMode):
        """
        Register a Singularity Mode for a specific media type.
        
        Args:
            media_type: Type of media ('image', 'audio', or 'video')
            mode: SingularityMode instance
        """
        if media_type not in self.modes:
            self.modes[media_type] = []
        
        self.modes[media_type].append(mode)
    
    def get_available_modes(self, media_type: str) -> List[Dict[str, str]]:
        """
        Get information about available Singularity Modes for a media type.
        
        Args:
            media_type: Type of media ('image', 'audio', or 'video')
            
        Returns:
            List of dictionaries with mode name and description
        """
        if media_type not in self.modes:
            return []
        
        return [{'name': mode.name, 'description': mode.description} for mode in self.modes[media_type]]
    
    def apply_mode(self, media_type: str, mode_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a specific Singularity Mode to a detection result.
        
        Args:
            media_type: Type of media ('image', 'audio', or 'video')
            mode_name: Name of the Singularity Mode to apply
            result: Detection result to enhance
            
        Returns:
            Enhanced detection result
        """
        if media_type not in self.modes:
            return result
        
        # Find matching mode
        mode = None
        for m in self.modes[media_type]:
            if m.name == mode_name:
                mode = m
                break
        
        if mode is None:
            return result
        
        # Apply the mode
        return mode.process(result)


# Example integration with processor.py
# This would be placed in app/core/processor.py

# Add to DeepfakeDetectorProcessor class:
"""
def __init__(
    self,
    image_ensemble,
    audio_ensemble,
    video_ensemble,
    result_handler,
    cache_results=True,
    cache_dir='cache/results',
    singularity_manager=None  # Add this parameter
):
    self.image_ensemble = image_ensemble
    self.audio_ensemble = audio_ensemble
    self.video_ensemble = video_ensemble
    self.result_handler = result_handler
    self.cache_results = cache_results
    self.cache_dir = cache_dir
    
    # Initialize or use provided singularity manager
    self.singularity_manager = singularity_manager or SingularityManager()
    
    # Create cache directory if needed
    if cache_results:
        os.makedirs(cache_dir, exist_ok=True)

def process_media(self, file_path, use_singularity_mode=False, singularity_mode_name=None):
    # [Existing code for media type detection and processing]
    
    # Process based on media type
    if media_type == 'image':
        result = self._process_image(file_path)
    elif media_type == 'audio':
        result = self._process_audio(file_path)
    elif media_type == 'video':
        result = self._process_video(file_path)
    else:
        raise ValueError(f"Unsupported media type for file: {file_path}")
    
    # Apply Singularity Mode if requested
    if use_singularity_mode:
        if singularity_mode_name:
            # Apply specific Singularity Mode
            result = self.singularity_manager.apply_mode(media_type, singularity_mode_name, result)
        else:
            # Use default Singularity Mode for this media type
            available_modes = self.singularity_manager.get_available_modes(media_type)
            if available_modes:
                result = self.singularity_manager.apply_mode(media_type, available_modes[0]['name'], result)
    
    # [Rest of the existing code for metadata, caching, etc.]
"""