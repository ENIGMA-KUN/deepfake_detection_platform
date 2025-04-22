# detectors/ensemble_detector.py
from typing import List, Dict, Any, Optional, Union
import numpy as np

class EnsembleDetector:
    """Base class for ensemble-based deepfake detection."""
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None, threshold: float = 0.5):
        """
        Initialize the ensemble detector with multiple detector models.
        
        Args:
            detectors: List of detector instances to ensemble
            weights: Optional weights for each detector (default: equal weights)
            threshold: Confidence threshold for classifying as deepfake
        """
        self.detectors = detectors
        self.num_detectors = len(detectors)
        
        # Use equal weights if none provided
        if weights is None:
            self.weights = [1.0 / self.num_detectors] * self.num_detectors
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.threshold = threshold
        
    def load_all_models(self):
        """Load all detector models."""
        for detector in self.detectors:
            detector.load()
    
    def predict(self, media):
        """
        Predict whether media is authentic or deepfake using ensemble voting.
        
        Args:
            media: The media file to analyze
            
        Returns:
            Dict containing prediction results
        """
        # Get predictions from all detectors
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(media)
            predictions.append(pred)
        
        # Extract confidence scores
        confidence_scores = [detector.get_confidence(pred) for detector, pred in zip(self.detectors, predictions)]
        
        # Calculate weighted average confidence
        weighted_confidence = sum(w * c for w, c in zip(self.weights, confidence_scores))
        
        # Determine if deepfake based on threshold
        is_deepfake = weighted_confidence > self.threshold
        
        # Combine results
        result = {
            'is_deepfake': is_deepfake,
            'confidence': weighted_confidence,
            'individual_results': [
                {
                    'model': detector.name,
                    'confidence': score,
                    'weight': weight,
                    'raw_prediction': pred
                }
                for detector, score, weight, pred in zip(
                    self.detectors, confidence_scores, self.weights, predictions
                )
            ]
        }
        
        # Add visualization data (to be implemented by subclasses)
        result.update(self._get_visualization_data(predictions))
        
        return result
    
    def _get_visualization_data(self, predictions):
        """
        Generate visualization data from predictions.
        To be implemented by media-specific subclasses.
        """
        return {}
    
    def calibrate_weights(self, validation_data, ground_truth):
        """
        Calibrate weights based on performance on validation data.
        
        Args:
            validation_data: List of media files for validation
            ground_truth: List of boolean values indicating if each file is a deepfake
            
        Returns:
            Optimized weights
        """
        # This is a simplified implementation
        # In a real system, you'd use a more sophisticated optimization method
        
        # Get individual model accuracies
        accuracies = []
        
        for detector in self.detectors:
            correct = 0
            for media, is_fake in zip(validation_data, ground_truth):
                pred = detector.predict(media)
                confidence = detector.get_confidence(pred)
                pred_is_fake = confidence > self.threshold
                if pred_is_fake == is_fake:
                    correct += 1
            
            accuracy = correct / len(validation_data)
            accuracies.append(accuracy)
            
        # Use accuracies as weights
        total = sum(accuracies)
        self.weights = [acc / total for acc in accuracies]
        
        return self.weights
    
    def analyze_content(self, media):
        """
        Analyze content characteristics to determine optimal model weights.
        To be implemented by media-specific subclasses.
        
        Args:
            media: The media file to analyze
            
        Returns:
            Dict with content analysis results
        """
        return {}
    
    def adaptive_weighting(self, media, predictions):
        """
        Compute adaptive weights based on content characteristics.
        To be implemented by media-specific subclasses.
        
        Args:
            media: The media file being analyzed
            predictions: List of prediction results from individual detectors
            
        Returns:
            List of adaptive weights
        """
        # Default implementation uses fixed weights
        return self.weights.copy()


# detectors/image_detector/ensemble.py
import numpy as np

class ImageEnsembleDetector:
    """Ensemble detector for image deepfakes with Singularity Mode support."""
    
    def __init__(self, detectors, weights=None, threshold=0.5, enable_singularity=False):
        """
        Initialize the image ensemble detector.
        
        Args:
            detectors: List of detector instances to ensemble
            weights: Optional weights for each detector
            threshold: Confidence threshold for classifying as deepfake
            enable_singularity: Whether to enable advanced Singularity Mode features
        """
        self.detectors = detectors
        self.num_detectors = len(detectors)
        
        # Use equal weights if none provided
        if weights is None:
            self.weights = [1.0 / self.num_detectors] * self.num_detectors
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.threshold = threshold
        self.enable_singularity = enable_singularity
        
        # Content-based weighting factors
        self.content_weights = {
            "faces": {
                "vit": 0.25,
                "deit": 0.25,
                "beit": 0.20,
                "swin": 0.30
            },
            "landscapes": {
                "vit": 0.30,
                "deit": 0.20,
                "beit": 0.25,
                "swin": 0.25
            },
            "other": {
                "vit": 0.25,
                "deit": 0.25,
                "beit": 0.25,
                "swin": 0.25
            }
        }
    
    def predict(self, image):
        """
        Predict whether image is authentic or deepfake.
        
        Args:
            image: Numpy array representing the image
            
        Returns:
            Dict containing prediction results
        """
        # Get basic ensemble prediction
        result = self._predict(image)
        
        # Add image data for Singularity Mode processing
        result['image_data'] = image
        
        # If Singularity Mode enabled, enhance prediction with adaptive weighting
        if self.enable_singularity:
            result = self._enhance_with_singularity(image, result)
        
        return result
    
    def _predict(self, media):
        """
        Predict whether media is authentic or deepfake using ensemble voting.
        
        Args:
            media: The media file to analyze
            
        Returns:
            Dict containing prediction results
        """
        # Get predictions from all detectors
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(media)
            predictions.append(pred)
        
        # Extract confidence scores
        confidence_scores = [detector.get_confidence(pred) for detector, pred in zip(self.detectors, predictions)]
        
        # Calculate weighted average confidence
        weighted_confidence = sum(w * c for w, c in zip(self.weights, confidence_scores))
        
        # Determine if deepfake based on threshold
        is_deepfake = weighted_confidence > self.threshold
        
        # Combine results
        result = {
            'is_deepfake': is_deepfake,
            'confidence': weighted_confidence,
            'individual_results': [
                {
                    'model': detector.name,
                    'confidence': score,
                    'weight': weight,
                    'raw_prediction': pred
                }
                for detector, score, weight, pred in zip(
                    self.detectors, confidence_scores, self.weights, predictions
                )
            ]
        }
        
        # Add visualization data (to be implemented by subclasses)
        result.update(self._get_visualization_data(predictions))
        
        return result
    
    def _get_visualization_data(self, predictions):
        """
        Generate visualization data from predictions.
        
        Args:
            predictions: List of prediction results from individual detectors
            
        Returns:
            Dict with visualization data
        """
        # Extract heatmaps from predictions (if available)
        heatmaps = []
        for pred in predictions:
            if 'heatmap' in pred and pred['heatmap'] is not None:
                heatmaps.append(pred['heatmap'])
        
        if not heatmaps:
            return {'heatmap': None}
        
        # Ensure all heatmaps have the same shape
        # In practice, you might need to resize them
        
        # Combine heatmaps with weights
        combined_heatmap = np.zeros_like(heatmaps[0])
        for i, heatmap in enumerate(heatmaps):
            weight = self.weights[i]
            combined_heatmap += heatmap * weight
            
        # Normalize combined heatmap to [0, 1] range
        if np.max(combined_heatmap) > 0:
            combined_heatmap = combined_heatmap / np.max(combined_heatmap)
            
        return {'heatmap': combined_heatmap}
    
    def _enhance_with_singularity(self, image, result):
        """
        Enhance prediction with Singularity Mode features.
        
        Args:
            image: Numpy array representing the image
            result: Basic ensemble prediction result
            
        Returns:
            Enhanced result with Singularity Mode features
        """
        # Analyze content
        content_type = self._analyze_image_content(image)
        
        # Get adaptive weights
        adaptive_weights = self._get_adaptive_weights(content_type, result['individual_results'])
        
        # Recalculate confidence with adaptive weights
        weighted_confidence = 0.0
        for idx, ind_result in enumerate(result['individual_results']):
            model_name = ind_result['model'].lower()
            weight = self._get_model_weight(model_name, adaptive_weights)
            weighted_confidence += ind_result['confidence'] * weight
            
            # Update individual result weights
            result['individual_results'][idx]['adaptive_weight'] = weight
        
        # Extract heatmaps and anomaly regions
        heatmaps = self._extract_heatmaps(result['individual_results'])
        anomaly_regions = self._extract_anomaly_regions(result['individual_results'])
        
        # Generate enhanced heatmap and merged anomaly regions
        enhanced_heatmap = self._generate_enhanced_heatmap(heatmaps, adaptive_weights)
        merged_regions = self._merge_anomaly_regions(anomaly_regions)
        
        # Update result
        result['singularity'] = {
            'enabled': True,
            'mode': 'Visual Sentinel',
            'content_type': content_type,
            'adaptive_weights': adaptive_weights,
            'enhanced_confidence': weighted_confidence,
            'confidence_improvement': weighted_confidence - result['confidence']
        }
        
        result['confidence'] = weighted_confidence
        result['is_deepfake'] = weighted_confidence > self.threshold
        
        if enhanced_heatmap is not None:
            result['enhanced_heatmap'] = enhanced_heatmap
        
        if merged_regions:
            result['merged_anomaly_regions'] = merged_regions
        
        return result
    
    def _analyze_image_content(self, image):
        """
        Analyze image content to determine type.
        
        Args:
            image: Numpy array representing the image
            
        Returns:
            String indicating content type
        """
        # In a real implementation, would use computer vision to classify content
        # For now, assume images mostly contain faces
        return "faces"
    
    def _get_adaptive_weights(self, content_type, individual_results):
        """
        Get content-adaptive weights for models.
        
        Args:
            content_type: String indicating content type
            individual_results: List of individual model results
            
        Returns:
            Dict of model types to weights
        """
        # Get base weights for content type
        if content_type in self.content_weights:
            weights = self.content_weights[content_type].copy()
        else:
            weights = self.content_weights["other"].copy()
        
        # Adjust weights based on confidence
        confidences = {}
        for result in individual_results:
            model_name = result['model'].lower()
            confidence = result['confidence']
            
            # Determine model type
            model_type = None
            for key in weights.keys():
                if key in model_name:
                    model_type = key
                    break
            
            if model_type:
                confidences[model_type] = confidence
        
        # Calculate average confidence
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
        
        # Adjust weights based on confidence deviation from average
        for model_type, confidence in confidences.items():
            # More confident models get higher weight
            confidence_factor = confidence / avg_confidence if avg_confidence > 0 else 1.0
            weights[model_type] *= confidence_factor
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _get_model_weight(self, model_name, adaptive_weights):
        """
        Get weight for a specific model from adaptive weights.
        
        Args:
            model_name: Name of model
            adaptive_weights: Dict of model types to weights
            
        Returns:
            Weight value
        """
        for model_type, weight in adaptive_weights.items():
            if model_type in model_name:
                return weight
        
        # Default to equal weight if no match
        return 1.0 / len(adaptive_weights)
    
    def _extract_heatmaps(self, individual_results):
        """
        Extract heatmaps from individual results.
        
        Args:
            individual_results: List of individual model results
            
        Returns:
            Dict of model names to heatmaps
        """
        heatmaps = {}
        
        for result in individual_results:
            model = result['model']
            raw_pred = result.get('raw_prediction', {})
            
            if 'heatmap' in raw_pred:
                heatmaps[model] = raw_pred['heatmap']
        
        return heatmaps
    
    def _generate_enhanced_heatmap(self, heatmaps, adaptive_weights):
        """
        Generate enhanced heatmap using adaptive weights.
        
        Args:
            heatmaps: Dict of model names to heatmaps
            adaptive_weights: Dict of model types to weights
            
        Returns:
            Combined heatmap
        """
        if not heatmaps:
            return None
        
        # Get first heatmap shape
        first_map = next(iter(heatmaps.values()))
        combined = np.zeros_like(first_map)
        
        # Combine heatmaps with weights
        for model, heatmap in heatmaps.items():
            # Skip if shapes don't match
            if heatmap.shape != first_map.shape:
                continue
                
            # Determine model type
            weight = 1.0 / len(heatmaps)  # Default weight
            for model_type, type_weight in adaptive_weights.items():
                if model_type in model.lower():
                    weight = type_weight
                    break
            
            combined += heatmap * weight
        
        # Normalize
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        return combined
    
    def _extract_anomaly_regions(self, individual_results):
        """
        Extract anomaly regions from individual results.
        
        Args:
            individual_results: List of individual model results
            
        Returns:
            List of anomaly regions with model information
        """
        all_regions = []
        
        for result in individual_results:
            model = result['model']
            raw_pred = result.get('raw_prediction', {})
            
            if 'anomaly_regions' in raw_pred and raw_pred['anomaly_regions']:
                for region in raw_pred['anomaly_regions']:
                    # Add model information to each region
                    region_with_model = region.copy()
                    region_with_model['model'] = model
                    all_regions.append(region_with_model)
        
        return all_regions
    
    def _merge_anomaly_regions(self, regions):
        """
        Merge overlapping anomaly regions.
        
        Args:
            regions: List of anomaly regions
            
        Returns:
            List of merged regions
        """
        if not regions:
            return []
        
        # Sort by start time
        sorted_regions = sorted(regions, key=lambda x: x.get('start_time', 0))
        
        merged = []
        current = sorted_regions[0].copy()
        current_models = [current.pop('model')]
        
        for region in sorted_regions[1:]:
            model = region.pop('model')
            
            # Check if regions overlap
            if region['start_time'] <= current['end_time']:
                # Extend current region
                current['end_time'] = max(current['end_time'], region['end_time'])
                # Update confidence (use max)
                current['confidence'] = max(current['confidence'], region['confidence'])
                # Add model to list if not already present
                if model not in current_models:
                    current_models.append(model)
            else:
                # No overlap, add current to results and start new one
                current['models'] = current_models
                merged.append(current)
                current = region.copy()
                current_models = [model]
        
        # Add the last region
        current['models'] = current_models
        merged.append(current)
        
        return merged


# detectors/audio_detector/ensemble.py
import numpy as np

class AudioEnsembleDetector:
    """Ensemble detector for audio deepfakes with Singularity Mode support."""
    
    def __init__(self, detectors, weights=None, threshold=0.5, enable_singularity=False):
        """
        Initialize the audio ensemble detector.
        
        Args:
            detectors: List of detector instances to ensemble
            weights: Optional weights for each detector
            threshold: Confidence threshold for classifying as deepfake
            enable_singularity: Whether to enable advanced Singularity Mode features
        """
        self.detectors = detectors
        self.num_detectors = len(detectors)
        
        # Use equal weights if none provided
        if weights is None:
            self.weights = [1.0 / self.num_detectors] * self.num_detectors
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.threshold = threshold
        self.enable_singularity = enable_singularity
        
        # Audio type weighting factors
        self.audio_type_weights = {
            "speech": {
                "wav2vec2": 0.35,
                "xlsr": 0.25,
                "mamba": 0.25,
                "tcn": 0.15
            },
            "music": {
                "wav2vec2": 0.25,
                "xlsr": 0.20,
                "mamba": 0.35,
                "tcn": 0.20
            },
            "ambient": {
                "wav2vec2": 0.25,
                "xlsr": 0.25,
                "mamba": 0.25,
                "tcn": 0.25
            }
        }
        
        # Frequency band importance weights
        self.frequency_bands = {
            "low": (0, 500),      # Hz ranges
            "mid": (500, 2000),
            "high": (2000, 8000)
        }
    
    def predict(self, audio):
        """
        Predict whether audio is authentic or deepfake.
        
        Args:
            audio: Audio data to analyze
            
        Returns:
            Dict containing prediction results
        """
        # Get basic ensemble prediction
        result = self._predict(audio)
        
        # Add audio data for Singularity Mode processing
        result['audio_data'] = audio
        
        # If Singularity Mode enabled, enhance prediction with adaptive weighting
        if self.enable_singularity:
            result = self._enhance_with_singularity(audio, result)
        
        return result
    
    def _predict(self, media):
        """
        Predict whether media is authentic or deepfake using ensemble voting.
        
        Args:
            media: The media file to analyze
            
        Returns:
            Dict containing prediction results
        """
        # Get predictions from all detectors
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(media)
            predictions.append(pred)
        
        # Extract confidence scores
        confidence_scores = [detector.get_confidence(pred) for detector, pred in zip(self.detectors, predictions)]
        
        # Calculate weighted average confidence
        weighted_confidence = sum(w * c for w, c in zip(self.weights, confidence_scores))
        
        # Determine if deepfake based on threshold
        is_deepfake = weighted_confidence > self.threshold
        
        # Combine results
        result = {
            'is_deepfake': is_deepfake,
            'confidence': weighted_confidence,
            'individual_results': [
                {
                    'model': detector.name,
                    'confidence': score,
                    'weight': weight,
                    'raw_prediction': pred
                }
                for detector, score, weight, pred in zip(
                    self.detectors, confidence_scores, self.weights, predictions
                )
            ]
        }
        
        # Add visualization data (to be implemented by subclasses)
        result.update(self._get_visualization_data(predictions))
        
        return result
    
    def _get_visualization_data(self, predictions):
        """
        Generate visualization data from predictions.
        
        Args:
            predictions: List of prediction results from individual detectors
            
        Returns:
            Dict with visualization data
        """
        # Extract spectrograms from predictions (if available)
        spectrograms = []
        anomaly_regions = []
        
        for pred in predictions:
            if 'spectrogram' in pred and pred['spectrogram'] is not None:
                spectrograms.append(pred['spectrogram'])
                
            if 'anomaly_regions' in pred and pred['anomaly_regions'] is not None:
                anomaly_regions.extend(pred['anomaly_regions'])
        
        result = {}
        
        # Process spectrograms
        if spectrograms:
            # Combine spectrograms with weights
            combined_spec = np.zeros_like(spectrograms[0])
            for i, spec in enumerate(spectrograms):
                weight = self.weights[i]
                combined_spec += spec * weight
                
            # Normalize
            if np.max(combined_spec) > 0:
                combined_spec = combined_spec / np.max(combined_spec)
                
            result['spectrogram'] = combined_spec
        else:
            result['spectrogram'] = None
            
        # Process anomaly regions
        if anomaly_regions:
            result['anomaly_regions'] = self._merge_anomaly_regions(anomaly_regions)
        else:
            result['anomaly_regions'] = []
            
        return result
    
    def _enhance_with_singularity(self, audio, result):
        """
        Enhance prediction with Singularity Mode features.
        
        Args:
            audio: Audio data
            result: Basic ensemble prediction result
            
        Returns:
            Enhanced result with Singularity Mode features
        """
        # Analyze audio type
        audio_type = self._analyze_audio_type(audio)
        
        # Get adaptive weights
        adaptive_weights = self._get_adaptive_weights(audio_type, result['individual_results'])
        
        # Recalculate confidence with adaptive weights
        weighted_confidence = 0.0
        for idx, ind_result in enumerate(result['individual_results']):
            model_name = ind_result['model'].lower()
            weight = self._get_model_weight(model_name, adaptive_weights)
            weighted_confidence += ind_result['confidence'] * weight
            
            # Update individual result weights
            result['individual_results'][idx]['adaptive_weight'] = weight
        
        # Extract spectrograms and anomaly regions
        spectrograms = self._extract_spectrograms(result['individual_results'])
        anomaly_regions = self._extract_anomaly_regions(result['individual_results'])
        
        # Generate enhanced spectrogram and merged anomaly regions
        enhanced_spectrogram = self._generate_enhanced_spectrogram(spectrograms, adaptive_weights)
        merged_regions = self._merge_anomaly_regions(anomaly_regions)
        
        # Analyze frequency bands
        frequency_analysis = self._analyze_frequency_bands(spectrograms)
        
        # Update result
        result['singularity'] = {
            'enabled': True,
            'mode': 'Acoustic Guardian',
            'audio_type': audio_type,
            'adaptive_weights': adaptive_weights,
            'enhanced_confidence': weighted_confidence,
            'confidence_improvement': weighted_confidence - result['confidence'],
            'frequency_analysis': frequency_analysis
        }
        
        result['confidence'] = weighted_confidence
        result['is_deepfake'] = weighted_confidence > self.threshold
        
        if enhanced_spectrogram is not None:
            result['enhanced_spectrogram'] = enhanced_spectrogram
            
        if merged_regions:
            result['merged_anomaly_regions'] = merged_regions
        
        return result
    
    def _analyze_audio_type(self, audio):
        """
        Analyze audio to determine its type.
        
        Args:
            audio: Audio data
            
        Returns:
            String indicating audio type
        """
        # In a real implementation, would use audio classification
        # For now, assume most inputs are speech
        return "speech"
    
    def _get_adaptive_weights(self, audio_type, individual_results):
        """
        Get content-adaptive weights for models.
        
        Args:
            audio_type: String indicating audio type
            individual_results: List of individual model results
            
        Returns:
            Dict of model types to weights
        """
        # Get base weights for audio type
        if audio_type in self.audio_type_weights:
            weights = self.audio_type_weights[audio_type].copy()
        else:
            weights = self.audio_type_weights["speech"].copy()
        
        # Adjust weights based on confidence
        confidences = {}
        for result in individual_results:
            model_name = result['model'].lower()
            confidence = result['confidence']
            
            # Determine model type
            model_type = None
            for key in weights.keys():
                if key in model_name:
                    model_type = key
                    break
            
            if model_type:
                confidences[model_type] = confidence
        
        # Calculate average confidence
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
        
        # Adjust weights based on confidence deviation from average
        for model_type, confidence in confidences.items():
            # More confident models get higher weight
            confidence_factor = confidence / avg_confidence if avg_confidence > 0 else 1.0
            weights[model_type] *= confidence_factor
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _get_model_weight(self, model_name, adaptive_weights):
        """
        Get weight for a specific model from adaptive weights.
        
        Args:
            model_name: Name of model
            adaptive_weights: Dict of model types to weights
            
        Returns:
            Weight value
        """
        for model_type, weight in adaptive_weights.items():
            if model_type in model_name:
                return weight
        
        # Default to equal weight if no match
        return 1.0 / len(adaptive_weights)
    
    def _extract_spectrograms(self, individual_results):
        """
        Extract spectrograms from individual results.
        
        Args:
            individual_results: List of individual model results
            
        Returns:
            Dict of model names to spectrograms
        """
        spectrograms = {}
        
        for result in individual_results:
            model = result['model']
            raw_pred = result.get('raw_prediction', {})
            
            if 'spectrogram' in raw_pred:
                spectrograms[model] = raw_pred['spectrogram']
        
        return spectrograms
    
    def _extract_anomaly_regions(self, individual_results):
        """
        Extract anomaly regions from individual results.
        
        Args:
            individual_results: List of individual model results
            
        Returns:
            List of anomaly regions with model information
        """
        all_regions = []
        
        for result in individual_results:
            model = result['model']
            raw_pred = result.get('raw_prediction', {})
            
            if 'anomaly_regions' in raw_pred and raw_pred['anomaly_regions']:
                for region in raw_pred['anomaly_regions']:
                    # Add model information to each region
                    region_with_model = region.copy()
                    region_with_model['model'] = model
                    all_regions.append(region_with_model)
        
        return all_regions
    
    def _generate_enhanced_spectrogram(self, spectrograms, adaptive_weights):
        """
        Generate enhanced spectrogram using adaptive weights.
        
        Args:
            spectrograms: Dict of model names to spectrograms
            adaptive_weights: Dict of model types to weights
            
        Returns:
            Combined spectrogram
        """
        if not spectrograms:
            return None
        
        # Get first spectrogram shape
        first_spec = next(iter(spectrograms.values()))
        combined = np.zeros_like(first_spec)
        
        # Combine spectrograms with weights
        for model, spectrogram in spectrograms.items():
            # Skip if shapes don't match
            if spectrogram.shape != first_spec.shape:
                continue
                
            # Determine model type
            weight = 1.0 / len(spectrograms)  # Default weight
            for model_type, type_weight in adaptive_weights.items():
                if model_type in model.lower():
                    weight = type_weight
                    break
            
            combined += spectrogram * weight
        
        # Normalize
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        return combined
    
    def _merge_anomaly_regions(self, regions):
        """
        Merge overlapping anomaly regions.
        
        Args:
            regions: List of anomaly regions
            
        Returns:
            List of merged regions
        """
        if not regions:
            return []
        
        # Sort by start time
        sorted_regions = sorted(regions, key=lambda x: x.get('start_time', 0))
        
        merged = []
        current = sorted_regions[0].copy()
        current_models = [current.pop('model')]
        
        for region in sorted_regions[1:]:
            model = region.pop('model')
            
            # Check if regions overlap
            if region['start_time'] <= current['end_time']:
                # Extend current region
                current['end_time'] = max(current['end_time'], region['end_time'])
                # Update confidence (use max)
                current['confidence'] = max(current['confidence'], region['confidence'])
                # Add model to list if not already present
                if model not in current_models:
                    current_models.append(model)
            else:
                # No overlap, add current to results and start new one
                current['models'] = current_models
                merged.append(current)
                current = region.copy()
                current_models = [model]
        
        # Add the last region
        current['models'] = current_models
        merged.append(current)
        
        return merged
    
    def _analyze_frequency_bands(self, spectrograms):
        """
        Analyze importance of different frequency bands.
        
        Args:
            spectrograms: Dict of model names to spectrograms
            
        Returns:
            Dict with frequency band analysis
        """
        # In a real implementation, would analyze frequency content
        # For now, return placeholder data
        return {
            'low_frequency_importance': 0.3,
            'mid_frequency_importance': 0.5,
            'high_frequency_importance': 0.2,
            'suspected_manipulation_bands': ['mid', 'high']
        }


# detectors/video_detector/ensemble.py
import numpy as np
import cv2
import os

class VideoEnsembleDetector:
    """Ensemble detector for video deepfakes with Singularity Mode support."""
    
    def __init__(
        self, 
        video_detectors,
        image_ensemble=None,
        audio_ensemble=None,
        weights=None, 
        threshold=0.5,
        frame_sample_rate=5,  # Analyze every Nth frame
        enable_singularity=False
    ):
        """
        Initialize the video ensemble detector.
        
        Args:
            video_detectors: List of video detector instances
            image_ensemble: Optional image ensemble detector for frame analysis
            audio_ensemble: Optional audio ensemble detector for audio analysis
            weights: Optional weights for video detectors
            threshold: Confidence threshold for classifying as deepfake
            frame_sample_rate: Analyze every Nth frame
            enable_singularity: Whether to enable advanced Singularity Mode features
        """
        self.detectors = video_detectors
        self.num_detectors = len(video_detectors)
        
        # Use equal weights if none provided
        if weights is None:
            self.weights = [1.0 / self.num_detectors] * self.num_detectors
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        self.threshold = threshold
        self.image_ensemble = image_ensemble
        self.audio_ensemble = audio_ensemble
        self.frame_sample_rate = frame_sample_rate
        self.enable_singularity = enable_singularity
        
        # Component weights for different video types
        self.component_weights = {
            "talking_head": {
                "frame_analysis": 0.25,
                "audio_analysis": 0.25,
                "temporal_analysis": 0.20,
                "av_sync_analysis": 0.20,
                "video_models": 0.10
            },
            "action": {
                "frame_analysis": 0.20,
                "audio_analysis": 0.15,
                "temporal_analysis": 0.40,
                "av_sync_analysis": 0.10,
                "video_models": 0.15
            },
            "scenery": {
                "frame_analysis": 0.35,
                "audio_analysis": 0.10,
                "temporal_analysis": 0.30,
                "av_sync_analysis": 0.05,
                "video_models": 0.20
            }
        }
        
        # Video model weights
        self.video_model_weights = {
            "genconvit": 0.25,
            "timesformer": 0.25,
            "slowfast": 0.15,
            "swin": 0.25,
            "x3d": 0.10
        }
    
    def predict(self, video_path):
        """
        Predict whether video is authentic or deepfake.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with prediction results
        """
        # First, get predictions from dedicated video models
        video_model_predictions = []
        for detector in self.detectors:
            pred = detector.predict(video_path)
            video_model_predictions.append(pred)
        
        # Extract video confidence scores
        video_confidence_scores = [detector.get_confidence(pred) for detector, pred in 
                                  zip(self.detectors, video_model_predictions)]
        
        # Calculate weighted video confidence
        weighted_video_confidence = sum(w * c for w, c in zip(self.weights, video_confidence_scores))
        
        # Frame-by-frame analysis (if image ensemble is available)
        frame_analysis = self._analyze_frames(video_path) if self.image_ensemble else None
        
        # Audio analysis (if audio ensemble is available)
        audio_analysis = self._analyze_audio(video_path) if self.audio_ensemble else None
        
        # Temporal consistency analysis
        temporal_analysis = self._analyze_temporal_consistency(frame_analysis) if frame_analysis else None
        
        # Audio-visual sync analysis
        sync_analysis = self._analyze_av_sync(frame_analysis, audio_analysis) if frame_analysis and audio_analysis else None
        
        # Combine all signals for final prediction
        # This is a simple weighted approach - could be more sophisticated
        all_confidences = [weighted_video_confidence]
        all_weights = [0.4]  # 40% weight to dedicated video models
        
        if frame_analysis:
            all_confidences.append(frame_analysis['confidence'])
            all_weights.append(0.2)  # 20% weight to frame analysis
            
        if temporal_analysis:
            all_confidences.append(temporal_analysis['confidence'])
            all_weights.append(0.15)  # 15% weight to temporal consistency
            
        if audio_analysis:
            all_confidences.append(audio_analysis['confidence'])
            all_weights.append(0.15)  # 15% weight to audio analysis
            
        if sync_analysis:
            all_confidences.append(sync_analysis['confidence'])
            all_weights.append(0.1)  # 10% weight to A/V sync analysis
            
        # Normalize weights
        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        
        # Calculate final confidence
        final_confidence = sum(c * w for c, w in zip(all_confidences, normalized_weights))
        is_deepfake = final_confidence > self.threshold
        
        # Extract video metadata
        metadata = self._extract_video_metadata(video_path)
        
        # Prepare comprehensive result
        result = {
            'is_deepfake': is_deepfake,
            'confidence': final_confidence,
            'video_model_confidence': weighted_video_confidence,
            'frame_analysis': frame_analysis,
            'temporal_analysis': temporal_analysis,
            'audio_analysis': audio_analysis,
            'av_sync_analysis': sync_analysis,
            'individual_video_models': [
                {
                    'model': detector.name,
                    'confidence': score,
                    'weight': weight
                }
                for detector, score, weight in zip(
                    self.detectors, video_confidence_scores, self.weights
                )
            ],
            'timeline': self._generate_timeline(
                frame_analysis, 
                audio_analysis, 
                temporal_analysis
            ),
            'metadata': metadata
        }
        
        # Apply Singularity Mode enhancements if enabled
        if self.enable_singularity:
            result = self._enhance_with_singularity(video_path, result)
        
        return result
    
    def _enhance_with_singularity(self, video_path, result):
        """
        Enhance result with Temporal Oracle Singularity Mode.
        
        Args:
            video_path: Path to video file
            result: Basic prediction result
            
        Returns:
            Enhanced result with Singularity Mode features
        """
        # Determine video type based on metadata and analysis
        video_type = self._determine_video_type(result)
        
        # Get optimal component weights
        component_weights = self._get_adaptive_component_weights(video_type, result)
        
        # Enhanced temporal analysis
        enhanced_temporal = self._enhance_temporal_analysis(result['temporal_analysis'])
        
        # Detect most likely manipulation type
        manipulation_type = self._detect_manipulation_type(result)
        
        # Recalculate confidence with adaptive component weights
        component_confidences = {
            'frame_analysis': result['frame_analysis']['confidence'] if result['frame_analysis'] else 0,
            'audio_analysis': result['audio_analysis']['confidence'] if result['audio_analysis'] else 0,
            'temporal_analysis': enhanced_temporal['confidence'] if enhanced_temporal else 0,
            'av_sync_analysis': result['av_sync_analysis']['confidence'] if result['av_sync_analysis'] else 0,
            'video_models': result['video_model_confidence']
        }
        
        enhanced_confidence = sum(
            component_weights.get(comp, 0) * conf 
            for comp, conf in component_confidences.items()
        )
        
        # Generate enhanced timeline
        enhanced_timeline = self._generate_enhanced_timeline(result)
        
        # Update result with Singularity Mode enhancements
        enhanced_result = result.copy()
        enhanced_result['singularity'] = {
            'enabled': True,
            'mode': 'Temporal Oracle',
            'video_type': video_type,
            'component_weights': component_weights,
            'enhanced_confidence': enhanced_confidence,
            'confidence_improvement': enhanced_confidence - result['confidence'],
            'manipulation_type': manipulation_type,
            'confidence_by_component': component_confidences
        }
        
        enhanced_result['confidence'] = enhanced_confidence
        enhanced_result['is_deepfake'] = enhanced_confidence > self.threshold
        enhanced_result['temporal_analysis'] = enhanced_temporal if enhanced_temporal else result['temporal_analysis']
        enhanced_result['enhanced_timeline'] = enhanced_timeline
        enhanced_result['manipulation_type'] = manipulation_type
        
        return enhanced_result
    
    def _determine_video_type(self, result):
        """
        Determine video type based on metadata and analysis.
        
        Args:
            result: Prediction result
            
        Returns:
            String indicating video type
        """
        # In a real implementation, would use content classification
        # For now, assume most videos are talking heads
        return "talking_head"
    
    def _get_adaptive_component_weights(self, video_type, result):
        """
        Get adaptive component weights based on video type and content.
        
        Args:
            video_type: String indicating video type
            result: Prediction result
            
        Returns:
            Dict of component names to weights
        """
        # Get base weights for video type
        if video_type in self.component_weights:
            weights = self.component_weights[video_type].copy()
        else:
            weights = self.component_weights["talking_head"].copy()
        
        # Adjust based on component confidence separation
        confidences = {
            'frame_analysis': result['frame_analysis']['confidence'] if result['frame_analysis'] else 0,
            'audio_analysis': result['audio_analysis']['confidence'] if result['audio_analysis'] else 0,
            'temporal_analysis': result['temporal_analysis']['confidence'] if result['temporal_analysis'] else 0,
            'av_sync_analysis': result['av_sync_analysis']['confidence'] if result['av_sync_analysis'] else 0,
            'video_models': result['video_model_confidence']
        }
        
        # Calculate average confidence
        valid_confidences = [c for c in confidences.values() if c > 0]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.5
        
        # Adjust weights - give higher weights to components with higher confidence
        for component, confidence in confidences.items():
            if confidence > 0:
                confidence_factor = confidence / avg_confidence
                weights[component] *= confidence_factor
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _enhance_temporal_analysis(self, temporal_analysis):
        """
        Enhance temporal analysis with additional metrics.
        
        Args:
            temporal_analysis: Original temporal analysis
            
        Returns:
            Enhanced temporal analysis
        """
        if not temporal_analysis:
            return None
        
        enhanced = temporal_analysis.copy()
        
        # Add temporal stability score
        if 'confidence_variance' in temporal_analysis:
            variance = temporal_analysis['confidence_variance']
            stability = 1.0 / (1.0 + variance * 10)  # Transform to 0-1 range
            enhanced['temporal_stability'] = stability
        
        # Analyze jumps if available
        if 'significant_jumps' in temporal_analysis and temporal_analysis['significant_jumps']:
            jumps = temporal_analysis['significant_jumps']
            
            # Calculate jump statistics
            jump_magnitudes = [abs(j.get('confidence_change', 0)) for j in jumps]
            enhanced['avg_jump_magnitude'] = sum(jump_magnitudes) / len(jump_magnitudes)
            enhanced['max_jump_magnitude'] = max(jump_magnitudes)
            
            # Flag suspicious regions
            suspicious_jumps = [j for j in jumps if abs(j.get('confidence_change', 0)) > 0.3]
            enhanced['suspicious_jump_count'] = len(suspicious_jumps)
            
            if 'confidence_trend' in temporal_analysis:
                # Analyze overall trend direction
                trend = temporal_analysis['confidence_trend']
                if trend and len(trend) > 1:
                    trend_times, trend_values = zip(*trend)
                    if trend_values[-1] > trend_values[0] + 0.1:
                        enhanced['trend_direction'] = 'increasing'
                    elif trend_values[0] > trend_values[-1] + 0.1:
                        enhanced['trend_direction'] = 'decreasing'
                    else:
                        enhanced['trend_direction'] = 'stable'
        
        return enhanced
    
    def _detect_manipulation_type(self, result):
        """
        Detect most likely manipulation type from results.
        
        Args:
            result: Prediction result
            
        Returns:
            String indicating manipulation type
        """
        # Define manipulation types and their indicators
        indicators = {
            'face_swap': 0,
            'lip_sync': 0,
            'full_synthesis': 0,
            'audio_only': 0,
            'temporal_edit': 0
        }
        
        # Check frame analysis results
        if result['frame_analysis']:
            frame_conf = result['frame_analysis']['confidence']
            if frame_conf > 0.7:
                indicators['face_swap'] += 2
                indicators['full_synthesis'] += 1
            
            # Check suspicious frames
            if 'top_suspicious_frames' in result['frame_analysis']:
                if len(result['frame_analysis']['top_suspicious_frames']) > 3:
                    indicators['face_swap'] += 1
        
        # Check audio analysis
        if result['audio_analysis']:
            audio_conf = result['audio_analysis']['confidence']
            if audio_conf > 0.7:
                indicators['audio_only'] += 2
                indicators['lip_sync'] += 1
                indicators['full_synthesis'] += 0.5
        
        # Check temporal analysis
        if result['temporal_analysis']:
            temp_conf = result['temporal_analysis']['confidence']
            if temp_conf > 0.7:
                indicators['temporal_edit'] += 2
                indicators['full_synthesis'] += 1
            
            if 'significant_jumps' in result['temporal_analysis']:
                jumps = result['temporal_analysis']['significant_jumps']
                if jumps and len(jumps) > 2:
                    indicators['temporal_edit'] += 1
        
        # Check AV sync analysis
        if result['av_sync_analysis']:
            sync_conf = result['av_sync_analysis']['confidence']
            if sync_conf > 0.7:
                indicators['lip_sync'] += 2
        
        # Return type with highest indicator score
        if max(indicators.values()) > 0:
            return max(indicators.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    def _generate_enhanced_timeline(self, result):
        """
        Generate enhanced timeline with additional metrics.
        
        Args:
            result: Prediction result
            
        Returns:
            Enhanced timeline data
        """
        # Start with base timeline
        timeline = result.get('timeline', {'events': [], 'duration': 0})
        
        # Add manipulation type indicators
        if 'manipulation_type' in result and result['manipulation_type'] != 'unknown':
            manip_type = result['manipulation_type']
            
            # Add type-specific markers
            if manip_type == 'face_swap' and result['frame_analysis']:
                # Mark top suspicious frames
                top_frames = result['frame_analysis'].get('top_suspicious_frames', [])
                for frame in top_frames:
                    timeline['events'].append({
                        'timestamp': frame.get('timestamp', 0),
                        'type': 'face_swap_region',
                        'confidence': frame.get('confidence', 0),
                        'frame_idx': frame.get('frame_idx', 0)
                    })
            
            elif manip_type == 'lip_sync' and result['av_sync_analysis']:
                # Mark sync issues
                sync_issues = result['av_sync_analysis'].get('sync_issues', [])
                for issue in sync_issues:
                    timeline['events'].append({
                        'timestamp': issue.get('timestamp', 0),
                        'type': 'lip_sync_issue',
                        'confidence': issue.get('magnitude', 0)
                    })
        
        # Add confidence timeline if not present
        if 'confidence_timeline' not in timeline and timeline['duration'] > 0:
            # Generate confidence over time (one point per second)
            confidence_timeline = []
            for t in np.arange(0, timeline['duration'], 1.0):
                # Find events at this time
                events_at_time = [
                    e for e in timeline['events'] 
                    if e.get('timestamp', 0) <= t <= e.get('end_time', e.get('timestamp', 0) + 0.1)
                ]
                
                # Calculate average confidence
                avg_conf = sum(e.get('confidence', 0) for e in events_at_time) / len(events_at_time) if events_at_time else 0
                
                confidence_timeline.append({
                    'time': t,
                    'confidence': avg_conf
                })
            
            timeline['confidence_timeline'] = confidence_timeline
        
        return timeline
    
    def _analyze_frames(self, video_path):
        """
        Analyze video frames using image ensemble.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with frame analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Analyze frames at regular intervals
        frame_results = []
        frame_confidences = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only analyze every Nth frame
            if frame_idx % self.frame_sample_rate == 0:
                # Convert frame to RGB (OpenCV uses BGR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Analyze frame with image ensemble
                prediction = self.image_ensemble.predict(rgb_frame)
                
                # Save timestamp and result
                timestamp = frame_idx / fps
                frame_result = {
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'confidence': prediction['confidence'],
                    'heatmap': prediction.get('heatmap')
                }
                
                frame_results.append(frame_result)
                frame_confidences.append(prediction['confidence'])
                
            frame_idx += 1
                
        cap.release()
        
        # Calculate overall confidence from frame analysis
        if frame_confidences:
            # Using mean confidence (could use other strategies)
            mean_confidence = np.mean(frame_confidences)
            
            # Find frames with highest deepfake confidence
            sorted_indices = np.argsort(frame_confidences)[::-1]  # Descending order
            top_frames = [frame_results[i] for i in sorted_indices[:5]]  # Top 5 suspicious frames
            
            return {
                'confidence': mean_confidence,
                'frame_count': frame_count,
                'analyzed_frame_count': len(frame_results),
                'duration': duration,
                'fps': fps,
                'frame_results': frame_results,
                'top_suspicious_frames': top_frames
            }
        else:
            return None
    
    def _analyze_audio(self, video_path):
        """
        Extract and analyze audio from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with audio analysis results
        """
        # In a real implementation, you would:
        # 1. Extract audio from video (e.g., using ffmpeg)
        # 2. Process the audio with the audio ensemble
        
        # For demonstration, simulate this process
        audio_data = np.random.random(1000)  # Placeholder
        
        # Analyze with audio ensemble
        audio_prediction = self.audio_ensemble.predict(audio_data)
        
        return audio_prediction
    
    def _analyze_temporal_consistency(self, frame_analysis):
        """
        Analyze temporal consistency across frames.
        
        Args:
            frame_analysis: Frame analysis results
            
        Returns:
            Dict with temporal consistency analysis
        """
        if not frame_analysis or not frame_analysis['frame_results']:
            return None
            
        frame_results = frame_analysis['frame_results']
        confidences = [f['confidence'] for f in frame_results]
        
        # Check for sudden spikes in confidence
        diffs = np.diff(confidences)
        abs_diffs = np.abs(diffs)
        
        # Identify significant jumps
        threshold = np.std(confidences) * 2  # 2 standard deviations
        jump_indices = np.where(abs_diffs > threshold)[0]
        
        # Map jump indices to frame results
        jumps = []
        for idx in jump_indices:
            jumps.append({
                'start_frame': frame_results[idx],
                'end_frame': frame_results[idx + 1],
                'confidence_change': diffs[idx]
            })
        
        # Calculate overall temporal consistency score
        if len(abs_diffs) > 0:
            mean_abs_diff = np.mean(abs_diffs)
            consistency_score = 1.0 - min(mean_abs_diff, 1.0)  # Bounded between 0 and 1
        else:
            consistency_score = 1.0  # Perfect consistency if only one frame
        
        # Convert to deepfake confidence (invert consistency)
        confidence = 1.0 - consistency_score
        
        return {
            'confidence': confidence,
            'consistency_score': consistency_score,
            'significant_jumps': jumps,
            'confidence_variance': np.var(confidences),
            'confidence_trend': list(zip(
                [f['timestamp'] for f in frame_results],
                confidences
            ))
        }
    
    def _analyze_av_sync(self, frame_analysis, audio_analysis):
        """
        Analyze audio-visual synchronization.
        
        Args:
            frame_analysis: Frame analysis results
            audio_analysis: Audio analysis results
            
        Returns:
            Dict with A/V sync analysis
        """
        if not frame_analysis or not audio_analysis:
            return None
            
        # In a real implementation, analyze lip movement vs audio
        # For demonstration, return simulated result
        sync_score = 0.9  # 0 = out of sync, 1 = perfect sync
        
        # Convert to deepfake confidence (invert sync score)
        confidence = 1.0 - sync_score
        
        return {
            'confidence': confidence,
            'sync_score': sync_score,
            'sync_issues': []  # Would contain timestamps of detected issues
        }
    
    def _generate_timeline(self, frame_analysis, audio_analysis, temporal_analysis):
        """
        Generate unified timeline of deepfake indicators.
        
        Args:
            frame_analysis: Frame analysis results
            audio_analysis: Audio analysis results
            temporal_analysis: Temporal consistency analysis
            
        Returns:
            Dict with timeline data
        """
        if not frame_analysis:
            return None
            
        duration = frame_analysis['duration']
        
        # Create timeline entries
        events = []
        
        # Add frame analysis data
        if frame_analysis and 'frame_results' in frame_analysis:
            for frame in frame_analysis['frame_results']:
                events.append({
                    'timestamp': frame['timestamp'],
                    'type': 'frame',
                    'confidence': frame['confidence'],
                    'frame_idx': frame['frame_idx']
                })
        
        # Add audio anomalies
        if audio_analysis and 'anomaly_regions' in audio_analysis:
            for region in audio_analysis['anomaly_regions']:
                events.append({
                    'timestamp': region['start_time'],
                    'end_time': region['end_time'],
                    'type': 'audio',
                    'confidence': region['confidence']
                })
        
        # Add temporal jumps
        if temporal_analysis and 'significant_jumps' in temporal_analysis:
            for jump in temporal_analysis['significant_jumps']:
                events.append({
                    'timestamp': jump['start_frame']['timestamp'],
                    'end_time': jump['end_frame']['timestamp'],
                    'type': 'temporal_jump',
                    'confidence': abs(jump['confidence_change'])
                })
        
        # Sort timeline by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        return {
            'duration': duration,
            'events': events
        }
    
    def _extract_video_metadata(self, video_path):
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
            
        # Extract basic metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }
    
    def _get_visualization_data(self, predictions):
        """Implementation required by base class but not used directly."""
        return {}