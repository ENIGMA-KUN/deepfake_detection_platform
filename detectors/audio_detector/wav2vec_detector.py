#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wav2Vec2 based deepfake detector for audio.
Utilizes the facebook/wav2vec2-large-960h pre-trained model for detection.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (AudioFeatures, DeepfakeCategory,
                                       DetectionResult, DetectionStatus,
                                       MediaType, Region, TimeSegment)
from detectors.detector_utils import (identify_deepfake_category,
                                     load_audio, measure_execution_time,
                                     normalize_confidence_score)

# Configure logger
logger = logging.getLogger(__name__)


class Wav2VecDetector(BaseDetector):
    """Wav2Vec2 based deepfake detector for audio."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the Wav2Vec2 detector.
        
        Args:
            config: Configuration dictionary
            model_path: Path to pre-trained model (optional)
        """
        self.model_name = config.get("model_name", "facebook/wav2vec2-large-960h")
        self.device = "cuda" if config.get("use_gpu", True) else "cpu"
        
        # Initialize preprocessing parameters
        self.sample_rate = config.get("preprocessing", {}).get("audio", {}).get("sample_rate", 16000)
        self.max_duration_seconds = config.get("preprocessing", {}).get("audio", {}).get("max_duration_seconds", 30)
        
        # Additional parameters
        self.segment_size_seconds = config.get("segment_size_seconds", 1.0)
        self.stride_seconds = config.get("stride_seconds", 0.5)
        self.voice_classifier_enabled = config.get("voice_classifier_enabled", True)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the Wav2Vec2 model and resources."""
        try:
            # Import here to avoid dependency if not needed
            from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                                     Wav2Vec2ForSequenceClassification, Wav2Vec2Processor)
            import torch
            
            logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            
            # Load pre-trained model for feature extraction
            self.feature_model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real vs deepfake
                ignore_mismatched_sizes=True
            )
            
            # Initialize voice classifier for speaker consistency if enabled
            if self.voice_classifier_enabled:
                self._initialize_voice_classifier()
            
            # Move models to appropriate device
            if self.device == "cuda":
                if torch.cuda.is_available():
                    self.feature_model = self.feature_model.to(self.device)
                    if hasattr(self, 'voice_classifier'):
                        self.voice_classifier = self.voice_classifier.to(self.device)
                    logger.info("Models loaded on GPU")
                else:
                    logger.warning("CUDA requested but not available, using CPU instead")
                    self.device = "cpu"
            
            logger.info("Wav2Vec2 detector initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("The 'transformers' library is required for Wav2Vec2 detector")
            
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2 detector: {e}")
            raise RuntimeError(f"Error initializing Wav2Vec2 detector: {e}")
    
    def _initialize_voice_classifier(self) -> None:
        """Initialize the voice/speaker classifier."""
        try:
            # Import here to avoid dependency if not needed
            from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
            
            # Use a model fine-tuned for speaker recognition
            # Note: This is a placeholder - in a real implementation,
            # you would use a specifically trained speaker recognition model
            self.voice_classifier = AutoModelForAudioClassification.from_pretrained(
                "facebook/wav2vec2-base", 
                num_labels=10  # Placeholder for multiple speaker classes
            )
            
            logger.info("Voice classifier initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice classifier: {e}")
            self.voice_classifier_enabled = False
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run deepfake detection on the provided audio.
        
        Args:
            media_path: Path to the audio file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.AUDIO, os.path.basename(media_path))
        result.model_used = self.model_name
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid audio file"
                return result
            
            # Preprocess audio
            audio_data, audio_features = self.preprocess(media_path)
            
            # Run model inference on segments
            import torch
            with torch.no_grad():
                segment_results = self._analyze_segments(audio_data)
            
            # Run voice consistency check if enabled
            voice_inconsistencies = []
            if self.voice_classifier_enabled and hasattr(self, 'voice_classifier'):
                voice_inconsistencies = self._check_voice_consistency(audio_data)
            
            # Postprocess results
            is_deepfake, confidence, time_segments = self.postprocess(segment_results)
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.time_segments = time_segments
            
            # Add audio-specific features
            spectrogram_anomalies = self._find_spectrogram_anomalies(audio_features)
            
            result.audio_features = AudioFeatures(
                spectrogram_anomalies=spectrogram_anomalies,
                voice_inconsistencies=voice_inconsistencies,
                synthesis_markers=self._detect_synthesis_markers(audio_features)
            )
            
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["duration"] = len(audio_data) / self.sample_rate
            result.metadata["sample_rate"] = self.sample_rate
            
            # Identify deepfake categories
            result.categories = identify_deepfake_category(
                MediaType.AUDIO, confidence, time_segments=time_segments
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess the audio for Wav2Vec2 model.
        
        Args:
            media_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, audio_features)
        """
        # Load audio
        audio_data, actual_sr = load_audio(media_path, self.sample_rate)
        
        # Trim to max duration if needed
        max_samples = int(self.max_duration_seconds * self.sample_rate)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        # Extract features for additional analysis
        audio_features = self._extract_audio_features(audio_data)
        
        return audio_data, audio_features
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract additional features from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        try:
            import librosa
            
            # Calculate spectrogram
            spectrogram = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=self.sample_rate,
                n_mels=128
            )
            features["spectrogram"] = librosa.power_to_db(spectrogram)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=13
            )
            features["mfccs"] = mfccs
            
            # Extract pitch and harmonics
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=self.sample_rate
            )
            features["pitches"] = pitches
            features["magnitudes"] = magnitudes
            
            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features["zero_crossing_rate"] = zcr
            
        except ImportError:
            logger.warning("librosa not available, skipping advanced feature extraction")
            
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
        
        return features
    
    def _analyze_segments(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze audio in segments to detect temporal inconsistencies.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            List of results for each segment
        """
        import torch
        
        segment_size = int(self.segment_size_seconds * self.sample_rate)
        stride = int(self.stride_seconds * self.sample_rate)
        
        results = []
        
        # Process audio in segments
        for i in range(0, len(audio_data) - segment_size + 1, stride):
            segment = audio_data[i:i+segment_size]
            
            # Process segment with Wav2Vec2
            inputs = self.processor(
                segment, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            outputs = self.feature_model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Extract deepfake probability
            fake_prob = float(probs[0, 1].cpu().numpy())
            
            results.append({
                "start_time": i / self.sample_rate,
                "end_time": (i + segment_size) / self.sample_rate,
                "confidence": fake_prob
            })
        
        return results
    
    def _check_voice_consistency(self, audio_data: np.ndarray) -> List[TimeSegment]:
        """Check for speaker/voice consistency across the audio.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            List of time segments with detected inconsistencies
        """
        import torch
        
        inconsistencies = []
        
        try:
            if not hasattr(self, 'voice_classifier'):
                return inconsistencies
            
            segment_size = int(self.segment_size_seconds * self.sample_rate)
            stride = int(self.stride_seconds * self.sample_rate)
            
            # Voice embeddings for each segment
            embeddings = []
            
            # Process audio in segments
            for i in range(0, len(audio_data) - segment_size + 1, stride):
                segment = audio_data[i:i+segment_size]
                
                # Process segment
                inputs = self.processor(
                    segment, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                # Move inputs to device
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    features = self.voice_classifier.wav2vec2(
                        inputs.input_values, 
                        attention_mask=inputs.attention_mask
                    )
                    
                # Get embedding from final hidden state
                embedding = features.last_hidden_state.mean(dim=1)
                embeddings.append({
                    "start_time": i / self.sample_rate,
                    "end_time": (i + segment_size) / self.sample_rate,
                    "embedding": embedding.cpu().numpy()
                })
            
            # Compare embeddings to detect inconsistencies
            if len(embeddings) > 1:
                for i in range(len(embeddings) - 1):
                    curr_embedding = embeddings[i]["embedding"]
                    next_embedding = embeddings[i+1]["embedding"]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(curr_embedding.flatten(), next_embedding.flatten()) / (
                        np.linalg.norm(curr_embedding) * np.linalg.norm(next_embedding)
                    )
                    
                    # If similarity is low, mark as potential inconsistency
                    if similarity < 0.7:  # Threshold for voice change
                        inconsistencies.append(TimeSegment(
                            start_time=embeddings[i]["end_time"] - 0.2,
                            end_time=embeddings[i+1]["start_time"] + 0.2,
                            confidence=1.0 - similarity,
                            label="voice_change",
                            category=DeepfakeCategory.VOICE_CLONING
                        ))
        
        except Exception as e:
            logger.error(f"Error during voice consistency check: {e}")
        
        return inconsistencies
    
    def _find_spectrogram_anomalies(self, audio_features: Dict[str, Any]) -> List[TimeSegment]:
        """Analyze spectrogram to find anomalies.
        
        Args:
            audio_features: Dictionary with audio features
            
        Returns:
            List of time segments with detected anomalies
        """
        anomalies = []
        
        try:
            if "spectrogram" not in audio_features:
                return anomalies
            
            spectrogram = audio_features["spectrogram"]
            
            # Calculate frame-wise gradient
            gradient = np.abs(np.diff(spectrogram, axis=1))
            
            # Find frames with high gradient (potential splices)
            mean_gradient = np.mean(gradient, axis=0)
            threshold = np.mean(mean_gradient) + 2 * np.std(mean_gradient)
            
            # Find consecutive frames above threshold
            anomaly_frames = np.where(mean_gradient > threshold)[0]
            
            if len(anomaly_frames) > 0:
                # Group consecutive frames
                groups = []
                current_group = [anomaly_frames[0]]
                
                for i in range(1, len(anomaly_frames)):
                    if anomaly_frames[i] == anomaly_frames[i-1] + 1:
                        current_group.append(anomaly_frames[i])
                    else:
                        groups.append(current_group)
                        current_group = [anomaly_frames[i]]
                
                if current_group:
                    groups.append(current_group)
                
                # Convert frame groups to time segments
                hop_length = 512  # Assumption based on librosa defaults
                for group in groups:
                    if len(group) >= 3:  # Minimum length for consideration
                        start_time = group[0] * hop_length / self.sample_rate
                        end_time = (group[-1] + 1) * hop_length / self.sample_rate
                        
                        # Calculate confidence based on gradient magnitude
                        confidence = min(1.0, np.mean(mean_gradient[group]) / threshold)
                        
                        anomalies.append(TimeSegment(
                            start_time=start_time,
                            end_time=end_time,
                            confidence=confidence,
                            label="spectral_anomaly",
                            category=DeepfakeCategory.AUDIO_SYNTHESIS
                        ))
        
        except Exception as e:
            logger.error(f"Error during spectrogram analysis: {e}")
        
        return anomalies
    
    def _detect_synthesis_markers(self, audio_features: Dict[str, Any]) -> List[str]:
        """Detect markers of synthetic audio.
        
        Args:
            audio_features: Dictionary with audio features
            
        Returns:
            List of detected synthesis markers
        """
        markers = []
        
        try:
            # Check for unnatural zero crossing rate
            if "zero_crossing_rate" in audio_features:
                zcr = audio_features["zero_crossing_rate"]
                if np.mean(zcr) > 0.1:  # Threshold based on empirical observation
                    markers.append("high_zero_crossing_rate")
            
            # Check for pitch consistencies in speech
            if "pitches" in audio_features and "magnitudes" in audio_features:
                pitches = audio_features["pitches"]
                magnitudes = audio_features["magnitudes"]
                
                # Find strongest pitch in each frame
                strongest_pitches = []
                for i in range(pitches.shape[1]):
                    idx = np.argmax(magnitudes[:, i])
                    if magnitudes[idx, i] > 0:
                        strongest_pitches.append(pitches[idx, i])
                
                if strongest_pitches:
                    # Check for unnaturally stable pitch
                    pitch_std = np.std(strongest_pitches)
                    if pitch_std < 5.0:  # Unnaturally stable pitch
                        markers.append("unnatural_pitch_stability")
                    
                    # Check for unnatural pitch transitions
                    if len(strongest_pitches) > 1:
                        pitch_diffs = np.abs(np.diff(strongest_pitches))
                        large_jumps = np.sum(pitch_diffs > 50)  # Large pitch jumps
                        
                        if large_jumps > len(pitch_diffs) * 0.1:
                            markers.append("unnatural_pitch_transitions")
            
            # Check for artifacts in MFCCs
            if "mfccs" in audio_features:
                mfccs = audio_features["mfccs"]
                
                # Check for unusually smooth MFCCs (common in synthetic speech)
                mfcc_std = np.std(mfccs, axis=1)
                if np.mean(mfcc_std) < 0.5:  # Unnaturally smooth
                    markers.append("unnatural_smoothness")
                
                # Check for MFCC patterns consistent with vocoders
                if np.std(mfccs[0, :]) < 0.3 * np.mean(np.std(mfccs[1:, :], axis=1)):
                    markers.append("vocoder_patterns")
        
        except Exception as e:
            logger.error(f"Error during synthesis marker detection: {e}")
        
        return markers
    
    def postprocess(self, model_output: List[Dict[str, Any]]) -> Tuple[bool, float, List[TimeSegment]]:
        """Postprocess the model output to get detection results.
        
        Args:
            model_output: List of segment analysis results
            
        Returns:
            Tuple of (is_deepfake, confidence_score, time_segments)
        """
        # Extract segment confidences
        confidences = [segment["confidence"] for segment in model_output]
        
        if not confidences:
            return False, 0.0, []
        
        # Calculate overall confidence
        # Approach: Use max confidence if it's high, otherwise use average
        max_confidence = max(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Combine: weight max confidence more if it's high
        if max_confidence > 0.8:
            overall_confidence = 0.7 * max_confidence + 0.3 * avg_confidence
        else:
            overall_confidence = 0.3 * max_confidence + 0.7 * avg_confidence
        
        # Normalize confidence
        confidence = normalize_confidence_score(overall_confidence)
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        # Create time segments for detected deepfake regions
        time_segments = []
        
        if is_deepfake:
            # Group consecutive segments with high confidence
            current_segment = None
            
            for segment in model_output:
                segment_confidence = segment["confidence"]
                
                if segment_confidence >= self.confidence_threshold:
                    if current_segment is None:
                        # Start a new segment
                        current_segment = {
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                            "confidences": [segment_confidence]
                        }
                    else:
                        # Extend current segment
                        current_segment["end_time"] = segment["end_time"]
                        current_segment["confidences"].append(segment_confidence)
                else:
                    if current_segment is not None:
                        # Finish current segment and add to list
                        time_segments.append(TimeSegment(
                            start_time=current_segment["start_time"],
                            end_time=current_segment["end_time"],
                            confidence=sum(current_segment["confidences"]) / len(current_segment["confidences"]),
                            label="deepfake_segment",
                            category=DeepfakeCategory.AUDIO_SYNTHESIS
                        ))
                        current_segment = None
            
            # Add final segment if exists
            if current_segment is not None:
                time_segments.append(TimeSegment(
                    start_time=current_segment["start_time"],
                    end_time=current_segment["end_time"],
                    confidence=sum(current_segment["confidences"]) / len(current_segment["confidences"]),
                    label="deepfake_segment",
                    category=DeepfakeCategory.AUDIO_SYNTHESIS
                ))
            
            # If no segments were created but overall is deepfake,
            # create a segment for the entire audio
            if not time_segments:
                time_segments.append(TimeSegment(
                    start_time=model_output[0]["start_time"],
                    end_time=model_output[-1]["end_time"],
                    confidence=confidence,
                    label="deepfake_audio",
                    category=DeepfakeCategory.AUDIO_SYNTHESIS
                ))
        
        return is_deepfake, confidence, time_segments
    
    def validate_media(self, media_path: str) -> bool:
        """Validate if the media file is a supported audio.
        
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
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        file_ext = os.path.splitext(media_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Unsupported audio format: {file_ext}")
            return False
        
        # Try loading the audio to ensure it's valid
        try:
            audio_data, _ = load_audio(media_path, self.sample_rate)
            
            # Check if audio has content
            if len(audio_data) == 0:
                logger.error(f"Empty audio file: {media_path}")
                return False
                
        except Exception as e:
            logger.error(f"Invalid audio file: {e}")
            return False
        
        return True
    
    def get_media_type(self) -> MediaType:
        """Get the media type supported by this detector.
        
        Returns:
            MediaType.AUDIO
        """
        return MediaType.AUDIO