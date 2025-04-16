#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spectrogram analyzer for audio deepfake detection.
Analyzes spectrograms to identify manipulation artifacts in audio.
"""

import logging
import os
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


class SpectrogramAnalyzer(BaseDetector):
    """Spectrogram-based analyzer for audio deepfake detection."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the spectrogram analyzer.
        
        Args:
            config: Configuration dictionary
            model_path: Not used for spectrogram analyzer (can be None)
        """
        # Spectrogram parameters
        self.sample_rate = config.get("preprocessing", {}).get("audio", {}).get("sample_rate", 16000)
        self.max_duration_seconds = config.get("preprocessing", {}).get("audio", {}).get("max_duration_seconds", 30)
        self.n_mels = config.get("n_mels", 128)
        self.n_fft = config.get("n_fft", 2048)
        self.hop_length = config.get("hop_length", 512)
        
        # Analysis parameters
        self.high_freq_threshold = config.get("high_freq_threshold", 0.7)
        self.boundary_threshold = config.get("boundary_threshold", 0.5)
        self.repetition_threshold = config.get("repetition_threshold", 0.9)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize resources for spectrogram analysis."""
        try:
            # Check if required libraries are available
            import librosa
            import scipy
            
            # Store required modules
            self.librosa = librosa
            self.scipy = scipy
            
            logger.info("Spectrogram analyzer initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("librosa and scipy are required for spectrogram analysis")
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run spectrogram analysis on the provided audio.
        
        Args:
            media_path: Path to the audio file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.AUDIO, os.path.basename(media_path))
        result.model_used = "SpectrogramAnalyzer"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid audio file"
                return result
            
            # Preprocess audio
            audio_data, spectrogram = self.preprocess(media_path)
            
            # Analyze spectrogram
            analysis_results = self._analyze_spectrogram(spectrogram, audio_data)
            
            # Postprocess results
            is_deepfake, confidence, time_segments = self.postprocess(analysis_results)
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.time_segments = time_segments
            
            # Add audio-specific features
            result.audio_features = AudioFeatures(
                spectrogram_anomalies=time_segments,
                voice_inconsistencies=[],
                synthesis_markers=analysis_results.get("synthesis_markers", [])
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
            logger.error(f"Error during spectrogram analysis: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the audio for spectrogram analysis.
        
        Args:
            media_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, spectrogram)
        """
        # Load audio
        audio_data, actual_sr = load_audio(media_path, self.sample_rate)
        
        # Trim to max duration if needed
        max_samples = int(self.max_duration_seconds * self.sample_rate)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        # Generate spectrogram
        spectrogram = self.librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        spectrogram_db = self.librosa.power_to_db(spectrogram, ref=np.max)
        
        return audio_data, spectrogram_db
    
    def _analyze_spectrogram(
        self,
        spectrogram: np.ndarray,
        audio_data: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze spectrogram for deepfake artifacts.
        
        Args:
            spectrogram: Mel spectrogram in dB scale
            audio_data: Original audio data
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "anomaly_segments": [],
            "high_frequency_content": None,
            "boundary_artifacts": [],
            "repetition_patterns": [],
            "synthesis_markers": []
        }
        
        # Normalize spectrogram for analysis
        norm_spec = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        
        # 1. Check for unusual high-frequency content
        high_freq_score = self._analyze_high_frequency_content(norm_spec)
        results["high_frequency_content"] = high_freq_score
        
        if high_freq_score > self.high_freq_threshold:
            results["synthesis_markers"].append("unusual_high_frequency_content")
        
        # 2. Check for boundary artifacts (splicing)
        boundary_artifacts = self._detect_boundary_artifacts(norm_spec)
        results["boundary_artifacts"] = boundary_artifacts
        
        if boundary_artifacts:
            results["synthesis_markers"].append("splicing_artifacts")
        
        # 3. Check for repetition patterns (common in synthetic audio)
        repetition_score, repetitions = self._detect_repetition_patterns(norm_spec)
        results["repetition_patterns"] = repetitions
        
        if repetition_score > self.repetition_threshold:
            results["synthesis_markers"].append("unnatural_repetition")
        
        # 4. Check for spectral inconsistencies
        spectral_inconsistencies = self._detect_spectral_inconsistencies(norm_spec)
        results["spectral_inconsistencies"] = spectral_inconsistencies
        
        if spectral_inconsistencies:
            results["synthesis_markers"].append("spectral_inconsistencies")
        
        # Combine all anomalies into segments
        for artifact in boundary_artifacts:
            results["anomaly_segments"].append({
                "start_time": artifact["frame"] * self.hop_length / self.sample_rate,
                "end_time": (artifact["frame"] + 1) * self.hop_length / self.sample_rate,
                "confidence": artifact["score"],
                "type": "boundary_artifact"
            })
        
        for repetition in repetitions:
            results["anomaly_segments"].append({
                "start_time": repetition["start_frame"] * self.hop_length / self.sample_rate,
                "end_time": repetition["end_frame"] * self.hop_length / self.sample_rate,
                "confidence": repetition["score"],
                "type": "repetition_pattern"
            })
        
        for inconsistency in spectral_inconsistencies:
            results["anomaly_segments"].append({
                "start_time": inconsistency["start_frame"] * self.hop_length / self.sample_rate,
                "end_time": inconsistency["end_frame"] * self.hop_length / self.sample_rate,
                "confidence": inconsistency["score"],
                "type": "spectral_inconsistency"
            })
        
        return results
    
    def _analyze_high_frequency_content(self, spectrogram: np.ndarray) -> float:
        """Analyze high frequency content for synthetic artifacts.
        
        Args:
            spectrogram: Normalized spectrogram
            
        Returns:
            Score indicating likelihood of synthetic content
        """
        # Divide spectrogram into high and low frequency bands
        high_freq_band = spectrogram[int(self.n_mels * 0.7):, :]
        low_freq_band = spectrogram[:int(self.n_mels * 0.3), :]
        
        # Calculate energy in each band
        high_energy = np.mean(high_freq_band)
        low_energy = np.mean(low_freq_band)
        
        # Calculate ratio (higher ratio indicates potential synthetic content)
        if low_energy > 0:
            ratio = high_energy / low_energy
        else:
            ratio = high_energy
        
        # Normalize to 0-1 range
        score = min(1.0, ratio / 0.5)  # 0.5 is a heuristic threshold
        
        return score
    
    def _detect_boundary_artifacts(self, spectrogram: np.ndarray) -> List[Dict[str, Any]]:
        """Detect boundary artifacts from splicing.
        
        Args:
            spectrogram: Normalized spectrogram
            
        Returns:
            List of detected boundary artifacts
        """
        artifacts = []
        
        # Calculate frame-wise derivative
        frame_diff = np.abs(np.diff(spectrogram, axis=1))
        
        # Sum over frequency axis to get boundary strength per frame
        boundary_strength = np.sum(frame_diff, axis=0)
        
        # Normalize
        if np.max(boundary_strength) > 0:
            boundary_strength = boundary_strength / np.max(boundary_strength)
        
        # Find peaks in boundary strength
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(boundary_strength, height=self.boundary_threshold, distance=20)
        
        for peak in peaks:
            artifacts.append({
                "frame": peak,
                "score": boundary_strength[peak]
            })
        
        return artifacts
    
    def _detect_repetition_patterns(self, spectrogram: np.ndarray) -> Tuple[float, List[Dict[str, Any]]]:
        """Detect unnatural repetition patterns.
        
        Args:
            spectrogram: Normalized spectrogram
            
        Returns:
            Tuple of (overall_score, list_of_repetitions)
        """
        repetitions = []
        
        # Calculate self-similarity matrix
        n_frames = spectrogram.shape[1]
        similarity_matrix = np.zeros((n_frames, n_frames))
        
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                # Cosine similarity between frames
                frame_i = spectrogram[:, i]
                frame_j = spectrogram[:, j]
                
                dot_product = np.sum(frame_i * frame_j)
                norm_i = np.sqrt(np.sum(frame_i ** 2))
                norm_j = np.sqrt(np.sum(frame_j ** 2))
                
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                else:
                    similarity = 0
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Look for diagonal patterns (indicating repeated segments)
        min_length = 20  # Minimum segment length to consider
        
        for i in range(n_frames - min_length):
            for j in range(i + min_length, n_frames - min_length):
                # Check for diagonal pattern starting at (i,j)
                diag_similarity = []
                
                for k in range(min_length):
                    if i + k < n_frames and j + k < n_frames:
                        diag_similarity.append(similarity_matrix[i + k, j + k])
                
                if diag_similarity:
                    avg_similarity = sum(diag_similarity) / len(diag_similarity)
                    
                    if avg_similarity > self.repetition_threshold:
                        repetitions.append({
                            "start_frame": i,
                            "end_frame": i + min_length,
                            "repeated_at": j,
                            "score": avg_similarity
                        })
        
        # Calculate overall repetition score
        if repetitions:
            overall_score = max(rep["score"] for rep in repetitions)
        else:
            overall_score = 0.0
        
        return overall_score, repetitions
    
    def _detect_spectral_inconsistencies(self, spectrogram: np.ndarray) -> List[Dict[str, Any]]:
        """Detect spectral inconsistencies.
        
        Args:
            spectrogram: Normalized spectrogram
            
        Returns:
            List of detected spectral inconsistencies
        """
        inconsistencies = []
        
        # Calculate spectral flux
        diff_spec = np.diff(spectrogram, axis=1)
        spectral_flux = np.sum(np.abs(diff_spec), axis=0)
        
        # Normalize
        if np.max(spectral_flux) > 0:
            spectral_flux = spectral_flux / np.max(spectral_flux)
        
        # Calculate local statistics
        window_size = 50
        padded_flux = np.pad(spectral_flux, (window_size // 2, window_size // 2), mode='edge')
        
        for i in range(len(spectral_flux)):
            # Get local window
            window = padded_flux[i:i+window_size]
            
            # Calculate local mean and std
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            # Check if current point is an outlier
            if local_std > 0:
                z_score = abs(spectral_flux[i] - local_mean) / local_std
                
                if z_score > 3.0:  # 3 sigma rule
                    # Find local segment
                    start_frame = max(0, i - 10)
                    end_frame = min(len(spectral_flux), i + 10)
                    
                    inconsistencies.append({
                        "frame": i,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "score": min(1.0, z_score / 10.0)  # Normalize score
                    })
        
        return inconsistencies
    
    def postprocess(self, analysis_results: Dict[str, Any]) -> Tuple[bool, float, List[TimeSegment]]:
        """Postprocess the analysis results to get detection results.
        
        Args:
            analysis_results: Results from spectrogram analysis
            
        Returns:
            Tuple of (is_deepfake, confidence_score, time_segments)
        """
        # Extract anomaly segments
        anomaly_segments = analysis_results.get("anomaly_segments", [])
        
        # Calculate confidence scores
        scores = []
        
        # Add high frequency content score
        high_freq_score = analysis_results.get("high_frequency_content", 0.0)
        if high_freq_score is not None:
            scores.append(high_freq_score)
        
        # Add boundary artifact scores
        boundary_artifacts = analysis_results.get("boundary_artifacts", [])
        if boundary_artifacts:
            max_boundary_score = max(artifact["score"] for artifact in boundary_artifacts)
            scores.append(max_boundary_score)
        
        # Add repetition pattern scores
        repetition_patterns = analysis_results.get("repetition_patterns", [])
        if repetition_patterns:
            max_repetition_score = max(pattern["score"] for pattern in repetition_patterns)
            scores.append(max_repetition_score)
        
        # Add spectral inconsistency scores
        spectral_inconsistencies = analysis_results.get("spectral_inconsistencies", [])
        if spectral_inconsistencies:
            max_inconsistency_score = max(inc["score"] for inc in spectral_inconsistencies)
            scores.append(max_inconsistency_score)
        
        # Add synthesis marker count score
        synthesis_markers = analysis_results.get("synthesis_markers", [])
        if synthesis_markers:
            marker_score = min(1.0, len(synthesis_markers) / 4.0)  # Max 4 markers expected
            scores.append(marker_score)
        
        # Calculate overall confidence
        if scores:
            # Use a weighted average with more weight on higher scores
            scores.sort(reverse=True)
            weights = [0.5, 0.3, 0.1, 0.05, 0.05]  # Weights for sorted scores
            
            # Pad weights if needed
            if len(scores) < len(weights):
                weights = weights[:len(scores)]
            elif len(scores) > len(weights):
                weights.extend([0.05] * (len(scores) - len(weights)))
            
            # Normalize weights
            weights = [w / sum(weights) for w in weights]
            
            # Calculate weighted score
            overall_confidence = sum(s * w for s, w in zip(scores, weights))
        else:
            overall_confidence = 0.0
        
        # Normalize confidence
        confidence = normalize_confidence_score(overall_confidence)
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        # Create time segments from anomaly segments
        time_segments = []
        
        for segment in anomaly_segments:
            time_segments.append(TimeSegment(
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                confidence=segment["confidence"],
                label=f"spectral_{segment['type']}",
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