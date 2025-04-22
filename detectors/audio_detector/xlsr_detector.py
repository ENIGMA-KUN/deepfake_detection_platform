"""
XLS-R-based deepfake detector for audio.
XLS-R (Cross-Lingual Speech Representations) models are powerful cross-lingual speech models
that can be used for deepfake detection across multiple languages.
"""
import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import librosa
import librosa.display
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

from detectors.base_detector import BaseDetector

class XLSRAudioDetector(BaseDetector):
    """
    XLS-R based detector for audio deepfakes.
    Uses pretrained XLS-R model with a classification head for deepfake detection.
    XLS-R models are particularly effective for multilingual audio deepfake detection.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", 
                 confidence_threshold: float = 0.5,
                 device: str = None,
                 sample_rate: int = 16000):
        """
        Initialize the XLS-R audio detector.
        
        Args:
            model_name: Name of the pretrained XLS-R model to use
            confidence_threshold: Threshold for classifying audio as deepfake
            device: Device to run the model on ('cuda' or 'cpu')
            sample_rate: Target sample rate for audio processing
        """
        super().__init__(model_name, confidence_threshold)
        
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Set sample rate
        self.sample_rate = sample_rate
        
        # Initialize model and processor as None (lazy loading)
        self.processor = None
        self.model = None
        
        # Temporal analysis parameters
        self.window_size = 4.0  # Window size in seconds
        self.hop_length = 2.0   # Hop length in seconds
        
        # XLSR specific parameters
        self.frequency_attention = True  # Enable frequency attention mechanism
        
    def load_model(self):
        """
        Load the XLS-R model and processor.
        """
        if self.model is not None:
            return
            
        try:
            self.logger.info(f"Loading XLS-R model: {self.model_name}")
            
            # Load processor for feature extraction
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            
            # Load model with classification head
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move model to the appropriate device
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("XLS-R model loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading XLS-R model: {str(e)}")
            raise RuntimeError(f"Failed to load XLS-R model: {str(e)}")
    
    def detect(self, media_path: str) -> Dict[str, Any]:
        """
        Detect if the audio is a deepfake.
        
        Args:
            media_path: Path to the audio file
            
        Returns:
            Dictionary containing detection results
            
        Raises:
            FileNotFoundError: If the audio file does not exist
            ValueError: If the file is not a valid audio
        """
        # Validate the audio file
        self._validate_media(media_path)
        
        # Load the model if not already loaded
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Load and preprocess the audio
            waveform, loaded_sample_rate = self._load_audio(media_path)
            
            # Perform temporal analysis
            temporal_scores, segments = self._temporal_analysis(waveform)
            
            # Calculate overall score
            overall_score = np.mean(temporal_scores)
            
            # Generate spectrogram features
            spectrogram = self._generate_spectrogram(waveform)
            
            # Calculate temporal inconsistency
            inconsistency = self._calculate_inconsistency(temporal_scores)
            
            # Normalize confidence score
            confidence = self.normalize_confidence(overall_score)
            
            # Determine if deepfake based on confidence threshold
            is_deepfake = confidence >= self.confidence_threshold
            
            # Calculate frequency attention features (XLSR specific)
            frequency_features = self._analyze_frequency_content(spectrogram) if self.frequency_attention else None
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "model_name": self.model_name,
                "processing_time": processing_time,
                "sample_rate": self.sample_rate,
                "temporal_analysis": {
                    "window_size": self.window_size,
                    "hop_length": self.hop_length,
                    "segment_count": len(temporal_scores),
                    "segment_scores": temporal_scores.tolist(),
                    "segment_times": segments.tolist(),
                    "inconsistency_index": inconsistency
                },
                "spectrogram": spectrogram.tolist() if spectrogram is not None else None,
                "frequency_analysis": frequency_features,
                "raw_score": overall_score
            }
            
            # Prepare result
            result = {
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "model_type": "xlsr",
                "metadata": metadata
            }
            
            self.logger.info(f"XLS-R detection completed in {processing_time:.2f}s - " +
                           f"Deepfake: {is_deepfake}, Confidence: {confidence:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in XLS-R detection: {str(e)}")
            raise
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple containing:
            - Preprocessed audio waveform
            - Original sample rate
        """
        try:
            # Load audio with librosa
            waveform, original_sample_rate = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize waveform
            waveform = librosa.util.normalize(waveform)
            
            self.logger.info(f"Loaded audio: {audio_path}, " +
                           f"Length: {len(waveform) / self.sample_rate:.2f}s, " +
                           f"Sample rate: {self.sample_rate}Hz")
                           
            return waveform, self.sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise ValueError(f"Error processing audio file {audio_path}: {str(e)}")
    
    def _temporal_analysis(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform temporal analysis on audio by processing segments.
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            Tuple containing:
            - Array of deepfake scores for each segment
            - Array of segment start times
        """
        # Convert window and hop sizes from seconds to samples
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        # Calculate number of segments
        num_segments = max(1, (len(waveform) - window_samples) // hop_samples + 1)
        
        # Initialize arrays for results
        segment_scores = np.zeros(num_segments)
        segment_times = np.zeros(num_segments)
        
        for i in range(num_segments):
            # Calculate segment boundaries
            start = i * hop_samples
            end = min(start + window_samples, len(waveform))
            
            # Extract segment
            segment = waveform[start:end]
            
            # Process segment
            if len(segment) < window_samples:
                # Pad if segment is shorter than window size
                padding = np.zeros(window_samples - len(segment))
                segment = np.concatenate([segment, padding])
            
            # Get segment score
            score = self._process_audio_segment(segment)
            
            # Store results
            segment_scores[i] = score
            segment_times[i] = start / self.sample_rate
        
        return segment_scores, segment_times
    
    def _process_audio_segment(self, segment: np.ndarray) -> float:
        """
        Process an audio segment through the XLS-R model.
        
        Args:
            segment: Audio segment as numpy array
            
        Returns:
            Deepfake confidence score (0-1)
        """
        # Prepare input features
        inputs = self.processor(
            segment, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get logits and convert to probabilities
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get the deepfake probability (assuming index 1 is "fake")
            deepfake_score = probs[0, 1].item()
        
        return deepfake_score
    
    def _generate_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate spectrogram features for visualization.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Spectrogram as numpy array
        """
        try:
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=waveform, 
                sr=self.sample_rate,
                n_mels=128,
                hop_length=512
            )
            
            # Convert to dB scale
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            return S_dB
            
        except Exception as e:
            self.logger.error(f"Error generating spectrogram: {str(e)}")
            return None
    
    def _analyze_frequency_content(self, spectrogram: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frequency content of the spectrogram.
        XLS-R specific feature for better cross-lingual deepfake detection.
        
        Args:
            spectrogram: Mel spectrogram as numpy array
            
        Returns:
            Dictionary with frequency analysis metrics
        """
        if spectrogram is None:
            return None
            
        try:
            # Calculate energy distribution across frequency bands
            frequency_energy = np.mean(spectrogram, axis=1)
            
            # Calculate energy in low, mid, and high frequency bands
            freq_bands = len(frequency_energy)
            low_band = frequency_energy[:freq_bands//3]
            mid_band = frequency_energy[freq_bands//3:2*freq_bands//3]
            high_band = frequency_energy[2*freq_bands//3:]
            
            # Calculate energy ratios
            low_energy = np.mean(low_band)
            mid_energy = np.mean(mid_band)
            high_energy = np.mean(high_band)
            total_energy = np.mean(frequency_energy)
            
            # Calculate spectral centroid
            spectral_centroid = np.sum(np.arange(freq_bands) * frequency_energy) / np.sum(frequency_energy)
            
            # Calculate harmonic ratio (approximated)
            harmonic_ratio = np.var(frequency_energy) / (np.mean(frequency_energy) + 1e-6)
            
            return {
                "low_band_energy": float(low_energy),
                "mid_band_energy": float(mid_energy),
                "high_band_energy": float(high_energy),
                "low_to_high_ratio": float(low_energy / (high_energy + 1e-6)),
                "spectral_centroid": float(spectral_centroid / freq_bands),
                "harmonic_ratio": float(harmonic_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error in frequency analysis: {str(e)}")
            return None
    
    def _calculate_inconsistency(self, scores: np.ndarray) -> float:
        """
        Calculate the temporal inconsistency index.
        Higher values indicate more inconsistency between segments,
        which is a potential indicator of deepfakes.
        
        Args:
            scores: Array of segment scores
            
        Returns:
            Inconsistency index (0-1)
        """
        if len(scores) <= 1:
            return 0.0
            
        # Calculate standard deviation of scores
        std_dev = np.std(scores)
        
        # Calculate the rate of change between adjacent segments
        diffs = np.abs(np.diff(scores))
        mean_diff = np.mean(diffs)
        
        # Combined inconsistency measure (scaled to 0-1)
        inconsistency = 0.6 * std_dev + 0.4 * mean_diff
        
        # Normalize to 0-1 range (assuming max reasonable inconsistency is 0.5)
        normalized = min(1.0, inconsistency * 2.0)
        
        return normalized
    
    def normalize_confidence(self, raw_score: float) -> float:
        """
        Normalize the raw confidence score.
        
        Args:
            raw_score: Raw score from the model
            
        Returns:
            Normalized confidence score
        """
        # XLS-R models may need slight calibration for better performance
        # Apply a slight sigmoid transformation to center the distribution
        import math
        
        # Apply slight sigmoid calibration (centers around 0.5)
        calibrated = 1.0 / (1.0 + math.exp(-10 * (raw_score - 0.5)))
        
        return calibrated
