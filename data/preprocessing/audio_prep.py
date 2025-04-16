"""
Audio preprocessing module for deepfake detection.
"""
import os
import logging
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from typing import Dict, Any, List, Tuple, Optional, Union
import torch

class AudioPreprocessor:
    """
    Preprocessor for audio data to prepare for deepfake detection.
    Handles operations like resampling, segmentation, and feature extraction.
    """
    
    def __init__(self, target_sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 segment_duration: float = 5.0):
        """
        Initialize the audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate for audio processing
            n_mels: Number of mel bands for spectrogram generation
            n_fft: FFT window size for spectrogram
            hop_length: Hop length for spectrogram
            segment_duration: Duration of audio segments in seconds
        """
        self.logger = logging.getLogger(__name__)
        
        # Set parameters
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        
        self.logger.info(f"AudioPreprocessor initialized (target SR: {target_sample_rate}Hz)")
    
    def preprocess(self, audio_path: str, generate_spectrogram: bool = True,
                  segment_audio: bool = True) -> Dict[str, Any]:
        """
        Preprocess an audio file for deepfake detection.
        
        Args:
            audio_path: Path to the audio file
            generate_spectrogram: Whether to generate mel spectrogram
            segment_audio: Whether to segment the audio into fixed-length chunks
            
        Returns:
            Dictionary containing:
            - waveform: Preprocessed audio waveform
            - spectrogram: Mel spectrogram (if generate_spectrogram=True)
            - segments: List of audio segments (if segment_audio=True)
            - metadata: Additional preprocessing metadata
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio cannot be processed
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Load and resample audio
            waveform, original_sr = self._load_audio(audio_path)
            
            result = {
                "waveform": waveform,
                "metadata": {
                    "original_sample_rate": original_sr,
                    "target_sample_rate": self.target_sample_rate,
                    "duration": len(waveform) / self.target_sample_rate,
                    "filename": os.path.basename(audio_path)
                }
            }
            
            # Generate mel spectrogram if requested
            if generate_spectrogram:
                spectrogram = self._generate_mel_spectrogram(waveform)
                result["spectrogram"] = spectrogram
            
            # Segment audio if requested
            if segment_audio:
                segments = self._segment_audio(waveform)
                result["segments"] = segments
                result["metadata"]["num_segments"] = len(segments)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preprocessing audio {audio_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess audio: {str(e)}")
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and resample an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple containing:
            - Resampled audio waveform as numpy array
            - Original sample rate
            
        Raises:
            ValueError: If the audio cannot be loaded
        """
        try:
            # Load audio with librosa
            waveform, original_sr = librosa.load(
                audio_path, 
                sr=self.target_sample_rate, 
                mono=True
            )
            
            # Normalize waveform
            waveform = self._normalize_waveform(waveform)
            
            return waveform, original_sr
            
        except Exception as e:
            self.logger.error(f"Error loading audio {audio_path}: {str(e)}")
            raise ValueError(f"Failed to load audio file: {str(e)}")
    
    def _normalize_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize audio waveform.
        
        Args:
            waveform: Audio waveform to normalize
            
        Returns:
            Normalized waveform
        """
        # Peak normalization
        if np.max(np.abs(waveform)) > 0:
            return waveform / np.max(np.abs(waveform))
        return waveform
    
    def _generate_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate mel spectrogram from audio waveform.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Mel spectrogram as numpy array
        """
        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.target_sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        return S_dB
    
    def _segment_audio(self, waveform: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            List of audio segments as numpy arrays
        """
        # Calculate segment length in samples
        segment_length = int(self.segment_duration * self.target_sample_rate)
        
        # Calculate number of segments
        num_segments = max(1, int(np.ceil(len(waveform) / segment_length)))
        
        segments = []
        
        for i in range(num_segments):
            # Calculate segment boundaries
            start = i * segment_length
            end = min(start + segment_length, len(waveform))
            
            # Extract segment
            segment = waveform[start:end]
            
            # Pad if segment is shorter than desired length
            if len(segment) < segment_length:
                padding = np.zeros(segment_length - len(segment))
                segment = np.concatenate([segment, padding])
            
            segments.append(segment)
        
        return segments
    
    def extract_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features from waveform.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # Mel spectrogram
        features["mel_spectrogram"] = self._generate_mel_spectrogram(waveform)
        
        # MFCC
        mfccs = librosa.feature.mfcc(
            y=waveform, 
            sr=self.target_sample_rate, 
            n_mfcc=20
        )
        features["mfccs"] = mfccs
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=waveform, 
            sr=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features["chroma"] = chroma
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=waveform, 
            sr=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features["spectral_contrast"] = contrast
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(
            y=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features["spectral_flatness"] = flatness
        
        return features
    
    def apply_augmentations(self, waveform: np.ndarray, 
                           augmentations: List[str] = None) -> np.ndarray:
        """
        Apply audio augmentations to the waveform.
        
        Args:
            waveform: Audio waveform
            augmentations: List of augmentation names to apply
                Options: 'pitch_shift', 'time_stretch', 'noise', 'reverse'
            
        Returns:
            Augmented waveform
        """
        if augmentations is None:
            return waveform
            
        result = waveform.copy()
        
        for aug in augmentations:
            if aug == 'pitch_shift':
                # Pitch shift by -2 to 2 semitones
                n_steps = np.random.uniform(-2, 2)
                result = librosa.effects.pitch_shift(
                    result, 
                    sr=self.target_sample_rate, 
                    n_steps=n_steps
                )
                
            elif aug == 'time_stretch':
                # Time stretch by factor 0.8 to 1.2
                rate = np.random.uniform(0.8, 1.2)
                result = librosa.effects.time_stretch(result, rate=rate)
                
                # Ensure same length as original
                if len(result) > len(waveform):
                    result = result[:len(waveform)]
                elif len(result) < len(waveform):
                    padding = np.zeros(len(waveform) - len(result))
                    result = np.concatenate([result, padding])
                
            elif aug == 'noise':
                # Add Gaussian noise
                noise_level = 0.005
                noise = np.random.normal(0, noise_level, result.shape)
                result = result + noise
                
                # Re-normalize
                result = self._normalize_waveform(result)
                
            elif aug == 'reverse':
                # Reverse segments of the audio
                segment_length = int(0.2 * self.target_sample_rate)  # 200ms segments
                for i in range(0, len(result), segment_length * 2):
                    end = min(i + segment_length, len(result))
                    if np.random.random() < 0.5:  # 50% chance to reverse a segment
                        result[i:end] = result[i:end][::-1]
        
        return result

def preprocess_batch(audio_paths: List[str], 
                    processor: Optional[AudioPreprocessor] = None,
                    generate_spectrogram: bool = True,
                    segment_audio: bool = True,
                    batch_size: int = 16) -> List[Dict[str, Any]]:
    """
    Preprocess a batch of audio files.
    
    Args:
        audio_paths: List of paths to audio files
        processor: Optional AudioPreprocessor instance
        generate_spectrogram: Whether to generate spectrograms
        segment_audio: Whether to segment audio files
        batch_size: Number of files to process at once
        
    Returns:
        List of preprocessing results for each audio file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing batch of {len(audio_paths)} audio files")
    
    # Create processor if not provided
    if processor is None:
        processor = AudioPreprocessor()
    
    results = []
    
    # Process in batches
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        
        for audio_path in batch:
            try:
                result = processor.preprocess(
                    audio_path, 
                    generate_spectrogram=generate_spectrogram,
                    segment_audio=segment_audio
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error preprocessing {audio_path}: {str(e)}")
                # Add empty result with error message
                results.append({
                    "error": str(e),
                    "audio_path": audio_path
                })
        
        logger.debug(f"Processed {min(i+batch_size, len(audio_paths))}/{len(audio_paths)} audio files")
    
    return results
