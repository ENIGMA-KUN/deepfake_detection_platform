import os
import numpy as np
import librosa
import torch
import logging
from typing import Dict, Tuple, List, Optional, Union

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Class for preprocessing audio data for deepfake detection."""
    
    def __init__(self, config=None):
        """
        Initialize the audio preprocessor.
        
        Args:
            config (dict, optional): Configuration parameters for preprocessing
        """
        self.config = config or {}
        
        # Target parameters for audio preprocessing
        self.target_sr = self.config.get('target_sr', 16000)  # Target sample rate (Hz)
        self.target_duration = self.config.get('target_duration', 5.0)  # Target duration (seconds)
        self.normalize = self.config.get('normalize', True)  # Whether to normalize audio
        self.n_mels = self.config.get('n_mels', 80)  # Number of mel bands
        self.n_fft = self.config.get('n_fft', 400)  # FFT window size
        self.hop_length = self.config.get('hop_length', 160)  # Hop length for STFT
        self.win_length = self.config.get('win_length', 400)  # Window length for STFT
        self.apply_augmentation = self.config.get('apply_augmentation', False)  # Apply augmentation
    
    def preprocess(self, audio_path: str, return_tensor: bool = True) -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
        """
        Preprocess an audio file for deepfake detection.
        
        Args:
            audio_path (str): Path to the audio file
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            preprocessed_audio: Preprocessed audio as tensor or NumPy array
            meta_info (dict): Additional information about preprocessing
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Store original information
            original_duration = len(y) / sr
            meta_info = {
                'original_sr': sr,
                'original_duration': original_duration,
                'preprocessing_steps': []
            }
            meta_info['preprocessing_steps'].append('load_audio')
            
            # Resample if needed
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
                meta_info['preprocessing_steps'].append('resample')
                meta_info['target_sr'] = self.target_sr
            
            # Handle audio duration
            target_length = int(self.target_duration * self.target_sr)
            
            if len(y) > target_length:
                # Take the middle segment
                start = (len(y) - target_length) // 2
                y = y[start:start + target_length]
                meta_info['preprocessing_steps'].append('trim')
            elif len(y) < target_length:
                # Pad with zeros
                pad_length = target_length - len(y)
                y = np.pad(y, (0, pad_length), 'constant')
                meta_info['preprocessing_steps'].append('pad')
            
            meta_info['final_duration'] = len(y) / self.target_sr
            
            # Normalize if requested
            if self.normalize:
                max_val = np.max(np.abs(y))
                if max_val > 0:
                    y = y / max_val
                meta_info['preprocessing_steps'].append('normalize')
            
            # Apply augmentation if requested
            if self.apply_augmentation:
                y = self._apply_augmentation(y)
                meta_info['preprocessing_steps'].append('augmentation')
            
            # Convert to tensor if requested
            if return_tensor:
                y = torch.from_numpy(y.astype(np.float32))
                meta_info['preprocessing_steps'].append('to_tensor')
            
            return y, meta_info
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to audio.
        
        Args:
            audio (np.ndarray): Audio data
            
        Returns:
            np.ndarray: Augmented audio
        """
        # Choose a random augmentation method
        augmentation_type = np.random.choice(['time_shift', 'noise', 'pitch_shift', 'time_stretch'])
        
        if augmentation_type == 'time_shift':
            # Apply time shift
            shift = np.random.randint(0, len(audio) // 10)
            return np.roll(audio, shift)
        
        elif augmentation_type == 'noise':
            # Add random noise
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, len(audio))
            return audio + noise
        
        elif augmentation_type == 'pitch_shift':
            # Apply pitch shift
            pitch_shift = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=pitch_shift)
        
        elif augmentation_type == 'time_stretch':
            # Apply time stretch and then resize back to original length
            stretch_factor = np.random.uniform(0.9, 1.1)
            audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            
            if len(audio_stretched) > len(audio):
                return audio_stretched[:len(audio)]
            else:
                return np.pad(audio_stretched, (0, max(0, len(audio) - len(audio_stretched))), 'constant')
        
        return audio
    
    def extract_features(self, audio: np.ndarray, sr: int = None) -> Dict[str, np.ndarray]:
        """
        Extract features from audio data.
        
        Args:
            audio (np.ndarray): Audio data
            sr (int, optional): Sample rate. If None, use target_sr
            
        Returns:
            dict: Dictionary of features
        """
        if sr is None:
            sr = self.target_sr
        
        features = {}
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        features['mel_spectrogram'] = librosa.power_to_db(mel_spectrogram)
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=20, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        features['mfcc'] = mfcc
        
        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        features['chroma'] = chroma
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        features['contrast'] = contrast
        
        # Tempogram (rhythm features)
        tempogram = librosa.feature.tempogram(
            y=audio, 
            sr=sr, 
            hop_length=self.hop_length
        )
        features['tempogram'] = tempogram
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        features['zcr'] = zcr
        
        return features
    
    def compute_silence_stats(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics about silence in audio.
        
        Args:
            audio (np.ndarray): Audio data
            
        Returns:
            dict: Dictionary of silence statistics
        """
        # Compute envelope
        envelope = np.abs(audio)
        
        # Threshold for silence
        threshold = 0.01  # Adjust as needed
        
        # Detect silence regions
        is_silence = envelope < threshold
        
        # Compute statistics
        total_samples = len(audio)
        silence_samples = np.sum(is_silence)
        silence_percentage = (silence_samples / total_samples) * 100
        
        # Find silence segments
        silence_changes = np.diff(np.concatenate(([0], is_silence.astype(int), [0])))
        silence_starts = np.where(silence_changes == 1)[0]
        silence_ends = np.where(silence_changes == -1)[0]
        
        # Compute segment lengths
        silence_lengths = silence_ends - silence_starts
        
        if len(silence_lengths) > 0:
            max_silence_length = np.max(silence_lengths) / self.target_sr  # in seconds
            mean_silence_length = np.mean(silence_lengths) / self.target_sr  # in seconds
        else:
            max_silence_length = 0
            mean_silence_length = 0
        
        return {
            'silence_percentage': silence_percentage,
            'max_silence_length': max_silence_length,
            'mean_silence_length': mean_silence_length,
            'num_silence_segments': len(silence_lengths)
        }
    
    def batch_preprocess(self, audio_paths: List[str], return_tensor: bool = True) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[Dict]]:
        """
        Preprocess a batch of audio files.
        
        Args:
            audio_paths (list): List of paths to audio files
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            list: List of preprocessed audio
            list: List of meta info dictionaries
        """
        preprocessed_audios = []
        meta_infos = []
        
        for audio_path in audio_paths:
            try:
                processed_audio, meta_info = self.preprocess(audio_path, return_tensor)
                preprocessed_audios.append(processed_audio)
                meta_infos.append(meta_info)
            except Exception as e:
                logger.error(f"Error preprocessing audio {audio_path}: {e}")
                # Skip failed audios
                continue
        
        return preprocessed_audios, meta_infos
    
    def create_spectrogram(self, audio_path: str, output_path: str = None) -> np.ndarray:
        """
        Create and save a spectrogram for visualization.
        
        Args:
            audio_path (str): Path to the audio file
            output_path (str, optional): Path to save the spectrogram image
            
        Returns:
            np.ndarray: Spectrogram data
        """
        # Load and preprocess audio
        y, meta_info = self.preprocess(audio_path, return_tensor=False)
        
        # Create spectrogram
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        if output_path is not None:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                S_db, 
                x_axis='time', 
                y_axis='log', 
                sr=self.target_sr, 
                hop_length=512
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram - {os.path.basename(audio_path)}')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        return S_db