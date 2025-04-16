import os
import logging
import mimetypes
from typing import Dict, List, Union, Tuple, Any, Optional
import numpy as np
import torch

from data.preprocessing.image_prep import ImagePreprocessor
from data.preprocessing.audio_prep import AudioPreprocessor
from data.preprocessing.video_prep import VideoPreprocessor
from data.augmentation.augmenters import AugmentationPipeline

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Unified preprocessing pipeline for all media types."""
    
    def __init__(self, config=None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config (dict, optional): Configuration parameters for preprocessing
        """
        self.config = config or {}
        
        # Initialize individual preprocessors
        self.image_preprocessor = ImagePreprocessor(self.config.get('image_config', {}))
        self.audio_preprocessor = AudioPreprocessor(self.config.get('audio_config', {}))
        self.video_preprocessor = VideoPreprocessor(self.config.get('video_config', {}))
        
        # Initialize augmentation pipeline if enabled
        self.apply_augmentation = self.config.get('apply_augmentation', False)
        if self.apply_augmentation:
            self.augmenter = AugmentationPipeline(self.config.get('augmentation_config', {}))
        
        # MIME type mappings
        self.image_mime_types = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
        self.audio_mime_types = ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/x-flac']
        self.video_mime_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo']
        
        # File extension mappings (as fallback)
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        self.audio_extensions = ['.wav', '.mp3', '.flac']
        self.video_extensions = ['.mp4', '.avi', '.mov', '.quicktime']
        
        # Cache for preprocessed data
        self.cache_enabled = self.config.get('enable_cache', False)
        self.cache = {}
    
    def determine_media_type(self, file_path: str) -> str:
        """
        Determine the media type from a file path.
        
        Args:
            file_path (str): Path to the media file
            
        Returns:
            str: Media type ('image', 'audio', 'video', or 'unknown')
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try to determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type in self.image_mime_types:
            return 'image'
        elif mime_type in self.audio_mime_types:
            return 'audio'
        elif mime_type in self.video_mime_types:
            return 'video'
        
        # Fallback to file extension
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in self.image_extensions:
            return 'image'
        elif ext in self.audio_extensions:
            return 'audio'
        elif ext in self.video_extensions:
            return 'video'
        
        # Unknown type
        logger.warning(f"Unknown media type for file: {file_path}")
        return 'unknown'
    
    def preprocess(self, file_path: str, media_type: str = None, return_tensor: bool = True) -> Tuple[Any, Dict]:
        """
        Preprocess a media file.
        
        Args:
            file_path (str): Path to the media file
            media_type (str, optional): Media type override ('image', 'audio', 'video')
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            Preprocessed media data
            dict: Meta information about preprocessing
        """
        # Check cache if enabled
        cache_key = f"{file_path}_{media_type}_{return_tensor}"
        if self.cache_enabled and cache_key in self.cache:
            logger.debug(f"Using cached preprocessed data for {file_path}")
            return self.cache[cache_key]
        
        # Determine media type if not provided
        if media_type is None:
            media_type = self.determine_media_type(file_path)
        
        # Preprocess based on media type
        if media_type == 'image':
            preprocessed_data, meta_info = self.image_preprocessor.preprocess(file_path, return_tensor)
        elif media_type == 'audio':
            preprocessed_data, meta_info = self.audio_preprocessor.preprocess(file_path, return_tensor)
        elif media_type == 'video':
            preprocessed_data, meta_info = self.video_preprocessor.preprocess(file_path, return_tensor)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Apply augmentation if enabled
        if self.apply_augmentation:
            if media_type == 'image':
                preprocessed_data, aug_info = self.augmenter.augment_image(preprocessed_data)
            elif media_type == 'audio':
                # Need sample rate for audio augmentation
                sr = meta_info.get('original_sr', 16000)
                preprocessed_data, aug_info = self.augmenter.augment_audio(preprocessed_data, sr)
            elif media_type == 'video':
                preprocessed_data, aug_info = self.augmenter.augment_video(preprocessed_data)
            
            # Update meta info with augmentation details
            meta_info['augmentation'] = aug_info
        
        # Add general info
        meta_info['media_type'] = media_type
        meta_info['file_path'] = file_path
        
        # Cache result if enabled
        if self.cache_enabled:
            self.cache[cache_key] = (preprocessed_data, meta_info)
        
        return preprocessed_data, meta_info
    
    def batch_preprocess(self, file_paths: List[str], media_types: List[str] = None, 
                         return_tensor: bool = True) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        Preprocess a batch of media files.
        
        Args:
            file_paths (list): List of paths to media files
            media_types (list, optional): List of media types for each file
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            dict: Dictionary of preprocessed media data grouped by media type
            dict: Dictionary of meta information grouped by media type
        """
        # Determine media types if not provided
        if media_types is None:
            media_types = [self.determine_media_type(path) for path in file_paths]
        
        # Group files by media type
        image_paths = []
        audio_paths = []
        video_paths = []
        
        for path, media_type in zip(file_paths, media_types):
            if media_type == 'image':
                image_paths.append(path)
            elif media_type == 'audio':
                audio_paths.append(path)
            elif media_type == 'video':
                video_paths.append(path)
            else:
                logger.warning(f"Skipping file with unsupported media type: {path}")
        
        # Preprocess each media type in batch
        preprocessed_data = {}
        meta_info = {}
        
        # Process images
        if image_paths:
            image_data, image_info = self.image_preprocessor.batch_preprocess(image_paths, return_tensor)
            preprocessed_data['images'] = image_data
            meta_info['images'] = image_info
        
        # Process audio
        if audio_paths:
            audio_data, audio_info = self.audio_preprocessor.batch_preprocess(audio_paths, return_tensor)
            preprocessed_data['audio'] = audio_data
            meta_info['audio'] = audio_info
        
        # Process video
        if video_paths:
            video_data, video_info = self.video_preprocessor.batch_preprocess(video_paths, return_tensor)
            preprocessed_data['video'] = video_data
            meta_info['video'] = video_info
        
        # Apply augmentation if enabled
        if self.apply_augmentation:
            # Create a batch with all media types
            augmentation_batch = {}
            
            if 'images' in preprocessed_data:
                augmentation_batch['images'] = preprocessed_data['images']
            
            if 'audio' in preprocessed_data:
                augmentation_batch['audio'] = preprocessed_data['audio']
                # Add sample rate if available from meta info
                if meta_info['audio'] and 'original_sr' in meta_info['audio'][0]:
                    augmentation_batch['sr'] = meta_info['audio'][0]['original_sr']
            
            if 'video' in preprocessed_data:
                augmentation_batch['video'] = preprocessed_data['video']
            
            # Apply batch augmentation
            augmented_batch = self.augmenter.augment_batch(augmentation_batch)
            
            # Update preprocessed data and meta info
            for key in ['images', 'audio', 'video']:
                if key in augmented_batch:
                    preprocessed_data[key] = augmented_batch[key]
                    
                    if key in meta_info:
                        for i, info in enumerate(meta_info[key]):
                            if i < len(augmented_batch['augmentation_info'].get(key, [])):
                                info['augmentation'] = augmented_batch['augmentation_info'][key][i]
        
        return preprocessed_data, meta_info
    
    def extract_features(self, preprocessed_data: Any, media_type: str, meta_info: Dict = None) -> Dict:
        """
        Extract features from preprocessed media data.
        
        Args:
            preprocessed_data: Preprocessed media data
            media_type (str): Media type ('image', 'audio', 'video')
            meta_info (dict, optional): Meta information from preprocessing
            
        Returns:
            dict: Extracted features
        """
        if media_type == 'audio':
            # For audio, we need the sample rate
            sr = meta_info.get('original_sr', 16000) if meta_info else 16000
            
            # Convert tensor to numpy if needed
            if isinstance(preprocessed_data, torch.Tensor):
                preprocessed_data = preprocessed_data.cpu().numpy()
            
            return self.audio_preprocessor.extract_features(preprocessed_data, sr)
        
        elif media_type == 'video':
            # For video, extract motion metrics
            return self.video_preprocessor.compute_motion_metrics(preprocessed_data)
        
        elif media_type == 'image':
            # For image, we don't have a specific feature extraction method
            # but could add one in the future if needed
            logger.warning("Feature extraction not implemented for image media type")
            return {}
        
        else:
            raise ValueError(f"Unsupported media type for feature extraction: {media_type}")
    
    def clear_cache(self):
        """Clear the preprocessing cache."""
        self.cache = {}
        logger.debug("Preprocessing cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the number of items in the preprocessing cache."""
        return len(self.cache)
    
    def create_preprocessing_config(self) -> Dict:
        """
        Create a configuration dictionary for preprocessing.
        
        Returns:
            dict: Configuration dictionary with all settings
        """
        config = {
            'image_config': self.image_preprocessor.config,
            'audio_config': self.audio_preprocessor.config,
            'video_config': self.video_preprocessor.config,
            'apply_augmentation': self.apply_augmentation,
            'enable_cache': self.cache_enabled
        }
        
        if self.apply_augmentation:
            config['augmentation_config'] = self.augmenter.config
        
        return config