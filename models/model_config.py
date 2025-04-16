"""
Model configuration system for the Deepfake Detection Platform.

This module provides classes and utilities for configuring detector models,
including parameter management, validation, and persistence.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, TypeVar

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for config classes
T = TypeVar('T', bound='BaseModelConfig')

class ConfigValidationError(Exception):
    """Exception raised for model configuration validation errors."""
    pass

class BaseModelConfig:
    """
    Base class for model configurations.
    
    This class provides common utilities for model configuration including
    validation, serialization, and persistence.
    """
    
    # Default configuration directory
    DEFAULT_CONFIG_DIR = "models/configs"
    
    # Required configuration fields
    REQUIRED_FIELDS = []
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize model configuration.
        
        Args:
            config_dict: Dictionary with configuration values (optional)
        """
        # Initialize with empty dict if not provided
        self._config = config_dict if config_dict is not None else {}
        
        # Initialize with default values
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        # By default this does nothing
        # Override in subclasses to set defaults
        pass
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            bool: True if valid, otherwise raises ConfigValidationError
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Check required fields
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in self._config]
        if missing_fields:
            raise ConfigValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validation successful
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        self._config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self._config.copy()
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration instance
        """
        config = cls(config_dict)
        config.validate()
        return config
    
    def save(self, filename: str, config_dir: str = None) -> str:
        """
        Save configuration to file.
        
        Args:
            filename: Configuration filename
            config_dir: Configuration directory (optional)
            
        Returns:
            Path to saved configuration file
        """
        # Validate configuration before saving
        self.validate()
        
        # Determine target directory
        if config_dir is None:
            config_dir = self.DEFAULT_CONFIG_DIR
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Determine file path
        if not filename.endswith(('.json', '.yaml', '.yml')):
            filename += '.json'  # Default to JSON
        
        file_path = os.path.join(config_dir, filename)
        
        # Determine format from extension
        ext = os.path.splitext(filename)[1].lower()
        
        # Save configuration
        if ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        elif ext in ('.yaml', '.yml'):
            with open(file_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
        
        logger.info(f"Configuration saved to {file_path}")
        return file_path
    
    @classmethod
    def load(cls: Type[T], filename: str, config_dir: str = None) -> T:
        """
        Load configuration from file.
        
        Args:
            filename: Configuration filename
            config_dir: Configuration directory (optional)
            
        Returns:
            Configuration instance
            
        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If format is unsupported
        """
        # Determine target directory
        if config_dir is None:
            config_dir = cls.DEFAULT_CONFIG_DIR
        
        # Determine file path
        if not os.path.isabs(filename):
            file_path = os.path.join(config_dir, filename)
        else:
            file_path = filename
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Determine format from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Load configuration
        if ext == '.json':
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        elif ext in ('.yaml', '.yml'):
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
        
        # Create and validate configuration
        config = cls(config_dict)
        config.validate()
        
        logger.info(f"Configuration loaded from {file_path}")
        return config

class ImageModelConfig(BaseModelConfig):
    """
    Configuration for image detection models.
    
    This class extends BaseModelConfig with image-specific configuration.
    """
    
    # Required configuration fields
    REQUIRED_FIELDS = ['model_name', 'model_type']
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            'model_type': 'vit',
            'model_name': 'google/vit-base-patch16-224',
            'image_size': 224,
            'batch_size': 8,
            'use_face_detection': True,
            'use_ela': True,
            'confidence_threshold': 0.5,
            'cache_dir': 'models/cache',
            'device': 'cuda',  # Options: 'cpu', 'cuda', 'auto'
            'ensemble_weights': {
                'vit': 0.7,
                'ela': 0.2,
                'face': 0.1
            }
        }
        
        # Apply defaults for missing values
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def validate(self) -> bool:
        """
        Validate the image model configuration.
        
        Returns:
            bool: True if valid, otherwise raises ConfigValidationError
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Call base validation
        super().validate()
        
        # Validate model type
        valid_model_types = ['vit', 'efficientnet', 'resnet', 'custom']
        model_type = self._config.get('model_type')
        if model_type not in valid_model_types:
            raise ConfigValidationError(
                f"Invalid model_type: {model_type}. "
                f"Must be one of: {', '.join(valid_model_types)}"
            )
        
        # Validate image size
        image_size = self._config.get('image_size')
        if not isinstance(image_size, int) or image_size < 32 or image_size > 4096:
            raise ConfigValidationError(
                f"Invalid image_size: {image_size}. "
                f"Must be an integer between 32 and 4096."
            )
        
        # Validate batch size
        batch_size = self._config.get('batch_size')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigValidationError(
                f"Invalid batch_size: {batch_size}. "
                f"Must be a positive integer."
            )
        
        # Validate confidence threshold
        confidence_threshold = self._config.get('confidence_threshold')
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            raise ConfigValidationError(
                f"Invalid confidence_threshold: {confidence_threshold}. "
                f"Must be a float between 0 and 1."
            )
        
        # Validate device
        device = self._config.get('device')
        valid_devices = ['cpu', 'cuda', 'auto']
        if device not in valid_devices:
            raise ConfigValidationError(
                f"Invalid device: {device}. "
                f"Must be one of: {', '.join(valid_devices)}"
            )
        
        # Validate ensemble weights
        ensemble_weights = self._config.get('ensemble_weights', {})
        if not isinstance(ensemble_weights, dict):
            raise ConfigValidationError(
                f"Invalid ensemble_weights: {ensemble_weights}. "
                f"Must be a dictionary."
            )
        
        # Check ensemble weights sum to approximately 1
        weights_sum = sum(ensemble_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            raise ConfigValidationError(
                f"Invalid ensemble_weights: {ensemble_weights}. "
                f"Weights must sum to 1.0 (current sum: {weights_sum:.2f})."
            )
        
        # Validation successful
        return True

class AudioModelConfig(BaseModelConfig):
    """
    Configuration for audio detection models.
    
    This class extends BaseModelConfig with audio-specific configuration.
    """
    
    # Required configuration fields
    REQUIRED_FIELDS = ['model_name', 'model_type']
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            'model_type': 'wav2vec2',
            'model_name': 'facebook/wav2vec2-large-960h',
            'sample_rate': 16000,
            'max_length': 10,  # seconds
            'batch_size': 1,
            'use_spectrogram': True,
            'confidence_threshold': 0.5,
            'cache_dir': 'models/cache',
            'device': 'cuda',  # Options: 'cpu', 'cuda', 'auto'
            'ensemble_weights': {
                'wav2vec2': 0.6,
                'spectrogram': 0.4
            }
        }
        
        # Apply defaults for missing values
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def validate(self) -> bool:
        """
        Validate the audio model configuration.
        
        Returns:
            bool: True if valid, otherwise raises ConfigValidationError
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Call base validation
        super().validate()
        
        # Validate model type
        valid_model_types = ['wav2vec2', 'melspectrogram', 'custom']
        model_type = self._config.get('model_type')
        if model_type not in valid_model_types:
            raise ConfigValidationError(
                f"Invalid model_type: {model_type}. "
                f"Must be one of: {', '.join(valid_model_types)}"
            )
        
        # Validate sample rate
        sample_rate = self._config.get('sample_rate')
        valid_sample_rates = [8000, 16000, 22050, 44100, 48000]
        if sample_rate not in valid_sample_rates:
            raise ConfigValidationError(
                f"Invalid sample_rate: {sample_rate}. "
                f"Common rates are: {', '.join(map(str, valid_sample_rates))}"
            )
        
        # Validate max length
        max_length = self._config.get('max_length')
        if not isinstance(max_length, (int, float)) or max_length <= 0 or max_length > 60:
            raise ConfigValidationError(
                f"Invalid max_length: {max_length}. "
                f"Must be a positive number up to 60 seconds."
            )
        
        # Validate confidence threshold
        confidence_threshold = self._config.get('confidence_threshold')
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            raise ConfigValidationError(
                f"Invalid confidence_threshold: {confidence_threshold}. "
                f"Must be a float between 0 and 1."
            )
        
        # Validate device
        device = self._config.get('device')
        valid_devices = ['cpu', 'cuda', 'auto']
        if device not in valid_devices:
            raise ConfigValidationError(
                f"Invalid device: {device}. "
                f"Must be one of: {', '.join(valid_devices)}"
            )
        
        # Validate ensemble weights
        ensemble_weights = self._config.get('ensemble_weights', {})
        if not isinstance(ensemble_weights, dict):
            raise ConfigValidationError(
                f"Invalid ensemble_weights: {ensemble_weights}. "
                f"Must be a dictionary."
            )
        
        # Check ensemble weights sum to approximately 1
        weights_sum = sum(ensemble_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            raise ConfigValidationError(
                f"Invalid ensemble_weights: {ensemble_weights}. "
                f"Weights must sum to 1.0 (current sum: {weights_sum:.2f})."
            )
        
        # Validation successful
        return True

class VideoModelConfig(BaseModelConfig):
    """
    Configuration for video detection models.
    
    This class extends BaseModelConfig with video-specific configuration.
    """
    
    # Required configuration fields
    REQUIRED_FIELDS = ['model_type']
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            'model_type': 'hybrid',  # Options: 'timesformer', 'genconvit', 'hybrid'
            'frame_model_name': 'google/vit-base-patch16-224',
            'frame_sample_rate': 1,  # Process every n-th frame
            'max_frames': 30,  # Maximum frames to process
            'batch_size': 4,
            'use_audio_sync': True,
            'confidence_threshold': 0.5,
            'cache_dir': 'models/cache',
            'device': 'cuda',  # Options: 'cpu', 'cuda', 'auto'
            'ensemble_weights': {
                'frame': 0.5,
                'temporal': 0.3,
                'audio_sync': 0.2
            }
        }
        
        # Apply defaults for missing values
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def validate(self) -> bool:
        """
        Validate the video model configuration.
        
        Returns:
            bool: True if valid, otherwise raises ConfigValidationError
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Call base validation
        super().validate()
        
        # Validate model type
        valid_model_types = ['timesformer', 'genconvit', 'hybrid', 'custom']
        model_type = self._config.get('model_type')
        if model_type not in valid_model_types:
            raise ConfigValidationError(
                f"Invalid model_type: {model_type}. "
                f"Must be one of: {', '.join(valid_model_types)}"
            )
        
        # Validate frame sample rate
        frame_sample_rate = self._config.get('frame_sample_rate')
        if not isinstance(frame_sample_rate, int) or frame_sample_rate < 1:
            raise ConfigValidationError(
                f"Invalid frame_sample_rate: {frame_sample_rate}. "
                f"Must be a positive integer."
            )
        
        # Validate max frames
        max_frames = self._config.get('max_frames')
        if not isinstance(max_frames, int) or max_frames < 1:
            raise ConfigValidationError(
                f"Invalid max_frames: {max_frames}. "
                f"Must be a positive integer."
            )
        
        # Validate batch size
        batch_size = self._config.get('batch_size')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigValidationError(
                f"Invalid batch_size: {batch_size}. "
                f"Must be a positive integer."
            )
        
        # Validate confidence threshold
        confidence_threshold = self._config.get('confidence_threshold')
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            raise ConfigValidationError(
                f"Invalid confidence_threshold: {confidence_threshold}. "
                f"Must be a float between 0 and 1."
            )
        
        # Validate device
        device = self._config.get('device')
        valid_devices = ['cpu', 'cuda', 'auto']
        if device not in valid_devices:
            raise ConfigValidationError(
                f"Invalid device: {device}. "
                f"Must be one of: {', '.join(valid_devices)}"
            )
        
        # Validate ensemble weights
        ensemble_weights = self._config.get('ensemble_weights', {})
        if not isinstance(ensemble_weights, dict):
            raise ConfigValidationError(
                f"Invalid ensemble_weights: {ensemble_weights}. "
                f"Must be a dictionary."
            )
        
        # Check ensemble weights sum to approximately 1
        weights_sum = sum(ensemble_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            raise ConfigValidationError(
                f"Invalid ensemble_weights: {ensemble_weights}. "
                f"Weights must sum to 1.0 (current sum: {weights_sum:.2f})."
            )
        
        # Validation successful
        return True

class ModelConfigManager:
    """
    Manager for model configurations.
    
    This class provides utilities for working with multiple model
    configurations, including loading, saving, and validation.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the model configuration manager.
        
        Args:
            config_dir: Configuration directory (optional)
        """
        # Set configuration directory
        self.config_dir = config_dir or BaseModelConfig.DEFAULT_CONFIG_DIR
        
        # Create directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize configuration caches
        self.image_configs = {}
        self.audio_configs = {}
        self.video_configs = {}
    
    def get_image_config(self, name: str = 'default') -> ImageModelConfig:
        """
        Get image model configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Image model configuration
        """
        # Check if already loaded
        if name in self.image_configs:
            return self.image_configs[name]
        
        # Try to load
        try:
            filename = f"image_{name}.json"
            config = ImageModelConfig.load(filename, self.config_dir)
            self.image_configs[name] = config
            return config
        except FileNotFoundError:
            # Create default configuration
            config = ImageModelConfig()
            self.image_configs[name] = config
            
            # Save default configuration
            config.save(f"image_{name}.json", self.config_dir)
            
            return config
    
    def get_audio_config(self, name: str = 'default') -> AudioModelConfig:
        """
        Get audio model configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Audio model configuration
        """
        # Check if already loaded
        if name in self.audio_configs:
            return self.audio_configs[name]
        
        # Try to load
        try:
            filename = f"audio_{name}.json"
            config = AudioModelConfig.load(filename, self.config_dir)
            self.audio_configs[name] = config
            return config
        except FileNotFoundError:
            # Create default configuration
            config = AudioModelConfig()
            self.audio_configs[name] = config
            
            # Save default configuration
            config.save(f"audio_{name}.json", self.config_dir)
            
            return config
    
    def get_video_config(self, name: str = 'default') -> VideoModelConfig:
        """
        Get video model configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Video model configuration
        """
        # Check if already loaded
        if name in self.video_configs:
            return self.video_configs[name]
        
        # Try to load
        try:
            filename = f"video_{name}.json"
            config = VideoModelConfig.load(filename, self.config_dir)
            self.video_configs[name] = config
            return config
        except FileNotFoundError:
            # Create default configuration
            config = VideoModelConfig()
            self.video_configs[name] = config
            
            # Save default configuration
            config.save(f"video_{name}.json", self.config_dir)
            
            return config
    
    def save_image_config(self, config: ImageModelConfig, name: str = 'default') -> str:
        """
        Save image model configuration.
        
        Args:
            config: Image model configuration
            name: Configuration name
            
        Returns:
            Path to saved configuration file
        """
        filename = f"image_{name}.json"
        file_path = config.save(filename, self.config_dir)
        self.image_configs[name] = config
        return file_path
    
    def save_audio_config(self, config: AudioModelConfig, name: str = 'default') -> str:
        """
        Save audio model configuration.
        
        Args:
            config: Audio model configuration
            name: Configuration name
            
        Returns:
            Path to saved configuration file
        """
        filename = f"audio_{name}.json"
        file_path = config.save(filename, self.config_dir)
        self.audio_configs[name] = config
        return file_path
    
    def save_video_config(self, config: VideoModelConfig, name: str = 'default') -> str:
        """
        Save video model configuration.
        
        Args:
            config: Video model configuration
            name: Configuration name
            
        Returns:
            Path to saved configuration file
        """
        filename = f"video_{name}.json"
        file_path = config.save(filename, self.config_dir)
        self.video_configs[name] = config
        return file_path
    
    def list_configs(self) -> Dict[str, List[str]]:
        """
        List available configurations.
        
        Returns:
            Dictionary of configuration names by type
        """
        result = {
            'image': [],
            'audio': [],
            'video': []
        }
        
        # Scan configuration directory
        for file in os.listdir(self.config_dir):
            if not file.endswith(('.json', '.yaml', '.yml')):
                continue
            
            # Check prefix
            if file.startswith('image_'):
                name = os.path.splitext(file)[0][6:]  # Remove 'image_' prefix
                result['image'].append(name)
            elif file.startswith('audio_'):
                name = os.path.splitext(file)[0][6:]  # Remove 'audio_' prefix
                result['audio'].append(name)
            elif file.startswith('video_'):
                name = os.path.splitext(file)[0][6:]  # Remove 'video_' prefix
                result['video'].append(name)
        
        return result
    
    def delete_config(self, config_type: str, name: str) -> bool:
        """
        Delete a configuration.
        
        Args:
            config_type: Configuration type ('image', 'audio', 'video')
            name: Configuration name
            
        Returns:
            True if deleted, False if not found
        """
        # Construct filename
        filename = f"{config_type}_{name}.json"
        file_path = os.path.join(self.config_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False
        
        # Remove from cache
        if config_type == 'image' and name in self.image_configs:
            del self.image_configs[name]
        elif config_type == 'audio' and name in self.audio_configs:
            del self.audio_configs[name]
        elif config_type == 'video' and name in self.video_configs:
            del self.video_configs[name]
        
        # Delete file
        os.remove(file_path)
        return True

def create_default_configs():
    """Create default model configurations if they don't exist."""
    # Create configuration manager
    config_manager = ModelConfigManager()
    
    # Get default configurations (creates if not exist)
    config_manager.get_image_config()
    config_manager.get_audio_config()
    config_manager.get_video_config()
    
    logger.info("Default model configurations created")