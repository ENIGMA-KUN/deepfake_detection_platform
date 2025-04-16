"""
Model loader for the Deepfake Detection Platform.

This module provides utilities for loading and managing pre-trained models
from Hugging Face and other sources, with support for caching and version
management.
"""

import os
import json
import hashlib
import logging
import time
import torch
import requests
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from urllib.parse import urlparse

# Hugging Face transformers imports
try:
    from transformers import AutoModel, AutoFeatureExtractor, AutoProcessor
    from transformers import AutoImageProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not available. Some models may not load.")

# Specialized model imports
try:
    from transformers import ViTForImageClassification, Wav2Vec2ForCTC
    SPECIALIZED_MODELS_AVAILABLE = True
except ImportError:
    SPECIALIZED_MODELS_AVAILABLE = False
    print("Warning: specialized models not available. Some functionalities may be limited.")

# Local imports
from models.model_config import (
    BaseModelConfig, ImageModelConfig, AudioModelConfig, VideoModelConfig,
    ModelConfigManager
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Exception raised for model loading errors."""
    pass

class ModelVersionError(Exception):
    """Exception raised for model version errors."""
    pass

class ModelLoader:
    """
    Utility class for loading machine learning models.
    
    This class provides functionality for loading models from various sources,
    including Hugging Face Hub and local files, with caching and version
    management.
    """
    
    def __init__(self, cache_dir: str = 'models/cache'):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory for caching models
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache for loaded models
        self.model_cache = {}
        
        # Initialize config manager
        self.config_manager = ModelConfigManager()
        
        # Check device availability
        self.device = self._get_default_device()
        
        logger.info(f"Model loader initialized with cache directory: {self.cache_dir}")
        logger.info(f"Using device: {self.device}")
    
    def _get_default_device(self) -> str:
        """
        Get the default device for model inference.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def get_device(self, requested_device: str = 'auto') -> torch.device:
        """
        Get torch device based on requested device.
        
        Args:
            requested_device: Requested device ('cpu', 'cuda', 'auto')
            
        Returns:
            torch.device instance
        """
        if requested_device == 'auto':
            # Use the default device
            return torch.device(self.device)
        elif requested_device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _cache_key(self, model_name: str, model_type: str) -> str:
        """
        Generate a cache key for a model.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of the model
            
        Returns:
            Cache key string
        """
        # Create a cache key from model name and type
        key_str = f"{model_name}_{model_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, model_name: str, model_type: str) -> Optional[Any]:
        """
        Get a model from the cache if available.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of the model
            
        Returns:
            Cached model or None if not found
        """
        cache_key = self._cache_key(model_name, model_type)
        return self.model_cache.get(cache_key)
    
    def _add_to_cache(self, model_name: str, model_type: str, model: Any) -> None:
        """
        Add a model to the cache.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of the model
            model: Model to cache
        """
        cache_key = self._cache_key(model_name, model_type)
        self.model_cache[cache_key] = model
        logger.debug(f"Model {model_name} ({model_type}) added to cache with key {cache_key}")
    
    def _clear_cache(self) -> None:
        """Clear the model cache."""
        self.model_cache = {}
        logger.info("Model cache cleared")
    
    def _is_huggingface_model(self, model_name: str) -> bool:
        """
        Check if a model name refers to a Hugging Face model.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            True if model is from Hugging Face, False otherwise
        """
        # Check if model name contains a slash (organization/model pattern)
        if '/' in model_name:
            # Check if it's a URL
            parsed_url = urlparse(model_name)
            if parsed_url.scheme:
                # It's a URL, check if it's from huggingface.co
                return 'huggingface.co' in parsed_url.netloc
            else:
                # It's an organization/model pattern, assume it's from Hugging Face
                return True
        
        # If it's just a model name without organization, assume it's from Hugging Face
        return True
    
    def _check_model_compatibility(self, model: Any, model_type: str) -> bool:
        """
        Check if a model is compatible with the expected type.
        
        Args:
            model: Loaded model
            model_type: Expected model type
            
        Returns:
            True if compatible, False otherwise
        """
        # Placeholder for model compatibility checking
        # In a real implementation, this would do more sophisticated checks
        if model_type == 'vit':
            return hasattr(model, 'forward') and hasattr(model, 'config')
        elif model_type == 'wav2vec2':
            return hasattr(model, 'forward') and hasattr(model, 'config')
        elif model_type in ['timesformer', 'genconvit']:
            return hasattr(model, 'forward') and hasattr(model, 'config')
        
        # Default to True for unknown model types
        return True
    
    def load_huggingface_model(
        self,
        model_name: str,
        model_type: str,
        use_cache: bool = True,
        device: str = 'auto',
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a model from Hugging Face.
        
        Args:
            model_name: Name of the model on Hugging Face Hub
            model_type: Type of the model
            use_cache: Whether to use the cache
            device: Device to load model on ('cpu', 'cuda', 'auto')
            **kwargs: Additional arguments to pass to the model loader
            
        Returns:
            Tuple of (model, processor)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ModelLoadError(
                "transformers package is not available. "
                "Install it with: pip install transformers"
            )
        
        # Check cache first if enabled
        if use_cache:
            cached_model = self._get_from_cache(model_name, model_type)
            if cached_model is not None:
                logger.info(f"Loaded model {model_name} ({model_type}) from cache")
                return cached_model
        
        # Get actual device
        torch_device = self.get_device(device)
        
        # Load model based on type
        try:
            if model_type == 'vit':
                # Vision Transformer
                model = ViTForImageClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    **kwargs
                )
                processor = AutoImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                
            elif model_type == 'wav2vec2':
                # Wav2Vec2 for audio
                model = Wav2Vec2ForCTC.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    **kwargs
                )
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                
            elif model_type in ['timesformer', 'genconvit']:
                # Video models
                # Note: These specific model classes might need to be imported
                # from specialized packages or implemented locally
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    **kwargs
                )
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                
            else:
                # Generic auto model
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                    **kwargs
                )
                
                # Try to load an appropriate processor
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=self.cache_dir,
                        local_files_only=False
                    )
                except Exception:
                    # Fall back to feature extractor
                    try:
                        processor = AutoFeatureExtractor.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir,
                            local_files_only=False
                        )
                    except Exception:
                        # Fall back to tokenizer
                        try:
                            processor = AutoTokenizer.from_pretrained(
                                model_name,
                                cache_dir=self.cache_dir,
                                local_files_only=False
                            )
                        except Exception:
                            processor = None
                            logger.warning(f"No processor found for model {model_name}")
            
            # Move model to device
            model = model.to(torch_device)
            
            # Set evaluation mode
            model.eval()
            
            # Check compatibility
            if not self._check_model_compatibility(model, model_type):
                raise ModelLoadError(
                    f"Model {model_name} is not compatible with type {model_type}"
                )
            
            # Cache the model if enabled
            if use_cache:
                self._add_to_cache(model_name, model_type, (model, processor))
            
            logger.info(f"Loaded model {model_name} ({model_type}) from Hugging Face")
            return model, processor
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def load_custom_model(
        self,
        model_path: str,
        model_type: str,
        custom_loader=None,
        use_cache: bool = True,
        device: str = 'auto',
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a custom model from a local path.
        
        Args:
            model_path: Path to the model
            model_type: Type of the model
            custom_loader: Custom loading function (optional)
            use_cache: Whether to use the cache
            device: Device to load model on ('cpu', 'cuda', 'auto')
            **kwargs: Additional arguments to pass to the model loader
            
        Returns:
            Tuple of (model, processor or None)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # Check cache first if enabled
        if use_cache:
            cached_model = self._get_from_cache(model_path, model_type)
            if cached_model is not None:
                logger.info(f"Loaded model {model_path} ({model_type}) from cache")
                return cached_model
        
        # Get actual device
        torch_device = self.get_device(device)
        
        try:
            # Use custom loader if provided
            if custom_loader:
                model, processor = custom_loader(model_path, **kwargs)
            else:
                # Default loading based on file extension
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    # PyTorch model
                    model = torch.load(model_path, map_location=torch_device)
                    processor = None
                elif model_path.endswith('.onnx'):
                    # ONNX model
                    try:
                        import onnxruntime as ort
                        model = ort.InferenceSession(model_path)
                        processor = None
                    except ImportError:
                        raise ModelLoadError(
                            "onnxruntime package is not available. "
                            "Install it with: pip install onnxruntime"
                        )
                else:
                    raise ModelLoadError(
                        f"Unsupported model format for {model_path}. "
                        "Provide a custom_loader function."
                    )
            
            # Move model to device if it's a PyTorch model
            if hasattr(model, 'to') and callable(getattr(model, 'to')):
                model = model.to(torch_device)
                
                # Set evaluation mode if applicable
                if hasattr(model, 'eval') and callable(getattr(model, 'eval')):
                    model.eval()
            
            # Cache the model if enabled
            if use_cache:
                self._add_to_cache(model_path, model_type, (model, processor))
            
            logger.info(f"Loaded custom model {model_path} ({model_type})")
            return model, processor
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load custom model {model_path}: {str(e)}")
    
    def load_model(
        self,
        model_name_or_path: str,
        model_type: str,
        custom_loader=None,
        use_cache: bool = True,
        device: str = 'auto',
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a model (auto-detects if it's a Hugging Face model or local file).
        
        Args:
            model_name_or_path: Model name (Hugging Face) or path (local)
            model_type: Type of the model
            custom_loader: Custom loading function for local models
            use_cache: Whether to use the cache
            device: Device to load model on ('cpu', 'cuda', 'auto')
            **kwargs: Additional arguments to pass to the model loader
            
        Returns:
            Tuple of (model, processor or None)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # Determine if it's a Hugging Face model or local file
        if self._is_huggingface_model(model_name_or_path):
            return self.load_huggingface_model(
                model_name_or_path, model_type, use_cache, device, **kwargs
            )
        else:
            return self.load_custom_model(
                model_name_or_path, model_type, custom_loader, use_cache, device, **kwargs
            )
    
    def load_image_model(self, config_name: str = 'default') -> Tuple[Any, Any]:
        """
        Load an image model based on configuration.
        
        Args:
            config_name: Name of the configuration to use
            
        Returns:
            Tuple of (model, processor)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # Get configuration
        config = self.config_manager.get_image_config(config_name)
        
        # Extract parameters
        model_name = config.get('model_name')
        model_type = config.get('model_type')
        device = config.get('device')
        
        # Load model with configuration parameters
        return self.load_model(
            model_name,
            model_type,
            use_cache=True,
            device=device
        )
    
    def load_audio_model(self, config_name: str = 'default') -> Tuple[Any, Any]:
        """
        Load an audio model based on configuration.
        
        Args:
            config_name: Name of the configuration to use
            
        Returns:
            Tuple of (model, processor)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # Get configuration
        config = self.config_manager.get_audio_config(config_name)
        
        # Extract parameters
        model_name = config.get('model_name')
        model_type = config.get('model_type')
        device = config.get('device')
        
        # Load model with configuration parameters
        return self.load_model(
            model_name,
            model_type,
            use_cache=True,
            device=device
        )
    
    def load_video_model(self, config_name: str = 'default') -> Dict[str, Tuple[Any, Any]]:
        """
        Load video models based on configuration.
        
        Since video detection often uses multiple models (frame and temporal),
        this returns a dictionary of models.
        
        Args:
            config_name: Name of the configuration to use
            
        Returns:
            Dictionary of (model, processor) tuples by component
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # Get configuration
        config = self.config_manager.get_video_config(config_name)
        
        # Extract parameters
        model_type = config.get('model_type')
        frame_model_name = config.get('frame_model_name')
        device = config.get('device')
        
        # Dictionary to hold models
        models = {}
        
        # Load frame model
        models['frame'] = self.load_model(
            frame_model_name,
            'vit',  # Frame model is typically ViT based
            use_cache=True,
            device=device
        )
        
        # Load specific video model based on type
        if model_type == 'timesformer':
            # TimeSformer model - in a real implementation, use the actual model name
            # This is a placeholder for demonstration
            timesformer_name = "facebook/timesformer-base-finetuned-k400"
            models['temporal'] = self.load_model(
                timesformer_name,
                'timesformer',
                use_cache=True,
                device=device
            )
            
        elif model_type == 'genconvit':
            # GenConViT model - in a real implementation, use the actual model name
            # This is a placeholder for demonstration
            genconvit_name = "custom/genconvit-base"
            
            # This might use a custom loader for a specialized model
            def custom_genconvit_loader(path, **kwargs):
                # Placeholder for a custom loader
                # In a real implementation, this would load a specialized model
                if self._is_huggingface_model(path):
                    return self.load_huggingface_model(path, 'genconvit', **kwargs)
                else:
                    # Load a custom GenConViT model
                    # This is a placeholder
                    model = torch.load(path, map_location=self.get_device(device))
                    return model, None
            
            models['temporal'] = self.load_model(
                genconvit_name,
                'genconvit',
                custom_loader=custom_genconvit_loader,
                use_cache=True,
                device=device
            )
            
        elif model_type == 'hybrid':
            # Hybrid approach uses both models
            # TimeSformer for temporal analysis
            timesformer_name = "facebook/timesformer-base-finetuned-k400"
            models['temporal'] = self.load_model(
                timesformer_name,
                'timesformer',
                use_cache=True,
                device=device
            )
        
        # Return dictionary of models
        return models
    
    def check_model_update(self, model_name: str, current_version: str = None) -> Tuple[bool, str]:
        """
        Check if a model has an update available.
        
        Args:
            model_name: Name of the model on Hugging Face Hub
            current_version: Current version of the model (optional)
            
        Returns:
            Tuple of (update_available, latest_version)
            
        Raises:
            ModelVersionError: If version checking fails
        """
        # Only works for Hugging Face models
        if not self._is_huggingface_model(model_name):
            return False, current_version or "unknown"
        
        try:
            # Query Hugging Face API for model information
            api_url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(api_url)
            
            if response.status_code != 200:
                raise ModelVersionError(
                    f"Failed to get model information: {response.status_code}"
                )
            
            # Parse response
            model_info = response.json()
            
            # Extract version information
            # Note: Hugging Face doesn't have a standard version field,
            # so this is a bit of a guess based on common practices
            latest_version = model_info.get('sha', 'unknown')
            
            # If we don't know the current version, assume it's the latest
            if current_version is None:
                return False, latest_version
            
            # Check if update is available
            update_available = current_version != latest_version
            
            return update_available, latest_version
            
        except Exception as e:
            raise ModelVersionError(f"Failed to check for model updates: {str(e)}")
    
    def update_model(self, model_name: str, force: bool = False) -> Tuple[bool, str]:
        """
        Update a model to the latest version.
        
        Args:
            model_name: Name of the model on Hugging Face Hub
            force: Force update even if already up to date
            
        Returns:
            Tuple of (updated, version)
            
        Raises:
            ModelVersionError: If update fails
        """
        # Only works for Hugging Face models
        if not self._is_huggingface_model(model_name):
            raise ModelVersionError(f"Cannot update local model: {model_name}")
        
        try:
            # Get local version if available
            local_version = self._get_local_version(model_name)
            
            # Check if update is available
            update_available, latest_version = self.check_model_update(model_name, local_version)
            
            if not update_available and not force:
                logger.info(f"Model {model_name} is already up to date (version: {latest_version})")
                return False, latest_version
            
            # Clear model from cache
            self._remove_from_cache(model_name)
            
            # Update model files
            # This is done implicitly by Hugging Face when loading the model with local_files_only=False
            # But we need to explicitly clean the cache directory
            
            # Construct local directory path
            model_dir = os.path.join(self.cache_dir, model_name.replace('/', '--'))
            
            if os.path.exists(model_dir):
                # Backup the directory
                backup_dir = f"{model_dir}_backup_{int(time.time())}"
                shutil.move(model_dir, backup_dir)
                logger.info(f"Backed up existing model to {backup_dir}")
            
            # Reload the model which will fetch the latest version
            # This is a bit wasteful since we're not using the loaded model,
            # but it's a simple way to make sure we have the latest version
            self.load_huggingface_model(
                model_name,
                'auto',  # Generic model type
                use_cache=False,  # Don't cache this temporary load
                local_files_only=False  # Force fetch from Hugging Face
            )
            
            # Save the version information
            self._save_local_version(model_name, latest_version)
            
            logger.info(f"Model {model_name} updated to version {latest_version}")
            return True, latest_version
            
        except Exception as e:
            raise ModelVersionError(f"Failed to update model: {str(e)}")
    
    def _get_local_version(self, model_name: str) -> Optional[str]:
        """
        Get the local version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Version string or None if not found
        """
        version_file = os.path.join(self.cache_dir, f"{model_name.replace('/', '--')}_version.json")
        
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    version_info = json.load(f)
                return version_info.get('version')
            except Exception as e:
                logger.warning(f"Failed to read version file: {str(e)}")
        
        return None
    
    def _save_local_version(self, model_name: str, version: str) -> None:
        """
        Save the local version of a model.
        
        Args:
            model_name: Name of the model
            version: Version string
        """
        version_file = os.path.join(self.cache_dir, f"{model_name.replace('/', '--')}_version.json")
        
        try:
            version_info = {
                'name': model_name,
                'version': version,
                'timestamp': time.time()
            }
            
            with open(version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
                
            logger.debug(f"Saved version information for {model_name}: {version}")
                
        except Exception as e:
            logger.warning(f"Failed to save version information: {str(e)}")
    
    def _remove_from_cache(self, model_name: str) -> None:
        """
        Remove a model from the cache.
        
        Args:
            model_name: Name of the model
        """
        # Remove from all possible model types
        for model_type in ['vit', 'wav2vec2', 'timesformer', 'genconvit', 'auto']:
            cache_key = self._cache_key(model_name, model_type)
            if cache_key in self.model_cache:
                del self.model_cache[cache_key]
                logger.debug(f"Removed model {model_name} ({model_type}) from cache")