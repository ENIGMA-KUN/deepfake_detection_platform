"""
Model loader utility for the deepfake detection platform.

Handles model loading, caching, and API key validation for premium models.
"""
import os
import time
import json
import logging
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml
import requests
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

# Ensure cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Model registry - maps model names to HuggingFace paths and premium status
MODEL_REGISTRY = {
    # Image Models
    "vit": {
        "path": "google/vit-base-patch16-224",
        "premium": False,
        "type": "image"
    },
    "deit": {
        "path": "facebook/deit-base-distilled-patch16-224",
        "premium": False,
        "type": "image"
    },
    "beit": {
        "path": "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "premium": False,
        "type": "image"
    },
    "swin": {
        "path": "microsoft/swin-base-patch4-window7-224",
        "premium": True,
        "type": "image"
    },
    
    # Audio Models
    "wav2vec2": {
        "path": "facebook/wav2vec2-large-960h",
        "premium": False,
        "type": "audio"
    },
    "xlsr": {
        "path": "facebook/wav2vec2-large-xlsr-53",
        "premium": False,
        "type": "audio"
    },
    "xlsr-mamba": {
        "path": "facebook/wav2vec2-xls-r-300m",  # Base model, would be fine-tuned
        "premium": True,
        "type": "audio"
    },
    
    # Video Models
    "timesformer": {
        "path": "facebook/timesformer-base-finetuned-k400",
        "premium": True,
        "type": "video"
    },
    "video-swin": {
        "path": "microsoft/swin-tiny-patch4-window7-224",  # Base model, would be fine-tuned
        "premium": True,
        "type": "video"
    },
    "genconvit": {
        "path": "google/vit-base-patch16-224",  # Base model, would be fine-tuned for video
        "premium": False,
        "type": "video"
    }
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml
    
    Returns:
        Dict containing configuration
    """
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        else:
            logger.warning(f"Config file not found at {CONFIG_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def get_api_key(service: str = "huggingface") -> Optional[str]:
    """
    Get API key for a specific service
    
    Args:
        service: Service name (default: huggingface)
        
    Returns:
        API key string or None if not found
    """
    config = load_config()
    api_keys = config.get("api_keys", {})
    return api_keys.get(service)

def check_premium_access(model_key: str) -> bool:
    """
    Check if user has access to premium models
    
    Args:
        model_key: Model key in the registry
        
    Returns:
        True if user has access, False otherwise
    """
    model_info = MODEL_REGISTRY.get(model_key, {})
    
    # If not a premium model, always grant access
    if not model_info.get("premium", False):
        return True
    
    # Check for API key
    api_key = get_api_key("huggingface")
    if not api_key:
        logger.warning(f"No API key found for premium model: {model_key}")
        return False
    
    # In a real implementation, would validate the key with the API
    # For now, just check if it exists and looks like a token
    if len(api_key) > 10:
        logger.info(f"API key found, granting access to premium model: {model_key}")
        return True
    
    return False

def get_model_path(model_key: str) -> str:
    """
    Get HuggingFace path for a model
    
    Args:
        model_key: Model key in the registry
        
    Returns:
        HuggingFace model path
    
    Raises:
        ValueError: If model key is not found
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")
    
    return MODEL_REGISTRY[model_key]["path"]

def get_model_info(model_key: str) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model_key: Model key in the registry
        
    Returns:
        Dict containing model information
    
    Raises:
        ValueError: If model key is not found
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")
    
    return MODEL_REGISTRY[model_key]

def download_file(url: str, dest_path: str) -> None:
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        dest_path: Destination path
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(dest_path)}") as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)

def get_cache_path(model_key: str) -> str:
    """
    Get cache path for a model
    
    Args:
        model_key: Model key in the registry
        
    Returns:
        Path to model cache directory
    """
    return os.path.join(MODEL_CACHE_DIR, model_key)

def is_model_cached(model_key: str) -> bool:
    """
    Check if model is cached
    
    Args:
        model_key: Model key in the registry
        
    Returns:
        True if model is cached, False otherwise
    """
    cache_path = get_cache_path(model_key)
    return os.path.exists(cache_path) and len(os.listdir(cache_path)) > 0

def load_model(model_key: str, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a model from cache or download it
    
    Args:
        model_key: Model key in the registry
        device: Device to load model on (default: auto-detect)
        
    Returns:
        Dict containing model and processor
        
    Raises:
        ValueError: If model key is not found or premium access is required
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_info = MODEL_REGISTRY[model_key]
    
    # Check premium access
    if model_info["premium"] and not check_premium_access(model_key):
        raise ValueError(f"Premium API key required for model: {model_key}")
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create cache directory if it doesn't exist
    cache_path = get_cache_path(model_key)
    os.makedirs(cache_path, exist_ok=True)
    
    # Path to model in Hugging Face
    model_path = model_info["path"]
    
    # Load appropriate model based on type
    try:
        if model_info["type"] == "image":
            return load_image_model(model_key, model_path, device)
        elif model_info["type"] == "audio":
            return load_audio_model(model_key, model_path, device)
        elif model_info["type"] == "video":
            return load_video_model(model_key, model_path, device)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
    except Exception as e:
        logger.error(f"Error loading model {model_key}: {str(e)}")
        raise

def load_image_model(model_key: str, model_path: str, device: str) -> Dict[str, Any]:
    """
    Load an image model
    
    Args:
        model_key: Model key in the registry
        model_path: HuggingFace model path
        device: Device to load model on
        
    Returns:
        Dict containing model and processor
    """
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    
    # Load feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        num_labels=2,  # Binary classification: real or fake
        ignore_mismatched_sizes=True
    )
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Return model and processor
    return {
        "model": model,
        "processor": feature_extractor,
        "device": device,
        "model_key": model_key,
        "model_path": model_path
    }

def load_audio_model(model_key: str, model_path: str, device: str) -> Dict[str, Any]:
    """
    Load an audio model
    
    Args:
        model_key: Model key in the registry
        model_path: HuggingFace model path
        device: Device to load model on
        
    Returns:
        Dict containing model and processor
    """
    from transformers import AutoProcessor, AutoModelForAudioClassification
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(
        model_path,
        num_labels=2,  # Binary classification: real or fake
        ignore_mismatched_sizes=True
    )
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Return model and processor
    return {
        "model": model,
        "processor": processor,
        "device": device,
        "model_key": model_key,
        "model_path": model_path
    }

def load_video_model(model_key: str, model_path: str, device: str) -> Dict[str, Any]:
    """
    Load a video model
    
    Args:
        model_key: Model key in the registry
        model_path: HuggingFace model path
        device: Device to load model on
        
    Returns:
        Dict containing model and processor
    """
    if model_key == "timesformer":
        from transformers import TimesformerForVideoClassification, VideoMAEImageProcessor
        
        # Load processor and model
        processor = VideoMAEImageProcessor.from_pretrained(model_path)
        model = TimesformerForVideoClassification.from_pretrained(
            model_path,
            num_labels=2,  # Binary classification: real or fake
            ignore_mismatched_sizes=True
        )
    elif model_key == "video-swin":
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
        
        # Load processor and model
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForVideoClassification.from_pretrained(
            model_path,
            num_labels=2,  # Binary classification: real or fake
            ignore_mismatched_sizes=True
        )
    else:
        # Generic approach for other models
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        
        # Load feature extractor and model - would extend for video in full implementation
        processor = AutoFeatureExtractor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            num_labels=2,  # Binary classification: real or fake
            ignore_mismatched_sizes=True
        )
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Return model and processor
    return {
        "model": model,
        "processor": processor,
        "device": device,
        "model_key": model_key,
        "model_path": model_path
    }

def list_available_models(media_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available models
    
    Args:
        media_type: Filter by media type (image, audio, video)
        
    Returns:
        List of model information dictionaries
    """
    models = []
    
    for key, info in MODEL_REGISTRY.items():
        if media_type is None or info["type"] == media_type:
            model_info = info.copy()
            model_info["key"] = key
            model_info["available"] = not model_info["premium"] or check_premium_access(key)
            model_info["cached"] = is_model_cached(key)
            models.append(model_info)
    
    return models

def clear_cache(model_key: Optional[str] = None) -> None:
    """
    Clear model cache
    
    Args:
        model_key: Model key in the registry, or None to clear all
    """
    if model_key is None:
        # Clear all models
        for key in MODEL_REGISTRY:
            cache_path = get_cache_path(key)
            if os.path.exists(cache_path):
                for file in os.listdir(cache_path):
                    os.remove(os.path.join(cache_path, file))
                logger.info(f"Cleared cache for model: {key}")
    else:
        # Clear specific model
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}")
        
        cache_path = get_cache_path(model_key)
        if os.path.exists(cache_path):
            for file in os.listdir(cache_path):
                os.remove(os.path.join(cache_path, file))
            logger.info(f"Cleared cache for model: {model_key}")

# API Key Management Functions
def set_api_key(service: str, key: str) -> None:
    """
    Set API key for a service in the config file
    
    Args:
        service: Service name (e.g., huggingface)
        key: API key
    """
    config = load_config()
    
    if "api_keys" not in config:
        config["api_keys"] = {}
    
    config["api_keys"][service] = key
    
    try:
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"API key for {service} set successfully")
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        raise
