"""
Utilities for handling premium features in the Deepfake Detection Platform.
"""
import os
import yaml
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)

# Cache for premium models to avoid repeatedly parsing config
_PREMIUM_MODELS: Optional[Dict[str, List[str]]] = None

def get_config_path() -> str:
    """Get the path to the configuration file."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, 'config.yaml')

def load_config() -> Dict:
    """Load the configuration from config.yaml."""
    try:
        config_path = get_config_path()
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def get_premium_models() -> Dict[str, List[str]]:
    """
    Get a dictionary of premium models by media type.
    
    Returns:
        Dictionary mapping media type to list of premium model keys
    """
    global _PREMIUM_MODELS
    
    if _PREMIUM_MODELS is not None:
        return _PREMIUM_MODELS
        
    premium_models = {
        'image': [],
        'audio': [],
        'video': []
    }
    
    try:
        config = load_config()
        
        # Check which API keys are defined in config
        api_keys = config.get('api_keys', {})
        
        # Define premium model mappings
        premium_mappings = {
            'image': {
                'swin': ['swin'],
                'huggingface': ['beit']
            },
            'audio': {
                'xlsr_mamba': ['xlsr', 'mamba'],
                'huggingface': []
            },
            'video': {
                'timesformer': ['timesformer'],
                'video_swin': ['video_swin'],
                'huggingface': []
            }
        }
        
        # Populate premium models based on API key requirements
        for key_name, key_value in api_keys.items():
            for media_type, model_lists in premium_mappings.items():
                if key_name in model_lists:
                    premium_models[media_type].extend(model_lists[key_name])
        
        _PREMIUM_MODELS = premium_models
        return premium_models
        
    except Exception as e:
        logger.error(f"Error determining premium models: {str(e)}")
        return premium_models

def is_premium_model(model_key: str, media_type: str = None) -> bool:
    """
    Check if a model is a premium model requiring an API key.
    
    Args:
        model_key: The model key to check
        media_type: Optional media type for context-specific checks
        
    Returns:
        Boolean indicating if the model is premium
    """
    premium_models = get_premium_models()
    model_key = model_key.lower()
    
    # If media type is specified, only check that type
    if media_type:
        return model_key in premium_models.get(media_type, [])
    
    # Otherwise check all media types
    for models in premium_models.values():
        if model_key in models:
            return True
            
    return False

def has_api_key(model_key: str) -> bool:
    """
    Check if the required API key for a premium model is configured.
    
    Args:
        model_key: The model key to check
        
    Returns:
        Boolean indicating if the API key is configured
    """
    try:
        config = load_config()
        api_keys = config.get('api_keys', {})
        
        # Map model keys to their API key names
        model_to_key_map = {
            'swin': 'swin',
            'beit': 'huggingface',
            'xlsr': 'xlsr_mamba',
            'mamba': 'xlsr_mamba',
            'timesformer': 'timesformer',
            'video_swin': 'video_swin'
        }
        
        # Look up the API key name for this model
        api_key_name = model_to_key_map.get(model_key.lower())
        if not api_key_name:
            return False
            
        # Check if the API key has a value
        api_key_value = api_keys.get(api_key_name, "")
        return bool(api_key_value.strip())
        
    except Exception as e:
        logger.error(f"Error checking API key for {model_key}: {str(e)}")
        return False

def get_premium_badge_html(model_key: str, media_type: str = None):
    """
    Generate HTML for a premium model badge.
    
    Args:
        model_key: The model key to check
        media_type: Optional media type for context-specific checks
        
    Returns:
        HTML component for premium badge or None if not premium
    """
    from dash import html
    
    if not is_premium_model(model_key, media_type):
        return None
        
    has_key = has_api_key(model_key)
    
    if has_key:
        # Premium with key configured
        return html.Span(
            "PREMIUM", 
            className="ms-2 badge bg-success",
            style={"fontSize": "10px", "verticalAlign": "middle"},
            title="Premium model with API key configured"
        )
    else:
        # Premium without key
        return html.Span(
            "PREMIUM (API Key Required)", 
            className="ms-2 badge bg-warning text-dark",
            style={"fontSize": "10px", "verticalAlign": "middle"},
            title="Premium model requiring API key configuration"
        )
