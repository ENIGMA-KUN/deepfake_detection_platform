"""
Model loader module for loading and managing detection models.
"""
import os
import logging
import torch
from typing import Dict, Any, Optional
import yaml

def load_model(model_name: str, model_type: str, **kwargs) -> Any:
    """
    Load a pretrained model for deepfake detection.
    
    Args:
        model_name: Name or path of the model to load
        model_type: Type of model ('image', 'audio', 'video')
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded model instance
        
    Raises:
        ValueError: If model_type is not valid
        RuntimeError: If model loading fails
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {model_type} model: {model_name}")
    
    try:
        if model_type == 'image':
            from transformers import ViTForImageClassification
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True,
                **kwargs
            )
            
        elif model_type == 'audio':
            from transformers import Wav2Vec2ForSequenceClassification
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True,
                **kwargs
            )
            
        elif model_type == 'video':
            # For video, we typically need multiple models
            # Here we're just loading the frame-level model
            from transformers import ViTForImageClassification
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move model to appropriate device if specified
        if 'device' in kwargs:
            model = model.to(kwargs['device'])
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info(f"Successfully loaded {model_type} model: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {str(e)}")
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")

def get_feature_extractor(model_name: str, model_type: str) -> Any:
    """
    Get the appropriate feature extractor for a model.
    
    Args:
        model_name: Name or path of the model
        model_type: Type of model ('image', 'audio', 'video')
        
    Returns:
        Feature extractor/processor instance
        
    Raises:
        ValueError: If model_type is not valid
    """
    logger = logging.getLogger(__name__)
    
    try:
        if model_type == 'image':
            from transformers import ViTFeatureExtractor
            return ViTFeatureExtractor.from_pretrained(model_name)
            
        elif model_type == 'audio':
            from transformers import Wav2Vec2Processor
            return Wav2Vec2Processor.from_pretrained(model_name)
            
        elif model_type == 'video':
            # For video frame analysis, use the same as image
            from transformers import ViTFeatureExtractor
            return ViTFeatureExtractor.from_pretrained(model_name)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading feature extractor: {str(e)}")
        raise RuntimeError(f"Failed to load feature extractor: {str(e)}")

def save_model_checkpoint(model: Any, save_path: str, metadata: Dict[str, Any] = None):
    """
    Save a model checkpoint with optional metadata.
    
    Args:
        model: Model to save
        save_path: Path to save the model to
        metadata: Optional metadata to save with the model
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model state dictionary
        if hasattr(model, 'save_pretrained'):
            # For Hugging Face models
            model.save_pretrained(save_path)
        else:
            # For PyTorch models
            torch.save(model.state_dict(), save_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(os.path.dirname(save_path), 
                                        f"{os.path.basename(save_path)}.meta.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
        
        logger.info(f"Model checkpoint saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error saving model checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to save model checkpoint: {str(e)}")

def load_model_checkpoint(model: Any, checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a model checkpoint and return associated metadata if available.
    
    Args:
        model: Model to load the checkpoint into
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Dictionary of metadata if available, empty dict otherwise
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If loading fails
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        # Load the model state
        if hasattr(model, 'from_pretrained'):
            # For Hugging Face models
            model = model.from_pretrained(checkpoint_path)
        else:
            # For PyTorch models
            model.load_state_dict(torch.load(checkpoint_path))
        
        # Try to load metadata if it exists
        metadata = {}
        metadata_path = os.path.join(os.path.dirname(checkpoint_path), 
                                    f"{os.path.basename(checkpoint_path)}.meta.yaml")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
        
        logger.info(f"Model checkpoint loaded from {checkpoint_path}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
