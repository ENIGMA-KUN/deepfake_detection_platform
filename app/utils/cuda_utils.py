"""
Utility functions for CUDA and device management.
"""
import os
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def check_cuda_availability() -> Dict[str, Any]:
    """
    Check CUDA availability and return device information.
    
    Returns:
        Dictionary containing CUDA and device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cpu',
        'gpu_count': 0,
        'cuda_version': None,
        'gpu_names': []
    }
    
    if info['cuda_available']:
        info['device'] = 'cuda'
        info['gpu_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        
        # Get GPU names
        for i in range(info['gpu_count']):
            info['gpu_names'].append(torch.cuda.get_device_name(i))
        
        logger.info(f"CUDA is available: {info['cuda_version']}")
        logger.info(f"Found {info['gpu_count']} GPU(s): {', '.join(info['gpu_names'])}")
    else:
        logger.warning("CUDA is not available. Running on CPU only.")
    
    return info

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device (CUDA or CPU).
    
    Args:
        force_cpu: If True, will force CPU usage even if CUDA is available
        
    Returns:
        torch.device: The selected device
    """
    if force_cpu:
        logger.info("Forcing CPU usage as requested")
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Set default CUDA device (usually the first one)
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")
    
    return device

def set_cuda_cache_size(size_bytes: int = 1 * 1024 * 1024 * 1024):  # Default 1GB
    """
    Set the CUDA memory cache allocation size.
    
    Args:
        size_bytes: Size in bytes for the cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory allocation strategy
        if hasattr(torch.cuda, 'memory_reserved'):
            logger.info(f"Setting CUDA cache size to {size_bytes / (1024 * 1024 * 1024):.1f} GB")
            # For newer PyTorch versions
            torch.cuda.set_per_process_memory_fraction(min(size_bytes / torch.cuda.get_device_properties(0).total_memory, 0.9))
            
def configure_device_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Configure device settings based on config and environment.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with device information
    """
    # Get CUDA information
    cuda_info = check_cuda_availability()
    
    # Set environment variables if needed
    if cuda_info['cuda_available']:
        # Don't allow PyTorch to allocate all GPU memory at once
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Configure TF for GPU if using it
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Set cache size if config is provided
        if config and 'general' in config and 'cuda_memory_limit_gb' in config['general']:
            limit_gb = config['general']['cuda_memory_limit_gb']
            set_cuda_cache_size(int(limit_gb * 1024 * 1024 * 1024))
    
    return cuda_info
