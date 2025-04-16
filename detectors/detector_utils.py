#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for deepfake detectors.
Provides common functionality used across different detector types.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region,
                                       TimeSegment)

# Configure logger
logger = logging.getLogger(__name__)


def normalize_confidence_score(score: float) -> float:
    """Normalize confidence score to range [0.0, 1.0].
    
    Args:
        score: Raw confidence score
        
    Returns:
        Normalized confidence score between 0.0 and 1.0
    """
    return float(max(0.0, min(1.0, score)))


def calculate_ensemble_confidence(
    scores: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """Calculate weighted ensemble confidence score.
    
    Args:
        scores: List of individual confidence scores
        weights: Optional list of weights for each score (must sum to 1.0)
                If None, equal weights are used
                
    Returns:
        Weighted ensemble confidence score
        
    Raises:
        ValueError: If lengths of scores and weights do not match
    """
    if not scores:
        return 0.0
    
    if weights is None:
        # Use equal weights
        weights = [1.0 / len(scores)] * len(scores)
    elif len(weights) != len(scores):
        raise ValueError("Length of weights must match length of scores")
    
    # Ensure weights sum to 1.0
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        weights = [w / weight_sum for w in weights]
    
    # Calculate weighted average
    return sum(s * w for s, w in zip(scores, weights))


def identify_deepfake_category(
    media_type: MediaType,
    confidence: float,
    regions: List[Region] = None,
    time_segments: List[TimeSegment] = None,
    model_metadata: Dict[str, Any] = None
) -> List[DeepfakeCategory]:
    """Identify deepfake categories based on detection results.
    
    Args:
        media_type: Type of media analyzed
        confidence: Overall confidence score
        regions: List of detected regions (for images/video frames)
        time_segments: List of detected time segments (for audio/video)
        model_metadata: Additional model-specific metadata
        
    Returns:
        List of identified deepfake categories
    """
    if confidence < 0.5:
        return []  # Not classified as deepfake
    
    categories = set()
    
    if model_metadata is None:
        model_metadata = {}
    
    # Extract categories from regions if available
    if regions:
        for region in regions:
            if region.category:
                categories.add(region.category)
    
    # Extract categories from time segments if available
    if time_segments:
        for segment in time_segments:
            if segment.category:
                categories.add(segment.category)
    
    # If no specific categories were found, use media type to determine default category
    if not categories:
        if media_type == MediaType.IMAGE:
            categories.add(DeepfakeCategory.GAN_GENERATED)
        elif media_type == MediaType.AUDIO:
            categories.add(DeepfakeCategory.AUDIO_SYNTHESIS)
        elif media_type == MediaType.VIDEO:
            categories.add(DeepfakeCategory.VIDEO_MANIPULATION)
    
    return list(categories)


def measure_execution_time(func):
    """Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that measures execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"Function {func.__name__} executed in {execution_time:.3f} seconds")
        
        # If the result is a DetectionResult, update its execution_time field
        if isinstance(result, DetectionResult):
            result.execution_time = execution_time
        
        return result
    
    return wrapper


def create_heatmap(
    image: np.ndarray,
    regions: List[Region],
    colormap: str = 'jet',
    alpha: float = 0.5
) -> np.ndarray:
    """Create a heatmap visualization of detected regions.
    
    Args:
        image: Original image as numpy array (height, width, channels)
        regions: List of detected regions
        colormap: Name of colormap to use
        alpha: Transparency of heatmap overlay (0.0 to 1.0)
        
    Returns:
        Image with heatmap overlay as numpy array
    """
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from skimage import img_as_float, img_as_ubyte
    except ImportError:
        logger.error("Required packages for heatmap generation not installed")
        return image
    
    # Create empty heatmap
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add detected regions to heatmap
    for region in regions:
        # Convert normalized coordinates to pixel coordinates
        x = int(region.x * width)
        y = int(region.y * height)
        w = int(region.width * width)
        h = int(region.height * height)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(1, min(width - x, w))
        h = max(1, min(height - y, h))
        
        # Add weighted region to heatmap
        heatmap[y:y+h, x:x+w] += region.confidence
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Extract RGB channels
    
    # Convert image to float for blending
    image_float = img_as_float(image)
    
    # Blend original image with heatmap
    blended = (1 - alpha) * image_float + alpha * heatmap_colored
    
    # Convert back to uint8
    return img_as_ubyte(blended)


def extract_frames(
    video_path: str,
    frame_indices: Optional[List[int]] = None,
    max_frames: int = 30,
    interval: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        frame_indices: Specific frame indices to extract
                      If None, frames are sampled at regular intervals
        max_frames: Maximum number of frames to extract if frame_indices is None
        interval: Interval between frames if frame_indices is None
                 If None, calculated based on video length and max_frames
                 
    Returns:
        Dictionary mapping frame indices to numpy arrays (height, width, channels)
        
    Raises:
        ImportError: If OpenCV is not installed
        ValueError: If video file cannot be opened
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is required for video frame extraction")
        raise ImportError("OpenCV (cv2) is required for video frame extraction")
    
    # Open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = {}
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        if frame_indices is not None:
            # Extract specific frames
            for idx in frame_indices:
                if 0 <= idx < total_frames:
                    video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = video.read()
                    if ret:
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames[idx] = frame
        else:
            # Sample frames at regular intervals
            if interval is None:
                interval = max(1, total_frames // max_frames)
            
            frame_idx = 0
            count = 0
            
            while count < max_frames and frame_idx < total_frames:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames[frame_idx] = frame
                    count += 1
                
                frame_idx += interval
    finally:
        video.release()
    
    return frames


def load_image(image_path: str) -> np.ndarray:
    """Load an image file as a numpy array.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (height, width, channels)
        
    Raises:
        ValueError: If image file cannot be opened
    """
    try:
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise ValueError(f"Could not open image file: {image_path}")


def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file as a numpy array.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate in Hz
        
    Returns:
        Tuple of (audio_data, sample_rate)
        audio_data is a numpy array of shape (samples,) for mono or (samples, channels) for stereo
        
    Raises:
        ImportError: If librosa is not installed
        ValueError: If audio file cannot be opened
    """
    try:
        import librosa
    except ImportError:
        logger.error("librosa is required for audio loading")
        raise ImportError("librosa is required for audio loading")
    
    try:
        # Load audio with resampling
        audio_data, actual_sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        return audio_data, actual_sr
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        raise ValueError(f"Could not open audio file: {audio_path}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size.
    
    Args:
        image: Image as numpy array (height, width, channels)
        target_size: Target size as (width, height)
        
    Returns:
        Resized image as numpy array
    """
    target_width, target_height = target_size
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height))
    return np.array(resized_image)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Image as numpy array (height, width, channels)
        
    Returns:
        Normalized image as float32 numpy array
    """
    return image.astype(np.float32) / 255.0