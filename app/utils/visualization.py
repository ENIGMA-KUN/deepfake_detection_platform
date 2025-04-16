import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import base64
from io import BytesIO
import logging
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)

# Constants for visualization
DEFAULT_HEATMAP_COLORMAP = 'inferno'
DEFAULT_CONFIDENCE_COLORS = {
    'high': '#00FF00',  # Green for high confidence (authentic)
    'medium': '#FFFF00',  # Yellow for medium confidence
    'low': '#FF0000'  # Red for low confidence (deepfake)
}
DEFAULT_FONT = None  # Will try to find a system font

# Utility to encode images as base64 for web display
def encode_image_base64(img_array: np.ndarray) -> str:
    """
    Encode an image as base64 for embedding in HTML.
    
    Args:
        img_array: Image as numpy array (RGB)
        
    Returns:
        Base64 encoded string
    """
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Save to bytes buffer
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    
    # Encode as base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# Generate heatmap visualizations
def generate_heatmap(original_img: np.ndarray, activation_map: np.ndarray, 
                   alpha: float = 0.5, colormap: str = DEFAULT_HEATMAP_COLORMAP) -> np.ndarray:
    """
    Generate a heatmap overlay on an image.
    
    Args:
        original_img: Original image as numpy array
        activation_map: Activation values (0-1 range)
        alpha: Transparency factor for the overlay
        colormap: Matplotlib colormap name
        
    Returns:
        Heatmap visualization as numpy array
    """
    # Ensure activation map is normalized
    if activation_map.max() > 1.0 or activation_map.min() < 0.0:
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    
    # Resize activation map to match original image dimensions
    if activation_map.shape[:2] != original_img.shape[:2]:
        activation_map = cv2.resize(activation_map, 
                                   (original_img.shape[1], original_img.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(activation_map)
    heatmap = np.uint8(heatmap * 255)[:, :, :3]  # Remove alpha channel
    
    # Blend with original image
    blended = cv2.addWeighted(original_img, 1.0 - alpha, heatmap, alpha, 0)
    
    return blended

# Create confidence gauge visualization
def create_confidence_gauge(confidence: float, width: int = 200, height: int = 30, 
                           threshold_medium: float = 40.0, threshold_high: float = 70.0) -> np.ndarray:
    """
    Create a visual gauge showing confidence level.
    
    Args:
        confidence: Confidence value (0-100)
        width: Width of gauge image
        height: Height of gauge image
        threshold_medium: Threshold for medium confidence
        threshold_high: Threshold for high confidence
        
    Returns:
        Gauge visualization as numpy array
    """
    # Create blank image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Determine color based on confidence thresholds
    if confidence >= threshold_high:
        color = hex_to_rgb(DEFAULT_CONFIDENCE_COLORS['high'])
    elif confidence >= threshold_medium:
        color = hex_to_rgb(DEFAULT_CONFIDENCE_COLORS['medium'])
    else:
        color = hex_to_rgb(DEFAULT_CONFIDENCE_COLORS['low'])
    
    # Calculate fill width based on confidence
    fill_width = int((confidence / 100.0) * (width - 4))
    
    # Draw border and fill
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 0), 1)
    cv2.rectangle(img, (2, 2), (2 + fill_width, height - 3), color, -1)
    
    # Add text showing percentage
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{confidence:.1f}%"
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Add drop shadow for better visibility
    cv2.putText(img, text, (text_x + 1, text_y + 1), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img

# Helper function to convert hex color to RGB
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000')
        
    Returns:
        RGB tuple (0-255 range)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Generate timeline visualization for video analysis
def create_timeline_visualization(duration: float, markers: List[Dict[str, Any]], 
                                width: int = 800, height: int = 80) -> np.ndarray:
    """
    Create a visual timeline with markers for video analysis.
    
    Args:
        duration: Video duration in seconds
        markers: List of markers with position (0-100), width, and description
        width: Width of timeline image
        height: Height of timeline image
        
    Returns:
        Timeline visualization as numpy array
    """
    # Create blank image with Tron theme (dark background)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (10, 10, 30)  # Dark blue background
    
    # Draw timeline bar
    bar_y = height // 2
    bar_height = height // 6
    cv2.rectangle(img, (0, bar_y - bar_height//2), (width, bar_y + bar_height//2), (50, 50, 80), -1)
    cv2.rectangle(img, (0, bar_y - bar_height//2), (width, bar_y + bar_height//2), (0, 180, 220), 1)
    
    # Draw time markers
    for i in range(11):  # 0% to 100% in 10% increments
        x_pos = int((i / 10.0) * width)
        cv2.line(img, (x_pos, bar_y - bar_height), (x_pos, bar_y + bar_height), (0, 150, 200), 1)
        
        # Add time labels
        time_text = f"{duration * (i / 10.0):.1f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(time_text, font, 0.4, 1)[0]
        text_x = x_pos - text_size[0] // 2
        text_y = bar_y + bar_height + 15
        cv2.putText(img, time_text, (text_x, text_y), font, 0.4, (150, 220, 255), 1, cv2.LINE_AA)
    
    # Draw markers
    for marker in markers:
        position = marker.get('position', 0)  # 0-100 percentage
        marker_width = marker.get('width', 2)  # Width in percentage points
        description = marker.get('description', '')
        
        # Calculate pixel positions
        start_x = int((position / 100.0) * width)
        marker_w = max(5, int((marker_width / 100.0) * width))  # Ensure minimum visibility
        
        # Draw marker highlight on timeline
        cv2.rectangle(img, 
                     (start_x, bar_y - bar_height//2), 
                     (start_x + marker_w, bar_y + bar_height//2), 
                     (255, 50, 50), -1)  # Red marker
        
        # Add glow effect (Tron style)
        cv2.rectangle(img, 
                     (start_x - 1, bar_y - bar_height//2 - 1), 
                     (start_x + marker_w + 1, bar_y + bar_height//2 + 1), 
                     (255, 100, 100), 1)
        
        # Add description text above timeline
        if description:
            # Truncate long descriptions
            if len(description) > 20:
                description = description[:17] + "..."
                
            text_size = cv2.getTextSize(description, font, 0.4, 1)[0]
            text_x = min(start_x, width - text_size[0] - 5)  # Keep text within image
            text_y = bar_y - bar_height - 5
            
            # Draw text with glow effect
            cv2.putText(img, description, (text_x + 1, text_y + 1), font, 0.4, (255, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(img, description, (text_x, text_y), font, 0.4, (255, 150, 150), 1, cv2.LINE_AA)
    
    return img

# Generate spectrograms for audio visualization
def generate_audio_spectrogram(audio_data: np.ndarray, sample_rate: int, 
                             n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Generate a spectrogram visualization for audio analysis.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Audio sample rate in Hz
        n_fft: FFT window size
        hop_length: Number of samples between frames
        
    Returns:
        Spectrogram visualization as numpy array
    """
    try:
        import librosa
        import librosa.display
    except ImportError:
        logger.error("librosa is required for audio visualization")
        # Create an error image
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        cv2.putText(img, "librosa not installed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return img
    
    # Compute spectrogram
    D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create figure and plot spectrogram
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')  # Tron-like dark theme
    
    librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sample_rate, hop_length=hop_length, cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Analysis')
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Load image from buffer
    img = np.array(Image.open(buf))
    
    # Add Tron-style glow effects around the edges
    border_size = 2
    img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, 
                           cv2.BORDER_CONSTANT, value=(0, 220, 220))
    
    return img

# Save visualizations to disk
def save_visualization(img: np.ndarray, output_dir: str, filename: str, 
                     create_dir: bool = True) -> str:
    """
    Save visualization image to disk.
    
    Args:
        img: Image as numpy array
        output_dir: Directory to save the image
        filename: Filename (without directory)
        create_dir: Whether to create the output directory if it doesn't exist
        
    Returns:
        Path to the saved file
    """
    if create_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure filename has extension
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = f"{filename}.png"
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert to BGR for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # Save image
    cv2.imwrite(filepath, img_bgr)
    return filepath

# Create detection summary visualization
def create_detection_summary(original_img: np.ndarray, confidence: float, 
                           verdict: str, detector_name: str) -> np.ndarray:
    """
    Create a summary visualization with the image, confidence score, and verdict.
    
    Args:
        original_img: Original image as numpy array
        confidence: Confidence value (0-100)
        verdict: Detection verdict ('authentic' or 'deepfake')
        detector_name: Name of the detector used
        
    Returns:
        Summary visualization as numpy array
    """
    # Calculate dimensions
    img_height, img_width = original_img.shape[:2]
    padding = 20
    header_height = 60
    footer_height = 100
    total_height = img_height + header_height + footer_height + (padding * 3)
    total_width = max(img_width, 500) + (padding * 2)
    
    # Create Tron theme background (dark with grid)
    summary = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    summary[:, :] = (10, 20, 30)  # Dark blue-black background
    
    # Add subtle grid lines (Tron effect)
    grid_size = 40
    grid_color = (20, 40, 60)
    
    for i in range(0, total_width, grid_size):
        cv2.line(summary, (i, 0), (i, total_height), grid_color, 1)
    
    for i in range(0, total_height, grid_size):
        cv2.line(summary, (0, i), (total_width, i), grid_color, 1)
    
    # Add header
    header_y = padding
    header_color = (0, 200, 255) if verdict.lower() == 'authentic' else (0, 0, 255)
    
    # Draw header box with glow effect
    cv2.rectangle(summary, 
                 (padding, header_y), 
                 (total_width - padding, header_y + header_height), 
                 (header_color[0] // 4, header_color[1] // 4, header_color[2] // 4), -1)
    cv2.rectangle(summary, 
                 (padding, header_y), 
                 (total_width - padding, header_y + header_height), 
                 header_color, 2)
    
    # Add title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = f"Detection Result: {verdict.upper()}"
    title_size = cv2.getTextSize(title, font, 1, 2)[0]
    title_x = (total_width - title_size[0]) // 2
    title_y = header_y + (header_height + title_size[1]) // 2
    
    # Add glow effect to text
    cv2.putText(summary, title, (title_x + 2, title_y + 2), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(summary, title, (title_x, title_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Place original image
    img_y = header_y + header_height + padding
    summary[img_y:img_y + img_height, padding:padding + img_width] = original_img
    
    # Add border around image
    cv2.rectangle(summary, 
                 (padding - 2, img_y - 2), 
                 (padding + img_width + 1, img_y + img_height + 1), 
                 header_color, 2)
    
    # Add footer with confidence gauge
    footer_y = img_y + img_height + padding
    
    # Create confidence gauge
    gauge_width = min(400, img_width)
    gauge = create_confidence_gauge(confidence, width=gauge_width, height=30)
    gauge_x = (total_width - gauge_width) // 2
    gauge_y = footer_y + 20
    
    summary[gauge_y:gauge_y + gauge.shape[0], gauge_x:gauge_x + gauge.shape[1]] = gauge
    
    # Add detector info
    detector_text = f"Detector: {detector_name}"
    detector_size = cv2.getTextSize(detector_text, font, 0.5, 1)[0]
    detector_x = (total_width - detector_size[0]) // 2
    detector_y = gauge_y + gauge.shape[0] + 25
    
    cv2.putText(summary, detector_text, (detector_x, detector_y), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Add final border with glow effect
    cv2.rectangle(summary, (2, 2), (total_width - 3, total_height - 3), header_color, 2)
    
    return summary
