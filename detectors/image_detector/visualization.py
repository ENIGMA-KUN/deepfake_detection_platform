#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for image deepfake detection results.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.detection_result import DetectionResult, Region

# Configure logger
logger = logging.getLogger(__name__)


def visualize_detection_result(
    image_path: str,
    result: DetectionResult,
    output_path: Optional[str] = None,
    show_heatmap: bool = True,
    show_regions: bool = True,
    colormap: str = 'jet',
    output_format: str = 'png'
) -> Optional[np.ndarray]:
    """Visualize detection result on the original image.
    
    Args:
        image_path: Path to the original image
        result: Detection result to visualize
        output_path: Path to save the visualization (None = don't save)
        show_heatmap: Whether to show heatmap overlay
        show_regions: Whether to show region outlines
        colormap: Colormap to use for heatmap
        output_format: Image format for saving
        
    Returns:
        Visualization as numpy array if successful, None otherwise
    """
    try:
        # Import visualization libraries
        import cv2
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw, ImageFont
        
        # Load original image
        original_image = np.array(Image.open(image_path))
        
        # Create PIL image for drawing
        pil_image = Image.fromarray(original_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Get image dimensions
        height, width = original_image.shape[:2]
        
        # Create heatmap if requested
        if show_heatmap and result.is_deepfake:
            heatmap = _create_heatmap(original_image, result.regions, colormap)
            pil_image = Image.fromarray(heatmap)
            draw = ImageDraw.Draw(pil_image)
        
        # Draw region outlines if requested
        if show_regions and result.regions:
            _draw_regions(draw, result.regions, width, height)
        
        # Add result text
        _add_result_text(draw, result, width, height)
        
        # Convert back to numpy array
        output_image = np.array(pil_image)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(output_image).save(output_path, format=output_format.upper())
            logger.info(f"Visualization saved to {output_path}")
        
        return output_image
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None


def _create_heatmap(
    image: np.ndarray,
    regions: List[Region],
    colormap: str = 'jet',
    alpha: float = 0.5
) -> np.ndarray:
    """Create a heatmap visualization of detected regions.
    
    Args:
        image: Original image as numpy array
        regions: List of detected regions
        colormap: Colormap to use
        alpha: Transparency of heatmap overlay
        
    Returns:
        Image with heatmap overlay as numpy array
    """
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from PIL import Image
        
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
        
        # Blend with original image
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        result = np.copy(image)
        
        # Apply alpha blending
        for c in range(min(result.shape[2], 3)):  # Handle both RGB and RGBA
            result[:, :, c] = image[:, :, c] * (1 - alpha * heatmap[:, :]) + \
                              heatmap_colored[:, :, c] * alpha * heatmap[:, :]
        
        return result.astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return image


def _draw_regions(
    draw: Any,
    regions: List[Region],
    width: int,
    height: int,
    outline_width: int = 2
) -> None:
    """Draw region outlines on the image.
    
    Args:
        draw: PIL ImageDraw object
        regions: List of regions to draw
        width: Image width
        height: Image height
        outline_width: Width of the outline in pixels
    """
    try:
        # Define colors for different confidence levels
        colors = {
            "high": (255, 0, 0),    # Red
            "medium": (255, 165, 0),  # Orange
            "low": (255, 255, 0)    # Yellow
        }
        
        for region in regions:
            # Determine color based on confidence
            if region.confidence >= 0.7:
                color = colors["high"]
            elif region.confidence >= 0.5:
                color = colors["medium"]
            else:
                color = colors["low"]
            
            # Convert normalized coordinates to pixels
            x = int(region.x * width)
            y = int(region.y * height)
            w = int(region.width * width)
            h = int(region.height * height)
            
            # Draw rectangle outline
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=color,
                width=outline_width
            )
            
            # Draw confidence label
            label = f"{region.confidence:.2f}"
            draw.text((x + 5, y + 5), label, fill=color)
            
    except Exception as e:
        logger.error(f"Error drawing regions: {e}")


def _add_result_text(
    draw: Any,
    result: DetectionResult,
    width: int,
    height: int
) -> None:
    """Add summary text to the visualization.
    
    Args:
        draw: PIL ImageDraw object
        result: Detection result
        width: Image width
        height: Image height
    """
    try:
        # Define text colors
        text_color = (255, 255, 255)  # White
        box_color = (0, 0, 0)         # Black
        
        # Define text
        if result.is_deepfake:
            status_text = "LIKELY FAKE"
            status_color = (255, 0, 0)  # Red
        else:
            status_text = "LIKELY AUTHENTIC"
            status_color = (0, 255, 0)  # Green
        
        confidence_text = f"Confidence: {result.confidence_score:.2f}"
        
        # Add categories if available
        categories_text = ""
        if result.categories:
            categories = ", ".join([cat.value for cat in result.categories])
            categories_text = f"Categories: {categories}"
        
        # Position for text
        x = 10
        y = height - 60
        
        # Draw semi-transparent background
        if categories_text:
            draw.rectangle([(0, y - 10), (width, height)], fill=(0, 0, 0, 128))
        else:
            draw.rectangle([(0, y - 10), (width, y + 30)], fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text((x, y), status_text, fill=status_color)
        draw.text((x + 200, y), confidence_text, fill=text_color)
        
        if categories_text:
            draw.text((x, y + 20), categories_text, fill=text_color)
            
    except Exception as e:
        logger.error(f"Error adding result text: {e}")


def create_comparison_grid(
    original_image_path: str,
    result: DetectionResult,
    output_path: Optional[str] = None,
    output_format: str = 'png'
) -> Optional[np.ndarray]:
    """Create a grid visualization comparing different detection components.
    
    Args:
        original_image_path: Path to the original image
        result: Detection result
        output_path: Path to save the visualization (None = don't save)
        output_format: Image format for saving
        
    Returns:
        Grid visualization as numpy array if successful, None otherwise
    """
    try:
        # Import visualization libraries
        import cv2
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw, ImageFont
        
        # Load original image
        original_image = np.array(Image.open(original_image_path))
        
        # Create different visualizations
        standard_vis = visualize_detection_result(
            original_image_path, result, None, show_heatmap=False
        )
        
        heatmap_vis = visualize_detection_result(
            original_image_path, result, None, show_regions=False
        )
        
        # Get detector confidences from metadata
        detector_confidences = result.metadata.get("detector_confidences", {})
        
        # Create grid image
        grid_height = original_image.shape[0] * 2
        grid_width = original_image.shape[1] * 2
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid
        grid_image[:original_image.shape[0], :original_image.shape[1]] = original_image
        
        if standard_vis is not None:
            grid_image[:original_image.shape[0], original_image.shape[1]:] = standard_vis
        
        if heatmap_vis is not None:
            grid_image[original_image.shape[0]:, :original_image.shape[1]] = heatmap_vis
        
        # Create info panel with detector confidences
        info_panel = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
        info_panel.fill(30)  # Dark gray background
        
        # Convert to PIL for text drawing
        info_pil = Image.fromarray(info_panel)
        draw = ImageDraw.Draw(info_pil)
        
        # Add title
        title = "Detection Information"
        draw.text((20, 20), title, fill=(255, 255, 255))
        
        # Add overall confidence
        overall = f"Overall confidence: {result.confidence_score:.3f}"
        draw.text((20, 60), overall, fill=(255, 255, 255))
        
        # Add individual detector confidences
        y_pos = 100
        for detector, confidence in detector_confidences.items():
            text = f"{detector}: {confidence:.3f}"
            draw.text((20, y_pos), text, fill=(255, 255, 255))
            y_pos += 30
        
        # Add categories if available
        if result.categories:
            y_pos += 20
            draw.text((20, y_pos), "Detected categories:", fill=(255, 255, 255))
            y_pos += 30
            
            for category in result.categories:
                draw.text((20, y_pos), f"- {category.value}", fill=(255, 255, 255))
                y_pos += 30
        
        # Convert back to numpy and place in grid
        info_panel = np.array(info_pil)
        grid_image[original_image.shape[0]:, original_image.shape[1]:] = info_panel
        
        # Add labels
        grid_pil = Image.fromarray(grid_image)
        draw = ImageDraw.Draw(grid_pil)
        
        # Label each quadrant
        draw.text((10, 10), "Original Image", fill=(255, 255, 255))
        draw.text((original_image.shape[1] + 10, 10), "Region Detection", fill=(255, 255, 255))
        draw.text((10, original_image.shape[0] + 10), "Heatmap Visualization", fill=(255, 255, 255))
        draw.text((original_image.shape[1] + 10, original_image.shape[0] + 10), "Detection Info", fill=(255, 255, 255))
        
        grid_image = np.array(grid_pil)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(grid_image).save(output_path, format=output_format.upper())
            logger.info(f"Comparison grid saved to {output_path}")
        
        return grid_image
        
    except Exception as e:
        logger.error(f"Error creating comparison grid: {e}")
        return None