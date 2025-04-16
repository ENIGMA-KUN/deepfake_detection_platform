#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Error Level Analysis (ELA) detector for image manipulation detection.
ELA identifies areas in an image that have been digitally altered by examining
compression artifacts.
"""

import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region)
from detectors.detector_utils import (identify_deepfake_category,
                                     load_image, measure_execution_time,
                                     normalize_confidence_score)

# Configure logger
logger = logging.getLogger(__name__)


class ELADetector(BaseDetector):
    """Error Level Analysis detector for image manipulation detection."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the ELA detector.
        
        Args:
            config: Configuration dictionary
            model_path: Not used for ELA (can be None)
        """
        # ELA-specific parameters
        self.quality = config.get("ela_quality", 85)  # JPEG quality for ELA
        self.scale = config.get("ela_scale", 10)  # Scale factor for ELA visualization
        self.threshold = config.get("ela_threshold", 0.15)  # Threshold for detecting significant ELA differences
        self.min_region_size = config.get("ela_min_region_size", 0.01)  # Minimum region size (as percentage of image)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the ELA detector resources."""
        try:
            # Check if required libraries are available
            from PIL import Image, ImageChops, ImageEnhance
            
            # Store required modules
            self.PIL_Image = Image
            self.PIL_ImageChops = ImageChops
            self.PIL_ImageEnhance = ImageEnhance
            
            logger.info("ELA detector initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("PIL (Pillow) library is required for ELA detector")
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run ELA detection on the provided image.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.IMAGE, os.path.basename(media_path))
        result.model_used = "ELA_Detector"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid image file"
                return result
            
            # Preprocess image
            preprocessed = self.preprocess(media_path)
            
            # Run ELA analysis
            ela_image, ela_stats = self._perform_ela(preprocessed)
            
            # Postprocess results
            is_deepfake, confidence, regions = self.postprocess((ela_image, ela_stats))
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.regions = regions
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["ela_stats"] = ela_stats
            result.metadata["image_size"] = f"{preprocessed.width}x{preprocessed.height}"
            
            # Identify deepfake categories
            result.categories = identify_deepfake_category(
                MediaType.IMAGE, confidence, regions, 
                model_metadata={"detector_type": "ELA"}
            )
            
        except Exception as e:
            logger.error(f"Error during ELA detection: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> Any:
        """Preprocess the image for ELA analysis.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        # Load image as PIL Image
        return self.PIL_Image.open(media_path)
    
    def _perform_ela(self, image: Any) -> Tuple[Any, Dict[str, float]]:
        """Perform Error Level Analysis on the image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (ela_image, statistics)
        """
        # Save image to a temporary file with specified quality
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Save image with specified JPEG quality
            image.save(temp_path, 'JPEG', quality=self.quality)
            
            # Open the saved image
            saved_image = self.PIL_Image.open(temp_path)
            
            # Calculate ELA difference
            ela_image = self.PIL_ImageChops.difference(image, saved_image)
            
            # Enhance difference for better visualization
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale_factor = 255.0 / max(max_diff, 1)
            
            # Scale the difference image
            ela_image = self.PIL_ImageEnhance.Brightness(ela_image).enhance(scale_factor * self.scale)
            
            # Calculate statistics
            ela_array = np.array(ela_image)
            mean_diff = np.mean(ela_array)
            max_diff = np.max(ela_array)
            std_diff = np.std(ela_array)
            
            # Calculate percentage of pixels above threshold
            threshold_value = 255 * self.threshold
            above_threshold = np.sum(ela_array > threshold_value) / ela_array.size
            
            stats = {
                "mean_difference": float(mean_diff),
                "max_difference": float(max_diff),
                "std_difference": float(std_diff),
                "above_threshold_percentage": float(above_threshold)
            }
            
            return ela_image, stats
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def postprocess(self, model_output: Any) -> Tuple[bool, float, List[Region]]:
        """Postprocess the ELA output to get detection results.
        
        Args:
            model_output: Tuple of (ela_image, statistics)
            
        Returns:
            Tuple of (is_deepfake, confidence_score, regions)
        """
        ela_image, stats = model_output
        
        # Convert ELA image to numpy array for processing
        ela_array = np.array(ela_image)
        
        # Calculate confidence based on statistics
        # Higher percentage of pixels above threshold indicates manipulation
        above_threshold = stats["above_threshold_percentage"]
        std_normalized = min(1.0, stats["std_difference"] / 50.0)  # Normalize std
        
        # Combined confidence score - weighted average of metrics
        confidence = normalize_confidence_score(
            0.6 * above_threshold + 0.4 * std_normalized
        )
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        # Find regions with high ELA difference
        regions = []
        if is_deepfake:
            regions = self._identify_high_difference_regions(ela_array)
            
            # If no specific regions found, use whole image
            if not regions:
                regions.append(Region(
                    x=0.0,
                    y=0.0,
                    width=1.0,
                    height=1.0,
                    confidence=confidence,
                    label="ela_detected",
                    category=DeepfakeCategory.IMAGE_MANIPULATION
                ))
        
        return is_deepfake, confidence, regions
    
    def _identify_high_difference_regions(self, ela_array: np.ndarray) -> List[Region]:
        """Identify regions with high ELA differences.
        
        Args:
            ela_array: Numpy array of ELA image
            
        Returns:
            List of Region objects
        """
        try:
            from skimage import measure
            
            # Convert to grayscale if needed
            if len(ela_array.shape) == 3:
                gray_ela = np.mean(ela_array, axis=2)
            else:
                gray_ela = ela_array
            
            # Threshold the ELA image
            threshold_value = 255 * self.threshold
            binary = gray_ela > threshold_value
            
            # Find connected regions
            labeled = measure.label(binary, connectivity=2)
            regions_props = measure.regionprops(labeled)
            
            # Extract region properties
            height, width = gray_ela.shape
            min_size = self.min_region_size * width * height
            
            regions = []
            for props in regions_props:
                # Filter small regions
                if props.area < min_size:
                    continue
                
                # Calculate region properties
                y, x = props.centroid
                minr, minc, maxr, maxc = props.bbox
                
                # Calculate normalized coordinates
                norm_x = minc / width
                norm_y = minr / height
                norm_width = (maxc - minc) / width
                norm_height = (maxr - minr) / height
                
                # Calculate region confidence based on intensity
                intensity = props.mean_intensity
                region_confidence = min(1.0, intensity / 255.0)
                
                regions.append(Region(
                    x=norm_x,
                    y=norm_y,
                    width=norm_width,
                    height=norm_height,
                    confidence=region_confidence,
                    label=f"ela_region_{len(regions)}",
                    category=DeepfakeCategory.IMAGE_MANIPULATION
                ))
            
            return regions
            
        except ImportError:
            logger.warning("scikit-image not available, unable to identify specific regions")
            return []
    
    def validate_media(self, media_path: str) -> bool:
        """Validate if the media file is a supported image.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            True if media is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(media_path):
            logger.error(f"File does not exist: {media_path}")
            return False
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        file_ext = os.path.splitext(media_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Unsupported image format for ELA: {file_ext}")
            return False
        
        # Try opening the image to ensure it's valid
        try:
            with self.PIL_Image.open(media_path) as img:
                img.verify()
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return False
        
        return True
    
    def get_media_type(self) -> MediaType:
        """Get the media type supported by this detector.
        
        Returns:
            MediaType.IMAGE
        """
        return MediaType.IMAGE