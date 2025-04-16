#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ensemble detector for image deepfake detection.
Combines multiple detection methods for improved accuracy.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region)
from detectors.detector_utils import (calculate_ensemble_confidence,
                                     create_heatmap,
                                     identify_deepfake_category,
                                     measure_execution_time,
                                     normalize_confidence_score)
from detectors.image_detector.ela_detector import ELADetector
from detectors.image_detector.face_detector import FaceDetector
from detectors.image_detector.vit_detector import VitDetector

# Configure logger
logger = logging.getLogger(__name__)


class ImageEnsembleDetector(BaseDetector):
    """Ensemble detector combining multiple image deepfake detection methods."""
    
    def __init__(self, config: Dict[str, Any], model_paths: Optional[Dict[str, str]] = None):
        """Initialize the ensemble detector.
        
        Args:
            config: Configuration dictionary
            model_paths: Dictionary mapping detector names to model paths (optional)
        """
        # Initialize member variables
        self.detectors = {}
        self.detector_weights = {}
        self.model_paths = model_paths or {}
        
        # Ensemble parameters
        self.detector_types = config.get("ensemble_detectors", ["vit", "ela", "face"])
        self.use_face_focus = config.get("use_face_focus", True)
        self.region_overlap_threshold = config.get("region_overlap_threshold", 0.3)
        
        # Default weights, can be overridden in config
        default_weights = {
            "vit": 0.6,
            "ela": 0.3,
            "face": 0.1
        }
        
        # Override with configured weights if available
        configured_weights = config.get("ensemble_weights", {})
        self.detector_weights = {**default_weights, **configured_weights}
        
        # Normalize weights
        weight_sum = sum(self.detector_weights.values())
        if weight_sum > 0:
            self.detector_weights = {k: v / weight_sum for k, v in self.detector_weights.items()}
        
        # Call parent initializer
        super().__init__(config)
    
    def _initialize(self) -> None:
        """Initialize all component detectors."""
        logger.info("Initializing Image Ensemble Detector")
        
        try:
            # Initialize component detectors
            for detector_type in self.detector_types:
                self._initialize_detector(detector_type)
            
            if not self.detectors:
                raise ValueError("No detectors were successfully initialized")
            
            logger.info(f"Image Ensemble Detector initialized with {len(self.detectors)} detectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize Image Ensemble Detector: {e}")
            raise RuntimeError(f"Error initializing Image Ensemble Detector: {e}")
    
    def _initialize_detector(self, detector_type: str) -> None:
        """Initialize a specific detector component.
        
        Args:
            detector_type: Type of detector to initialize
        """
        try:
            model_path = self.model_paths.get(detector_type)
            
            if detector_type == "vit":
                detector = VitDetector(self.config, model_path)
                self.detectors["vit"] = detector
                logger.info("ViT detector initialized")
                
            elif detector_type == "ela":
                detector = ELADetector(self.config, model_path)
                self.detectors["ela"] = detector
                logger.info("ELA detector initialized")
                
            elif detector_type == "face":
                detector = FaceDetector(self.config, model_path)
                self.detectors["face"] = detector
                logger.info("Face detector initialized")
                
            else:
                logger.warning(f"Unknown detector type: {detector_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {detector_type} detector: {e}")
            # Continue with other detectors
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run ensemble detection on the provided image.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            DetectionResult with combined detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.IMAGE, os.path.basename(media_path))
        result.model_used = "ImageEnsembleDetector"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid image file"
                return result
            
            # First, detect faces if face focus is enabled
            faces = []
            if self.use_face_focus and "face" in self.detectors:
                face_result = self.detectors["face"].detect(media_path)
                faces = face_result.regions
                result.metadata["num_faces"] = len(faces)
            
            # Run each detector and collect results
            detector_results = {}
            for name, detector in self.detectors.items():
                if name == "face":
                    # Face detector already ran
                    continue
                
                try:
                    detector_result = detector.detect(media_path)
                    detector_results[name] = detector_result
                    logger.info(f"{name} detector confidence: {detector_result.confidence_score:.3f}")
                except Exception as e:
                    logger.error(f"Error running {name} detector: {e}")
            
            # Process and combine results
            is_deepfake, confidence, regions, categories = self._combine_results(
                detector_results, faces
            )
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.regions = regions
            result.categories = categories
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["detector_confidences"] = {
                name: res.confidence_score for name, res in detector_results.items()
            }
            
        except Exception as e:
            logger.error(f"Error during ensemble detection: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def _combine_results(
        self,
        detector_results: Dict[str, DetectionResult],
        faces: List[Region]
    ) -> Tuple[bool, float, List[Region], List[DeepfakeCategory]]:
        """Combine results from multiple detectors.
        
        Args:
            detector_results: Dictionary mapping detector names to their results
            faces: List of face regions detected by face detector
            
        Returns:
            Tuple of (is_deepfake, confidence, regions, categories)
        """
        # Extract confidence scores from each detector
        confidences = {}
        for name, result in detector_results.items():
            if result.status == DetectionStatus.COMPLETED:
                confidences[name] = result.confidence_score
        
        # Calculate ensemble confidence
        if confidences:
            # Get weights for active detectors
            weights = [self.detector_weights.get(name, 0.5) for name in confidences.keys()]
            scores = list(confidences.values())
            
            # Calculate weighted confidence
            ensemble_confidence = calculate_ensemble_confidence(scores, weights)
        else:
            ensemble_confidence = 0.0
        
        # Normalize confidence
        confidence = normalize_confidence_score(ensemble_confidence)
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        # Combine detected regions with filtering
        all_regions = []
        for name, result in detector_results.items():
            if result.status == DetectionStatus.COMPLETED:
                # Add detector name to region labels
                for region in result.regions:
                    region.label = f"{name}_{region.label}"
                
                all_regions.extend(result.regions)
        
        # Filter and merge overlapping regions
        filtered_regions = self._filter_regions(all_regions, faces)
        
        # Collect all categories
        all_categories = set()
        for result in detector_results.values():
            if result.status == DetectionStatus.COMPLETED:
                all_categories.update(result.categories)
        
        # If no categories found, add default based on confidence
        if not all_categories and is_deepfake:
            all_categories.add(DeepfakeCategory.IMAGE_MANIPULATION)
        
        return is_deepfake, confidence, filtered_regions, list(all_categories)
    
    def _filter_regions(
        self,
        regions: List[Region],
        faces: List[Region]
    ) -> List[Region]:
        """Filter and merge overlapping regions.
        
        Args:
            regions: List of regions from all detectors
            faces: List of face regions
            
        Returns:
            Filtered list of regions
        """
        if not regions:
            return []
        
        # If face focus is enabled and faces are detected,
        # prioritize regions that overlap with faces
        if self.use_face_focus and faces:
            # Calculate overlap of each region with faces
            face_overlapping = []
            non_overlapping = []
            
            for region in regions:
                overlaps_with_face = False
                
                for face in faces:
                    overlap = self._calculate_region_overlap(region, face)
                    if overlap >= self.region_overlap_threshold:
                        overlaps_with_face = True
                        break
                
                if overlaps_with_face:
                    face_overlapping.append(region)
                else:
                    non_overlapping.append(region)
            
            # If we have regions that overlap with faces, prioritize those
            if face_overlapping:
                return face_overlapping
            
            # Otherwise, use all regions
            return regions
        
        # If no face focus or no faces, return all regions
        return regions
    
    def _calculate_region_overlap(self, region1: Region, region2: Region) -> float:
        """Calculate intersection over union (IoU) between two regions.
        
        Args:
            region1: First region
            region2: Second region
            
        Returns:
            Intersection over Union (IoU) ratio
        """
        # Calculate coordinates of the intersection rectangle
        x_left = max(region1.x, region2.x)
        y_top = max(region1.y, region2.y)
        x_right = min(region1.x + region1.width, region2.x + region2.width)
        y_bottom = min(region1.y + region1.height, region2.y + region2.height)
        
        # Calculate area of intersection rectangle
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both regions
        region1_area = region1.width * region1.height
        region2_area = region2.width * region2.height
        
        # Calculate IoU
        if region1_area + region2_area - intersection_area == 0:
            return 0.0
        
        iou = intersection_area / (region1_area + region2_area - intersection_area)
        return iou
    
    def preprocess(self, media_path: str) -> Any:
        """Not used directly in ensemble detector."""
        pass
    
    def postprocess(self, model_output: Any) -> Tuple[bool, float, List[Region]]:
        """Not used directly in ensemble detector."""
        pass
    
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
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        file_ext = os.path.splitext(media_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Unsupported image format: {file_ext}")
            return False
        
        return True
    
    def get_media_type(self) -> MediaType:
        """Get the media type supported by this detector.
        
        Returns:
            MediaType.IMAGE
        """
        return MediaType.IMAGE