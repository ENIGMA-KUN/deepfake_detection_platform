#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frame analyzer for video deepfake detection.
Analyzes individual frames from videos using image deepfake detection techniques.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region,
                                       VideoFeatures)
from detectors.detector_utils import (extract_frames, identify_deepfake_category,
                                     measure_execution_time,
                                     normalize_confidence_score)
from detectors.image_detector.vit_detector import VitDetector

# Configure logger
logger = logging.getLogger(__name__)


class FrameAnalyzer(BaseDetector):
    """Frame-by-frame analyzer for video deepfake detection."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the frame analyzer.
        
        Args:
            config: Configuration dictionary
            model_path: Path to model file (optional)
        """
        # Frame extraction parameters
        self.max_frames = config.get("max_frames", 30)
        self.frame_interval = config.get("frame_interval", None)
        self.use_keyframes = config.get("use_keyframes", True)
        
        # Analysis parameters
        self.region_consistency_threshold = config.get("region_consistency_threshold", 0.3)
        self.min_detection_frames = config.get("min_detection_frames", 3)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the frame analyzer resources."""
        try:
            # Initialize image detector
            logger.info("Initializing image detector for frame analysis")
            self.image_detector = VitDetector(self.config, self.model_path)
            
            # Initialize object/face tracker if available
            self._initialize_tracker()
            
            logger.info("Frame analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize frame analyzer: {e}")
            raise RuntimeError(f"Error initializing frame analyzer: {e}")
    
    def _initialize_tracker(self) -> None:
        """Initialize object/face tracker."""
        try:
            import cv2
            
            # Choose appropriate tracker based on OpenCV version
            major_ver, minor_ver, _ = cv2.__version__.split('.')
            
            if int(major_ver) >= 4 and int(minor_ver) >= 5:
                # OpenCV 4.5+ has better trackers
                self.tracker_type = "KCF"  # Kernelized Correlation Filters
            else:
                # Fall back to older tracker for older OpenCV versions
                self.tracker_type = "KCF"
            
            logger.info(f"Using {self.tracker_type} tracker")
            
        except ImportError:
            logger.warning("OpenCV not available, tracking functionality will be limited")
            self.tracker_type = None
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run frame-by-frame analysis on the provided video.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.VIDEO, os.path.basename(media_path))
        result.model_used = "FrameAnalyzer"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid video file"
                return result
            
            # Preprocess video
            frames = self.preprocess(media_path)
            
            # Analyze frames
            frame_results = self._analyze_frames(frames)
            
            # Track regions across frames
            tracked_regions = self._track_regions_across_frames(frames, frame_results)
            
            # Postprocess results
            is_deepfake, confidence, frame_anomalies = self.postprocess(tracked_regions)
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            
            # Create video features
            result.video_features = VideoFeatures(
                temporal_inconsistencies=[],
                frame_anomalies=frame_anomalies,
                sync_issues=[]
            )
            
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["num_frames_analyzed"] = len(frames)
            result.metadata["frames_with_anomalies"] = len(frame_anomalies)
            
            # Identify deepfake categories
            result.categories = identify_deepfake_category(
                MediaType.VIDEO, confidence
            )
            
        except Exception as e:
            logger.error(f"Error during frame analysis: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> Dict[int, np.ndarray]:
        """Preprocess the video by extracting frames.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            Dictionary mapping frame indices to frame images
        """
        # Extract frames
        frames = extract_frames(
            media_path,
            frame_indices=None,
            max_frames=self.max_frames,
            interval=self.frame_interval
        )
        
        return frames
    
    def _analyze_frames(self, frames: Dict[int, np.ndarray]) -> Dict[int, DetectionResult]:
        """Analyze individual frames for deepfake content.
        
        Args:
            frames: Dictionary mapping frame indices to frame images
            
        Returns:
            Dictionary mapping frame indices to detection results
        """
        frame_results = {}
        
        # Create temporary directory for frame images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each frame
            for frame_idx, frame in frames.items():
                try:
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
                    
                    # Save using OpenCV to avoid PIL dependency
                    import cv2
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # Analyze frame using image detector
                    frame_result = self.image_detector.detect(frame_path)
                    
                    # Store result
                    frame_results[frame_idx] = frame_result
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame {frame_idx}: {e}")
        
        return frame_results
    
    def _track_regions_across_frames(
        self,
        frames: Dict[int, np.ndarray],
        frame_results: Dict[int, DetectionResult]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Track detected regions across frames.
        
        Args:
            frames: Dictionary mapping frame indices to frame images
            frame_results: Dictionary mapping frame indices to detection results
            
        Returns:
            Dictionary mapping frame indices to tracked regions
        """
        tracked_regions = {}
        
        try:
            import cv2
            
            # Sort frame indices
            frame_indices = sorted(frames.keys())
            
            # Initialize tracking
            active_trackers = []
            
            for i, frame_idx in enumerate(frame_indices):
                frame = frames[frame_idx]
                result = frame_results[frame_idx]
                
                # Height and width for coordinate conversion
                height, width = frame.shape[:2]
                
                # Regions for this frame
                frame_regions = []
                
                # Update existing trackers
                updated_trackers = []
                for tracker_info in active_trackers:
                    tracker = tracker_info["tracker"]
                    region_info = tracker_info["region_info"]
                    
                    # Convert frame to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Update tracker
                    success, bbox = tracker.update(frame_bgr)
                    
                    if success:
                        # Convert back to normalized coordinates
                        x, y, w, h = bbox
                        norm_x = x / width
                        norm_y = y / height
                        norm_width = w / width
                        norm_height = h / height
                        
                        # Decay confidence slightly over time (temporal consistency check)
                        confidence = region_info["confidence"] * 0.95
                        
                        # Create updated region info
                        updated_region = {
                            "x": norm_x,
                            "y": norm_y,
                            "width": norm_width,
                            "height": norm_height,
                            "confidence": confidence,
                            "label": region_info["label"],
                            "id": region_info["id"],
                            "tracked": True
                        }
                        
                        # Add to frame regions
                        frame_regions.append(updated_region)
                        
                        # Update tracker info
                        region_info["confidence"] = confidence
                        updated_trackers.append({
                            "tracker": tracker,
                            "region_info": region_info
                        })
                
                # Replace active trackers
                active_trackers = updated_trackers
                
                # Add new regions from detection
                if result.is_deepfake and result.regions:
                    for j, region in enumerate(result.regions):
                        # Check if this region overlaps significantly with any tracked region
                        is_new_region = True
                        
                        for tracked_region in frame_regions:
                            overlap = self._calculate_region_overlap(
                                region.x, region.y, region.width, region.height,
                                tracked_region["x"], tracked_region["y"],
                                tracked_region["width"], tracked_region["height"]
                            )
                            
                            if overlap > self.region_consistency_threshold:
                                is_new_region = False
                                break
                        
                        if is_new_region:
                            # Create region info
                            region_info = {
                                "x": region.x,
                                "y": region.y,
                                "width": region.width,
                                "height": region.height,
                                "confidence": region.confidence,
                                "label": region.label,
                                "id": f"region_{frame_idx}_{j}",
                                "tracked": False
                            }
                            
                            # Add to frame regions
                            frame_regions.append(region_info)
                            
                            # Initialize tracker for this region
                            if i < len(frame_indices) - 1 and self.tracker_type:
                                # Convert to pixel coordinates for tracker
                                x = int(region.x * width)
                                y = int(region.y * height)
                                w = int(region.width * width)
                                h = int(region.height * height)
                                
                                # Create appropriate tracker
                                if self.tracker_type == "KCF":
                                    tracker = cv2.TrackerKCF_create()
                                elif hasattr(cv2, "TrackerCSRT_create"):
                                    tracker = cv2.TrackerCSRT_create()
                                else:
                                    # Fall back to available tracker
                                    tracker = cv2.TrackerMIL_create()
                                
                                # Initialize tracker
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                tracker.init(frame_bgr, (x, y, w, h))
                                
                                # Add to active trackers
                                active_trackers.append({
                                    "tracker": tracker,
                                    "region_info": region_info
                                })
                
                # Store regions for this frame
                tracked_regions[frame_idx] = frame_regions
        
        except ImportError:
            logger.warning("OpenCV not available, tracking disabled")
            
            # Fall back to simple frame-by-frame regions without tracking
            for frame_idx, result in frame_results.items():
                frame_regions = []
                
                if result.is_deepfake and result.regions:
                    for j, region in enumerate(result.regions):
                        frame_regions.append({
                            "x": region.x,
                            "y": region.y,
                            "width": region.width,
                            "height": region.height,
                            "confidence": region.confidence,
                            "label": region.label,
                            "id": f"region_{frame_idx}_{j}",
                            "tracked": False
                        })
                
                tracked_regions[frame_idx] = frame_regions
        
        except Exception as e:
            logger.error(f"Error during region tracking: {e}")
        
        return tracked_regions
    
    def _calculate_region_overlap(
        self,
        x1: float, y1: float, w1: float, h1: float,
        x2: float, y2: float, w2: float, h2: float
    ) -> float:
        """Calculate intersection over union between two regions.
        
        Args:
            x1, y1, w1, h1: First region coordinates and dimensions
            x2, y2, w2, h2: Second region coordinates and dimensions
            
        Returns:
            Intersection over union ratio
        """
        # Calculate coordinates of the intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Calculate area of intersection rectangle
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both regions
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate IoU
        if area1 + area2 - intersection_area == 0:
            return 0.0
        
        iou = intersection_area / (area1 + area2 - intersection_area)
        return iou
    
    def postprocess(
        self,
        tracked_regions: Dict[int, List[Dict[str, Any]]]
    ) -> Tuple[bool, float, Dict[int, List[Region]]]:
        """Postprocess the tracking results to get detection results.
        
        Args:
            tracked_regions: Dictionary mapping frame indices to tracked regions
            
        Returns:
            Tuple of (is_deepfake, confidence_score, frame_anomalies)
        """
        # Count frames with detected regions
        frames_with_detections = 0
        total_confidence = 0.0
        
        for frame_idx, regions in tracked_regions.items():
            if regions:
                frames_with_detections += 1
                # Use maximum confidence in this frame
                frame_confidence = max(region["confidence"] for region in regions)
                total_confidence += frame_confidence
        
        # Calculate overall confidence
        if frames_with_detections > 0:
            # Scale by portion of frames with detections
            frame_ratio = frames_with_detections / len(tracked_regions)
            avg_confidence = total_confidence / frames_with_detections
            
            # Confidence is higher if more frames have detections
            overall_confidence = avg_confidence * min(1.0, frame_ratio * 2)
        else:
            overall_confidence = 0.0
        
        # Normalize confidence
        confidence = normalize_confidence_score(overall_confidence)
        
        # Determine if deepfake based on confidence threshold and minimum frames
        is_deepfake = (
            self.is_deepfake(confidence) and 
            frames_with_detections >= self.min_detection_frames
        )
        
        # Convert to Region objects for result
        frame_anomalies = {}
        
        for frame_idx, regions in tracked_regions.items():
            if regions:
                frame_regions = []
                
                for region_info in regions:
                    region = Region(
                        x=region_info["x"],
                        y=region_info["y"],
                        width=region_info["width"],
                        height=region_info["height"],
                        confidence=region_info["confidence"],
                        label=region_info["label"],
                        category=DeepfakeCategory.FACE_MANIPULATION
                    )
                    
                    frame_regions.append(region)
                
                frame_anomalies[frame_idx] = frame_regions
        
        return is_deepfake, confidence, frame_anomalies
    
    def validate_media(self, media_path: str) -> bool:
        """Validate if the media file is a supported video.
        
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
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_ext = os.path.splitext(media_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Unsupported video format: {file_ext}")
            return False
        
        # Try opening the video to ensure it's valid
        try:
            import cv2
            
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {media_path}")
                return False
            
            # Read first frame
            ret, _ = cap.read()
            if not ret:
                logger.error(f"Could not read frames from video: {media_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Basic validation
            if fps <= 0 or frame_count <= 0:
                logger.error(f"Invalid video properties: FPS={fps}, Frames={frame_count}")
                return False
            
            cap.release()
                
        except ImportError:
            logger.warning("OpenCV not available for video validation, performing basic checks only")
            
            # Basic file size check
            if os.path.getsize(media_path) == 0:
                logger.error(f"Empty file: {media_path}")
                return False
        
        except Exception as e:
            logger.error(f"Invalid video file: {e}")
            return False
        
        return True
    
    def get_media_type(self) -> MediaType:
        """Get the media type supported by this detector.
        
        Returns:
            MediaType.VIDEO
        """
        return MediaType.VIDEO