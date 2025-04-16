#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal analysis module for video deepfake detection.
Analyzes temporal patterns and inconsistencies across video frames to detect deepfakes.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region,
                                       TimeSegment, VideoFeatures)
from detectors.detector_utils import (extract_frames, identify_deepfake_category,
                                     measure_execution_time,
                                     normalize_confidence_score)

# Configure logger
logger = logging.getLogger(__name__)


class TemporalAnalyzer(BaseDetector):
    """Temporal analyzer for video deepfake detection."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the temporal analyzer.
        
        Args:
            config: Configuration dictionary
            model_path: Path to model file (optional)
        """
        # Frame extraction parameters
        self.max_frames = config.get("max_frames", 60)
        self.frame_interval = config.get("frame_interval", None)
        
        # Analysis parameters
        self.temporal_window_size = config.get("temporal_window_size", 10)
        self.optical_flow_threshold = config.get("optical_flow_threshold", 0.5)
        self.facial_landmark_threshold = config.get("facial_landmark_threshold", 0.3)
        self.motion_continuity_threshold = config.get("motion_continuity_threshold", 0.6)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the temporal analyzer resources."""
        try:
            # Check if required libraries are available
            import cv2
            
            # Initialize facial landmark detector if available
            self._initialize_facial_landmark_detector()
            
            logger.info("Temporal analyzer initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("OpenCV is required for temporal analysis")
    
    def _initialize_facial_landmark_detector(self) -> None:
        """Initialize facial landmark detector."""
        try:
            import cv2
            import dlib
            
            # Try to load the facial landmark predictor
            self.use_facial_landmarks = True
            
            # Set path to the shape predictor model file
            if self.model_path:
                shape_predictor_path = self.model_path
            else:
                # If model_path is not specified, check for default location
                # This is a placeholder, in a real implementation you would
                # provide the actual path to the model file
                shape_predictor_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models", "cache", "shape_predictor_68_face_landmarks.dat"
                )
            
            # Check if model file exists
            if os.path.exists(shape_predictor_path):
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)
                logger.info("Facial landmark detector initialized")
            else:
                logger.warning(f"Facial landmark model not found at {shape_predictor_path}")
                self.use_facial_landmarks = False
                
        except ImportError:
            logger.warning("dlib not available, facial landmark analysis disabled")
            self.use_facial_landmarks = False
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run temporal analysis on the provided video.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.VIDEO, os.path.basename(media_path))
        result.model_used = "TemporalAnalyzer"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid video file"
                return result
            
            # Preprocess video
            frames, video_info = self.preprocess(media_path)
            
            # Analyze optical flow
            optical_flow_results = self._analyze_optical_flow(frames)
            
            # Analyze facial landmarks if enabled
            if self.use_facial_landmarks:
                landmark_results = self._analyze_facial_landmarks(frames)
            else:
                landmark_results = {"inconsistencies": []}
            
            # Analyze motion continuity
            motion_results = self._analyze_motion_continuity(frames)
            
            # Combine results
            combined_results = {
                "optical_flow": optical_flow_results,
                "facial_landmarks": landmark_results,
                "motion_continuity": motion_results,
                "video_info": video_info
            }
            
            # Postprocess results
            is_deepfake, confidence, time_segments = self.postprocess(combined_results)
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.time_segments = time_segments
            
            # Create video features
            result.video_features = VideoFeatures(
                temporal_inconsistencies=time_segments,
                frame_anomalies={},
                sync_issues=[]
            )
            
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["num_frames_analyzed"] = len(frames)
            result.metadata["fps"] = video_info.get("fps", 0)
            result.metadata["duration"] = video_info.get("duration", 0)
            
            # Identify deepfake categories
            result.categories = identify_deepfake_category(
                MediaType.VIDEO, confidence, time_segments=time_segments
            )
            
        except Exception as e:
            logger.error(f"Error during temporal analysis: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """Preprocess the video by extracting frames and information.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            Tuple of (frames_dict, video_info)
        """
        # Extract video information
        video_info = self._extract_video_info(media_path)
        
        # Extract frames
        frames = extract_frames(
            media_path,
            frame_indices=None,
            max_frames=self.max_frames,
            interval=self.frame_interval
        )
        
        return frames, video_info
    
    def _extract_video_info(self, media_path: str) -> Dict[str, Any]:
        """Extract information from the video file.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        import cv2
        
        video_info = {}
        
        # Open video file
        cap = cv2.VideoCapture(media_path)
        
        if cap.isOpened():
            # Get video properties
            video_info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            video_info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            if video_info["fps"] > 0:
                video_info["duration"] = video_info["frame_count"] / video_info["fps"]
            else:
                video_info["duration"] = 0
        
        # Release video
        cap.release()
        
        return video_info
    
    def _analyze_optical_flow(self, frames: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Analyze optical flow between consecutive frames.
        
        Args:
            frames: Dictionary mapping frame indices to frame images
            
        Returns:
            Dictionary with optical flow analysis results
        """
        import cv2
        
        results = {
            "flow_magnitudes": {},
            "inconsistencies": []
        }
        
        try:
            # Sort frame indices
            frame_indices = sorted(frames.keys())
            
            if len(frame_indices) < 2:
                return results
            
            # Calculate optical flow between consecutive frames
            prev_frame_gray = None
            
            for i, frame_idx in enumerate(frame_indices):
                frame = frames[frame_idx]
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                if prev_frame_gray is not None:
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    
                    # Calculate flow magnitude
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_magnitude = np.mean(magnitude)
                    
                    # Store result
                    results["flow_magnitudes"][frame_idx] = mean_magnitude
                    
                    # Check for anomalies (abrupt changes)
                    if i > 1 and frame_indices[i-2] in results["flow_magnitudes"]:
                        prev_magnitude = results["flow_magnitudes"][frame_indices[i-2]]
                        current_magnitude = mean_magnitude
                        
                        # Calculate relative change
                        if prev_magnitude > 0:
                            relative_change = abs(current_magnitude - prev_magnitude) / prev_magnitude
                        else:
                            relative_change = abs(current_magnitude)
                        
                        # Check if change exceeds threshold
                        if relative_change > self.optical_flow_threshold:
                            # Mark as inconsistency
                            results["inconsistencies"].append({
                                "frame_idx": frame_idx,
                                "prev_frame_idx": frame_indices[i-1],
                                "magnitude_change": relative_change,
                                "confidence": min(1.0, relative_change / (2 * self.optical_flow_threshold))
                            })
                
                # Update previous frame
                prev_frame_gray = gray
        
        except Exception as e:
            logger.error(f"Error during optical flow analysis: {e}")
        
        return results
    
    def _analyze_facial_landmarks(self, frames: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Analyze facial landmarks across frames.
        
        Args:
            frames: Dictionary mapping frame indices to frame images
            
        Returns:
            Dictionary with facial landmark analysis results
        """
        results = {
            "landmark_trajectories": {},
            "inconsistencies": []
        }
        
        if not self.use_facial_landmarks:
            return results
        
        try:
            import dlib
            import cv2
            import numpy as np
            
            # Sort frame indices
            frame_indices = sorted(frames.keys())
            
            if len(frame_indices) < self.temporal_window_size:
                return results
            
            # Extract landmarks from each frame
            landmarks_by_frame = {}
            
            for frame_idx in frame_indices:
                frame = frames[frame_idx]
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Detect faces
                faces = self.face_detector(gray)
                
                # Extract landmarks for each face
                frame_landmarks = []
                
                for face in faces:
                    # Get facial landmarks
                    shape = self.landmark_predictor(gray, face)
                    
                    # Convert to numpy array
                    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
                    
                    frame_landmarks.append(landmarks)
                
                landmarks_by_frame[frame_idx] = frame_landmarks
            
            # Track landmark trajectories
            # We use simple tracking based on face overlap between frames
            if len(landmarks_by_frame) >= 2:
                # Initialize trajectories with the first frame
                trajectories = []
                
                for i, landmarks in enumerate(landmarks_by_frame[frame_indices[0]]):
                    trajectory = {
                        "landmarks": [landmarks],
                        "frame_indices": [frame_indices[0]]
                    }
                    trajectories.append(trajectory)
                
                # Process subsequent frames
                for i in range(1, len(frame_indices)):
                    frame_idx = frame_indices[i]
                    frame_landmarks = landmarks_by_frame[frame_idx]
                    
                    # Match current landmarks with trajectories
                    matched = [False] * len(frame_landmarks)
                    
                    for trajectory in trajectories:
                        if not frame_landmarks:
                            break
                        
                        # Get most recent landmarks in this trajectory
                        prev_landmarks = trajectory["landmarks"][-1]
                        
                        # Find best match
                        best_match = -1
                        best_iou = 0
                        
                        for j, landmarks in enumerate(frame_landmarks):
                            if matched[j]:
                                continue
                            
                            # Calculate bounding box IoU
                            prev_bbox = self._get_landmarks_bbox(prev_landmarks)
                            curr_bbox = self._get_landmarks_bbox(landmarks)
                            
                            iou = self._calculate_bbox_iou(prev_bbox, curr_bbox)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_match = j
                        
                        # If good match found, add to trajectory
                        if best_match >= 0 and best_iou > 0.5:
                            trajectory["landmarks"].append(frame_landmarks[best_match])
                            trajectory["frame_indices"].append(frame_idx)
                            matched[best_match] = True
                    
                    # Add new trajectories for unmatched landmarks
                    for j, landmarks in enumerate(frame_landmarks):
                        if not matched[j]:
                            trajectory = {
                                "landmarks": [landmarks],
                                "frame_indices": [frame_idx]
                            }
                            trajectories.append(trajectory)
                
                # Store trajectory information
                results["landmark_trajectories"] = trajectories
                
                # Analyze trajectories for inconsistencies
                for trajectory in trajectories:
                    if len(trajectory["landmarks"]) < self.temporal_window_size:
                        continue
                    
                    # Analyze landmark movement smoothness
                    inconsistencies = self._analyze_landmark_smoothness(
                        trajectory["landmarks"], trajectory["frame_indices"]
                    )
                    
                    results["inconsistencies"].extend(inconsistencies)
        
        except Exception as e:
            logger.error(f"Error during facial landmark analysis: {e}")
        
        return results
    
    def _get_landmarks_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box for landmarks.
        
        Args:
            landmarks: Numpy array of landmark points
            
        Returns:
            Tuple of (x, y, width, height)
        """
        min_x = np.min(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_x = np.max(landmarks[:, 0])
        max_y = np.max(landmarks[:, 1])
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _calculate_bbox_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate intersection over union for two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, width, height)
            bbox2: Second bounding box (x, y, width, height)
            
        Returns:
            Intersection over union ratio
        """
        # Convert to (x1, y1, x2, y2) format
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area
    
    def _analyze_landmark_smoothness(
        self,
        landmarks_sequence: List[np.ndarray],
        frame_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Analyze smoothness of landmark movement.
        
        Args:
            landmarks_sequence: List of landmark arrays for consecutive frames
            frame_indices: List of corresponding frame indices
            
        Returns:
            List of detected inconsistencies
        """
        inconsistencies = []
        
        # Select key landmarks to track
        # Typically eye corners, nose tip, mouth corners
        key_landmark_indices = [36, 39, 42, 45, 30, 48, 54]  # Standard indices in 68-point model
        
        for landmark_idx in key_landmark_indices:
            # Extract trajectory for this landmark
            trajectory = []
            
            for landmarks in landmarks_sequence:
                if landmark_idx < len(landmarks):
                    trajectory.append(landmarks[landmark_idx])
                else:
                    # Skip if landmark index is out of bounds
                    break
            
            if len(trajectory) < self.temporal_window_size:
                continue
            
            # Convert to numpy array
            trajectory = np.array(trajectory)
            
            # Calculate first and second derivatives
            velocity = np.diff(trajectory, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # Calculate jerk (rate of change of acceleration)
            jerk = np.diff(acceleration, axis=0)
            
            # Look for abnormal jerk values (indication of non-smooth motion)
            jerk_magnitude = np.linalg.norm(jerk, axis=1)
            mean_jerk = np.mean(jerk_magnitude)
            std_jerk = np.std(jerk_magnitude)
            
            # Check for outliers (values more than 3 sigma from mean)
            if std_jerk > 0:
                z_scores = (jerk_magnitude - mean_jerk) / std_jerk
                outliers = np.where(z_scores > 3)[0]
                
                for outlier_idx in outliers:
                    # Add 3 to account for the three diff operations
                    frame_idx = outlier_idx + 3
                    
                    if frame_idx < len(frame_indices) - 1:
                        # Calculate confidence based on how extreme the outlier is
                        confidence = min(1.0, z_scores[outlier_idx] / 6.0)  # Normalize
                        
                        inconsistencies.append({
                            "frame_idx": frame_indices[frame_idx],
                            "next_frame_idx": frame_indices[frame_idx + 1],
                            "landmark_idx": landmark_idx,
                            "jerk_value": float(jerk_magnitude[outlier_idx]),
                            "z_score": float(z_scores[outlier_idx]),
                            "confidence": float(confidence)
                        })
        
        return inconsistencies
    
    def _analyze_motion_continuity(self, frames: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Analyze overall motion continuity in the video.
        
        Args:
            frames: Dictionary mapping frame indices to frame images
            
        Returns:
            Dictionary with motion continuity analysis results
        """
        import cv2
        
        results = {
            "discontinuities": []
        }
        
        try:
            # Sort frame indices
            frame_indices = sorted(frames.keys())
            
            if len(frame_indices) < 3:
                return results
            
            # Calculate frame differences
            frame_diffs = []
            
            for i in range(1, len(frame_indices)):
                prev_idx = frame_indices[i-1]
                curr_idx = frame_indices[i]
                
                prev_frame = cv2.cvtColor(frames[prev_idx], cv2.COLOR_RGB2GRAY)
                curr_frame = cv2.cvtColor(frames[curr_idx], cv2.COLOR_RGB2GRAY)
                
                # Calculate absolute difference
                diff = cv2.absdiff(prev_frame, curr_frame)
                
                # Calculate mean difference
                mean_diff = np.mean(diff)
                
                frame_diffs.append((curr_idx, mean_diff))
            
            # Analyze differences for discontinuities
            if len(frame_diffs) < 2:
                return results
            
            # Calculate rate of change of differences
            diff_changes = []
            
            for i in range(1, len(frame_diffs)):
                prev_frame_idx, prev_diff = frame_diffs[i-1]
                curr_frame_idx, curr_diff = frame_diffs[i]
                
                if prev_diff > 0:
                    relative_change = abs(curr_diff - prev_diff) / prev_diff
                else:
                    relative_change = abs(curr_diff)
                
                diff_changes.append((curr_frame_idx, relative_change))
            
            # Detect abnormal changes
            mean_change = np.mean([change for _, change in diff_changes])
            std_change = np.std([change for _, change in diff_changes])
            
            threshold = mean_change + 2 * std_change
            
            for frame_idx, change in diff_changes:
                if change > threshold and change > self.motion_continuity_threshold:
                    # Calculate confidence score
                    confidence = min(1.0, change / (2 * threshold))
                    
                    results["discontinuities"].append({
                        "frame_idx": frame_idx,
                        "change_value": float(change),
                        "threshold": float(threshold),
                        "confidence": float(confidence)
                    })
        
        except Exception as e:
            logger.error(f"Error during motion continuity analysis: {e}")
        
        return results
    
    def postprocess(self, analysis_results: Dict[str, Any]) -> Tuple[bool, float, List[TimeSegment]]:
        """Postprocess the analysis results to get detection results.
        
        Args:
            analysis_results: Combined results from different analysis methods
            
        Returns:
            Tuple of (is_deepfake, confidence_score, time_segments)
        """
        # Extract video information
        video_info = analysis_results.get("video_info", {})
        fps = video_info.get("fps", 30.0)  # Default to 30 fps if not available
        
        # Collect all inconsistencies
        all_inconsistencies = []
        
        # Add optical flow inconsistencies
        for inconsistency in analysis_results.get("optical_flow", {}).get("inconsistencies", []):
            frame_idx = inconsistency["frame_idx"]
            all_inconsistencies.append({
                "frame_idx": frame_idx,
                "time": frame_idx / fps if fps > 0 else 0,
                "confidence": inconsistency["confidence"],
                "type": "optical_flow",
                "description": "Abrupt motion change"
            })
        
        # Add facial landmark inconsistencies
        for inconsistency in analysis_results.get("facial_landmarks", {}).get("inconsistencies", []):
            frame_idx = inconsistency["frame_idx"]
            all_inconsistencies.append({
                "frame_idx": frame_idx,
                "time": frame_idx / fps if fps > 0 else 0,
                "confidence": inconsistency["confidence"],
                "type": "facial_landmarks",
                "description": f"Unnatural landmark movement (landmark {inconsistency['landmark_idx']})"
            })
        
        # Add motion continuity inconsistencies
        for inconsistency in analysis_results.get("motion_continuity", {}).get("discontinuities", []):
            frame_idx = inconsistency["frame_idx"]
            all_inconsistencies.append({
                "frame_idx": frame_idx,
                "time": frame_idx / fps if fps > 0 else 0,
                "confidence": inconsistency["confidence"],
                "type": "motion_continuity",
                "description": "Motion discontinuity"
            })
        
        # Sort inconsistencies by time
        all_inconsistencies.sort(key=lambda x: x["time"])
        
        # Group inconsistencies into time segments
        time_segments = []
        
        if all_inconsistencies:
            # Time window for grouping (in seconds)
            window = 0.5
            
            current_segment = {
                "start_time": all_inconsistencies[0]["time"],
                "end_time": all_inconsistencies[0]["time"] + window,
                "inconsistencies": [all_inconsistencies[0]],
                "max_confidence": all_inconsistencies[0]["confidence"]
            }
            
            for inconsistency in all_inconsistencies[1:]:
                # Check if within window of current segment
                if inconsistency["time"] <= current_segment["end_time"]:
                    # Add to current segment
                    current_segment["inconsistencies"].append(inconsistency)
                    current_segment["end_time"] = inconsistency["time"] + window
                    current_segment["max_confidence"] = max(
                        current_segment["max_confidence"],
                        inconsistency["confidence"]
                    )
                else:
                    # Finish current segment and start a new one
                    time_segments.append(TimeSegment(
                        start_time=current_segment["start_time"],
                        end_time=current_segment["end_time"],
                        confidence=current_segment["max_confidence"],
                        label=f"temporal_anomaly_{len(time_segments)}",
                        category=DeepfakeCategory.VIDEO_MANIPULATION
                    ))
                    
                    current_segment = {
                        "start_time": inconsistency["time"],
                        "end_time": inconsistency["time"] + window,
                        "inconsistencies": [inconsistency],
                        "max_confidence": inconsistency["confidence"]
                    }
            
            # Add final segment
            time_segments.append(TimeSegment(
                start_time=current_segment["start_time"],
                end_time=current_segment["end_time"],
                confidence=current_segment["max_confidence"],
                label=f"temporal_anomaly_{len(time_segments)}",
                category=DeepfakeCategory.VIDEO_MANIPULATION
            ))
        
        # Calculate overall confidence
        if time_segments:
            # Use weighted combination of segment confidences
            # Higher weight for segments with higher confidence
            confidences = [segment.confidence for segment in time_segments]
            overall_confidence = sum(conf**2 for conf in confidences) / sum(conf for conf in confidences)
            
            # Scale by number of segments (more segments = higher confidence)
            coverage_factor = min(1.0, len(time_segments) / 5.0)  # Cap at 5 segments
            
            confidence = overall_confidence * coverage_factor
        else:
            confidence = 0.0
        
        # Normalize confidence
        confidence = normalize_confidence_score(confidence)
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        return is_deepfake, confidence, time_segments
    
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