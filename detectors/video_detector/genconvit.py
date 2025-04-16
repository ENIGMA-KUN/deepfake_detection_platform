#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GenConViT/TimeSformer hybrid video deepfake detector.
Combines frame analysis and temporal analysis for comprehensive deepfake detection.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region,
                                       TimeSegment, VideoFeatures)
from detectors.detector_utils import (calculate_ensemble_confidence,
                                     identify_deepfake_category,
                                     measure_execution_time,
                                     normalize_confidence_score)
from detectors.video_detector.frame_analyzer import FrameAnalyzer
from detectors.video_detector.temporal_analysis import TemporalAnalyzer

# Configure logger
logger = logging.getLogger(__name__)


class GenConVitDetector(BaseDetector):
    """GenConViT/TimeSformer hybrid video deepfake detector."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the GenConViT detector.
        
        Args:
            config: Configuration dictionary
            model_path: Path to model file (optional)
        """
        # Component weights
        self.frame_weight = config.get("frame_weight", 0.6)
        self.temporal_weight = config.get("temporal_weight", 0.4)
        
        # Sync analysis parameters
        self.sync_analysis_enabled = config.get("sync_analysis_enabled", True)
        self.sync_threshold = config.get("sync_threshold", 0.3)
        
        # Video processing parameters
        self.max_duration_seconds = config.get("preprocessing", {}).get("video", {}).get("max_duration_seconds", 30)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the GenConViT detector components."""
        try:
            logger.info("Initializing GenConViT detector components")
            
            # Initialize frame analyzer
            self.frame_analyzer = FrameAnalyzer(self.config, self.model_path)
            
            # Initialize temporal analyzer
            self.temporal_analyzer = TemporalAnalyzer(self.config, self.model_path)
            
            # Initialize audio analysis if available
            self._initialize_audio_analysis()
            
            logger.info("GenConViT detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GenConViT detector: {e}")
            raise RuntimeError(f"Error initializing GenConViT detector: {e}")
    
    def _initialize_audio_analysis(self) -> None:
        """Initialize audio analysis components."""
        try:
            if self.sync_analysis_enabled:
                # Import here to avoid circular imports
                from detectors.audio_detector.wav2vec_detector import Wav2VecDetector
                
                # Create audio detector
                self.audio_detector = Wav2VecDetector(self.config)
                
                logger.info("Audio analysis initialized for sync detection")
            else:
                logger.info("Audio-video sync analysis disabled")
                self.audio_detector = None
                
        except ImportError as e:
            logger.warning(f"Could not import audio detector modules: {e}")
            self.sync_analysis_enabled = False
            self.audio_detector = None
        
        except Exception as e:
            logger.error(f"Error initializing audio analysis: {e}")
            self.sync_analysis_enabled = False
            self.audio_detector = None
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run deepfake detection on the provided video.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.VIDEO, os.path.basename(media_path))
        result.model_used = "GenConViT_Hybrid"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid video file"
                return result
            
            # Step 1: Run frame analysis
            frame_result = self.frame_analyzer.detect(media_path)
            
            # Step 2: Run temporal analysis
            temporal_result = self.temporal_analyzer.detect(media_path)
            
            # Step 3: Run audio-video sync analysis if enabled
            if self.sync_analysis_enabled and self.audio_detector:
                sync_issues = self._analyze_audio_video_sync(media_path)
            else:
                sync_issues = []
            
            # Combine results
            is_deepfake, confidence, video_features = self._combine_results(
                frame_result, temporal_result, sync_issues
            )
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.video_features = video_features
            
            # Get unique time segments
            result.time_segments = self._merge_time_segments(
                temporal_result.time_segments, sync_issues
            )
            
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["frame_confidence"] = frame_result.confidence_score
            result.metadata["temporal_confidence"] = temporal_result.confidence_score
            
            # Identify deepfake categories
            result.categories = identify_deepfake_category(
                MediaType.VIDEO, confidence, 
                time_segments=result.time_segments
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def _analyze_audio_video_sync(self, media_path: str) -> List[TimeSegment]:
        """Analyze audio-video synchronization.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            List of detected sync issues as TimeSegment objects
        """
        sync_issues = []
        
        try:
            import cv2
            import librosa
            
            # 1. Extract audio from video
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # Extract audio using ffmpeg via system call
                import subprocess
                
                ffmpeg_cmd = [
                    'ffmpeg', '-i', media_path, '-q:a', '0', '-map', 'a',
                    '-t', str(self.max_duration_seconds), temp_audio_path
                ]
                
                subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 2. Analyze audio for voice activity
                audio, sr = librosa.load(temp_audio_path, sr=None)
                
                # Detect speech segments
                # Simple energy-based voice activity detection
                energy = librosa.feature.rms(y=audio)[0]
                frames = np.arange(len(energy))
                times = librosa.frames_to_time(frames, sr=sr)
                
                # Threshold for speech detection
                threshold = np.mean(energy) + 0.5 * np.std(energy)
                
                # Find speech segments
                speech_mask = energy > threshold
                speech_changes = np.diff(speech_mask.astype(int))
                
                speech_starts = np.where(speech_changes == 1)[0]
                speech_ends = np.where(speech_changes == -1)[0]
                
                # Handle case where speech starts at beginning or ends at end
                if len(speech_starts) > 0 and len(speech_ends) > 0:
                    if speech_starts[0] > speech_ends[0]:
                        speech_starts = np.insert(speech_starts, 0, 0)
                    
                    if speech_starts[-1] > speech_ends[-1]:
                        speech_ends = np.append(speech_ends, len(speech_mask) - 1)
                
                # 3. Analyze video for mouth movement
                cap = cv2.VideoCapture(media_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if speech_starts.size > 0 and speech_ends.size > 0:
                    for i in range(min(len(speech_starts), len(speech_ends))):
                        start_time = times[speech_starts[i]]
                        end_time = times[speech_ends[i]]
                        
                        # Skip very short segments
                        if end_time - start_time < 0.3:
                            continue
                        
                        # Check corresponding video frames for mouth movement
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        
                        # Limit to valid frame range
                        start_frame = max(0, min(start_frame, total_frames - 1))
                        end_frame = max(0, min(end_frame, total_frames - 1))
                        
                        # Sample frames in this segment
                        has_mouth_movement = self._check_mouth_movement(
                            cap, start_frame, end_frame
                        )
                        
                        # Detected potential sync issue if speech but no mouth movement
                        if not has_mouth_movement:
                            confidence = min(1.0, (end_time - start_time) / 1.0)
                            
                            sync_issues.append(TimeSegment(
                                start_time=start_time,
                                end_time=end_time,
                                confidence=confidence,
                                label="audio_video_sync_issue",
                                category=DeepfakeCategory.VIDEO_MANIPULATION
                            ))
                
                cap.release()
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        
        except ImportError as e:
            logger.warning(f"Required library for sync analysis not available: {e}")
            
        except Exception as e:
            logger.error(f"Error during audio-video sync analysis: {e}")
        
        return sync_issues
    
    def _check_mouth_movement(
        self,
        video_capture: Any,
        start_frame: int,
        end_frame: int,
        sample_interval: int = 5
    ) -> bool:
        """Check for mouth movement in video segment.
        
        Args:
            video_capture: OpenCV VideoCapture object
            start_frame: Starting frame index
            end_frame: Ending frame index
            sample_interval: Interval for sampling frames
            
        Returns:
            True if mouth movement detected, False otherwise
        """
        try:
            import cv2
            import dlib
            
            # Limit number of samples
            max_samples = 10
            num_frames = end_frame - start_frame + 1
            num_samples = min(max_samples, 1 + (num_frames - 1) // sample_interval)
            
            # Sample frames evenly
            frame_indices = []
            if num_samples > 1:
                step = (num_frames - 1) / (num_samples - 1)
                for i in range(num_samples):
                    idx = start_frame + int(i * step)
                    frame_indices.append(idx)
            else:
                frame_indices = [start_frame]
            
            # Face detector
            face_detector = dlib.get_frontal_face_detector()
            
            # Try to load landmark predictor if available
            try:
                # Get directory of current file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Navigate to models/cache directory
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(current_dir)),
                    "models", "cache", "shape_predictor_68_face_landmarks.dat"
                )
                
                if os.path.exists(model_path):
                    landmark_predictor = dlib.shape_predictor(model_path)
                else:
                    return True  # Assume movement if we can't check
                    
            except Exception:
                return True  # Assume movement if we can't check
            
            # Extract mouth landmarks from each sampled frame
            mouth_landmarks = []
            
            for frame_idx in frame_indices:
                # Set frame position
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = video_capture.read()
                if not ret:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_detector(gray)
                
                if faces:
                    # Get first face
                    face = faces[0]
                    
                    # Get landmarks
                    shape = landmark_predictor(gray, face)
                    
                    # Extract mouth landmarks (indices 48-67 in 68-point model)
                    mouth = np.array([(shape.part(i).x, shape.part(i).y) for i in range(48, 68)])
                    
                    mouth_landmarks.append(mouth)
            
            # Check for significant movement in mouth landmarks
            if len(mouth_landmarks) >= 2:
                # Calculate displacement between consecutive frames
                max_displacement = 0
                
                for i in range(1, len(mouth_landmarks)):
                    # Calculate mean displacement
                    displacement = np.mean(np.linalg.norm(
                        mouth_landmarks[i] - mouth_landmarks[i-1], axis=1
                    ))
                    
                    max_displacement = max(max_displacement, displacement)
                
                # Return True if significant movement detected
                return max_displacement > self.sync_threshold
            
            return True  # Assume movement if not enough samples
            
        except Exception as e:
            logger.error(f"Error checking mouth movement: {e}")
            return True  # Assume movement if check fails
    
    def _combine_results(
        self,
        frame_result: DetectionResult,
        temporal_result: DetectionResult,
        sync_issues: List[TimeSegment]
    ) -> Tuple[bool, float, VideoFeatures]:
        """Combine results from different analysis components.
        
        Args:
            frame_result: Results from frame analysis
            temporal_result: Results from temporal analysis
            sync_issues: List of detected audio-video sync issues
            
        Returns:
            Tuple of (is_deepfake, confidence_score, video_features)
        """
        # Get component confidences
        frame_confidence = frame_result.confidence_score
        temporal_confidence = temporal_result.confidence_score
        
        # Calculate sync confidence
        if sync_issues:
            # Use maximum confidence from sync issues
            sync_confidence = max(issue.confidence for issue in sync_issues)
        else:
            sync_confidence = 0.0
        
        # Get component results
        frame_anomalies = frame_result.video_features.frame_anomalies if frame_result.video_features else {}
        temporal_inconsistencies = temporal_result.time_segments
        
        # Apply weights and calculate ensemble confidence
        weights = [self.frame_weight, self.temporal_weight]
        confidences = [frame_confidence, temporal_confidence]
        
        # Add sync confidence if available
        if sync_confidence > 0:
            # Adjust weights to include sync
            total_weight = self.frame_weight + self.temporal_weight
            weights = [
                self.frame_weight * 0.8 / total_weight,
                self.temporal_weight * 0.8 / total_weight,
                0.2  # Weight for sync
            ]
            confidences.append(sync_confidence)
        
        # Calculate ensemble confidence
        confidence = calculate_ensemble_confidence(confidences, weights)
        
        # Determine if deepfake based on confidence threshold
        is_deepfake = self.is_deepfake(confidence)
        
        # Create combined video features
        video_features = VideoFeatures(
            temporal_inconsistencies=temporal_inconsistencies,
            frame_anomalies=frame_anomalies,
            sync_issues=sync_issues
        )
        
        return is_deepfake, confidence, video_features
    
    def _merge_time_segments(
        self,
        temporal_segments: List[TimeSegment],
        sync_segments: List[TimeSegment]
    ) -> List[TimeSegment]:
        """Merge time segments from different analysis components.
        
        Args:
            temporal_segments: Time segments from temporal analysis
            sync_segments: Time segments from sync analysis
            
        Returns:
            Merged list of time segments
        """
        # Start with all segments
        all_segments = temporal_segments.copy()
        all_segments.extend(sync_segments)
        
        # No need to merge if fewer than 2 segments
        if len(all_segments) < 2:
            return all_segments
        
        # Sort by start time
        all_segments.sort(key=lambda x: x.start_time)
        
        # Merge overlapping segments
        merged = []
        current = all_segments[0]
        
        for segment in all_segments[1:]:
            # Check for overlap
            if segment.start_time <= current.end_time:
                # Merge segments
                end_time = max(current.end_time, segment.end_time)
                confidence = max(current.confidence, segment.confidence)
                
                # Use category with higher confidence
                category = (
                    segment.category if segment.confidence > current.confidence
                    else current.category
                )
                
                # Create merged segment
                current = TimeSegment(
                    start_time=current.start_time,
                    end_time=end_time,
                    confidence=confidence,
                    label=f"merged_{len(merged)}",
                    category=category
                )
            else:
                # Add current segment and start a new one
                merged.append(current)
                current = segment
        
        # Add final segment
        merged.append(current)
        
        return merged
    
    def preprocess(self, media_path: str) -> Any:
        """Not used directly, as component detectors handle preprocessing."""
        # Preprocessing is delegated to component detectors
        pass
    
    def postprocess(self, model_output: Any) -> Tuple[bool, float, List[Region]]:
        """Not used directly, as component detectors handle postprocessing."""
        # Postprocessing is delegated to component detectors
        pass
    
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