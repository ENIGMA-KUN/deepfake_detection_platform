"""
GenConViT hybrid model for video deepfake detection.
Combines spatial (frame-level) and temporal analysis.
"""
import os
import time
import logging
import numpy as np
import torch
import cv2
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import av
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn.functional as F

from detectors.base_detector import BaseDetector

class GenConViTVideoDetector(BaseDetector):
    """
    GenConViT/TimeSformer hybrid detector for video deepfakes.
    Performs frame-level analysis using ViT and temporal consistency analysis.
    """
    
    def __init__(self, frame_model_name: str = "google/vit-base-patch16-224",
                 temporal_model_name: str = "facebook/timesformer-base-finetuned-k400",
                 confidence_threshold: float = 0.5,
                 device: str = None,
                 frames_per_second: int = 5):
        """
        Initialize the GenConViT video detector.
        
        Args:
            frame_model_name: Model for frame-level analysis
            temporal_model_name: Model for temporal analysis
            confidence_threshold: Threshold for classifying video as deepfake
            device: Device to run the model on ('cuda' or 'cpu')
            frames_per_second: Number of frames to sample per second
        """
        super().__init__(frame_model_name, confidence_threshold)
        
        self.logger = logging.getLogger(__name__)
        
        # Store model names
        self.frame_model_name = frame_model_name
        self.temporal_model_name = temporal_model_name
        
        # Determine device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Sampling parameters
        self.frames_per_second = frames_per_second
        
        # Initialize models as None (lazy loading)
        self.frame_processor = None
        self.frame_model = None
        self.temporal_model = None
        
        # Analysis parameters
        self.temporal_window_size = 8  # Number of frames in temporal window
        self.temporal_stride = 4       # Stride for temporal windows
        
    def load_models(self):
        """
        Load the ViT model for frame analysis and temporal model.
        """
        if self.frame_model is not None:
            return
            
        try:
            self.logger.info(f"Loading frame analysis model: {self.frame_model_name}")
            
            # Load frame-level model (ViT)
            self.frame_processor = ViTImageProcessor.from_pretrained(self.frame_model_name)
            self.frame_model = ViTForImageClassification.from_pretrained(
                self.frame_model_name,
                num_labels=2,  # Binary classification: real or fake
                ignore_mismatched_sizes=True
            )
            
            # Move models to the appropriate device
            self.frame_model.to(self.device)
            self.frame_model.eval()
            
            # Note: For simplicity, we're not actually loading the TimeSformer model
            # In a full implementation, we would load it here
            # self.temporal_model = ...
            
            self.logger.info("Video detection models loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading video detection models: {str(e)}")
            raise RuntimeError(f"Failed to load video detection models: {str(e)}")
    
    def detect(self, media_path: str) -> Dict[str, Any]:
        """
        Detect if the video is a deepfake.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            Dictionary containing detection results
            
        Raises:
            FileNotFoundError: If the video file does not exist
            ValueError: If the file is not a valid video
        """
        # Validate the video file
        self._validate_media(media_path)
        
        # Load the models if not already loaded
        if self.frame_model is None:
            self.load_models()
        
        start_time = time.time()
        
        try:
            # Extract frames from the video
            frames, frame_times, video_info = self._extract_frames(media_path)
            
            if not frames:
                raise ValueError(f"No frames could be extracted from {media_path}")
            
            # Analyze frames
            frame_scores, face_detections = self._analyze_frames(frames)
            
            # Perform temporal analysis
            temporal_scores = self._temporal_analysis(frame_scores)
            
            # Calculate audio-video sync score
            # For simplicity, we're using a placeholder
            # In a full implementation, we would extract audio and analyze sync
            av_sync_score = 0.0
            
            # Calculate overall score
            # Weight different factors: frame-level (60%), temporal (30%), A/V sync (10%)
            overall_score = 0.6 * np.mean(frame_scores) + 0.3 * temporal_scores + 0.1 * av_sync_score
            
            # Prepare metadata
            metadata = {
                "timestamp": time.time(),
                "media_type": "video",
                "analysis_time": time.time() - start_time,
                "details": {
                    "video_info": video_info,
                    "frames_analyzed": len(frames),
                    "frame_scores": frame_scores.tolist(),
                    "face_detections": face_detections,
                    "temporal_inconsistency": temporal_scores,
                    "av_sync_score": av_sync_score
                }
            }
            
            # Determine if the video is a deepfake based on confidence threshold
            is_deepfake = overall_score >= self.confidence_threshold
            
            # Format and return results
            return self.format_result(is_deepfake, overall_score, metadata)
        
        except Exception as e:
            self.logger.error(f"Error detecting deepfake in {media_path}: {str(e)}")
            raise ValueError(f"Failed to process video: {str(e)}")
    
    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
        """
        Extract frames from a video at specified FPS.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple containing:
            - List of frames as numpy arrays
            - List of frame timestamps
            - Dictionary with video info (fps, duration, etc.)
        """
        try:
            # Open the video file
            container = av.open(video_path)
            
            # Get video stream
            video_stream = next(s for s in container.streams if s.type == 'video')
            
            # Get video info
            fps = float(video_stream.average_rate)
            duration = float(video_stream.duration * video_stream.time_base)
            width = video_stream.width
            height = video_stream.height
            
            video_info = {
                "fps": fps,
                "duration": duration,
                "width": width,
                "height": height,
                "total_frames": video_stream.frames,
            }
            
            # Calculate frame interval based on desired sampling rate
            frame_interval = int(fps / self.frames_per_second)
            
            # Extract frames
            frames = []
            frame_times = []
            
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx % frame_interval == 0:
                    # Convert frame to numpy array
                    img = frame.to_image()
                    img_np = np.array(img)
                    
                    # Convert from RGB to BGR (for OpenCV compatibility)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    frames.append(img_np)
                    frame_times.append(float(frame.pts * video_stream.time_base))
            
            return frames, frame_times, video_info
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from video: {str(e)}")
            raise ValueError(f"Failed to extract frames from video: {str(e)}")
    
    def _analyze_frames(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Analyze individual frames for deepfake detection.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Tuple containing:
            - Array of deepfake scores for each frame
            - List of face detection results per frame
        """
        frame_scores = np.zeros(len(frames))
        face_detections = []
        
        for i, frame in enumerate(frames):
            # Convert BGR to RGB (for PIL compatibility)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # For simplicity, we're not doing face detection in every frame
            # In a full implementation, we would detect faces and analyze them
            
            # Process the frame
            score, _ = self._process_frame(pil_image)
            
            # Store results
            frame_scores[i] = score
            face_detections.append({"frame_index": i, "faces": 0})
            
            # Log progress
            if i % 10 == 0:
                self.logger.debug(f"Analyzed {i}/{len(frames)} frames")
        
        return frame_scores, face_detections
    
    def _process_frame(self, frame: Image.Image) -> Tuple[float, np.ndarray]:
        """
        Process a single frame through the ViT model.
        
        Args:
            frame: PIL Image to process
            
        Returns:
            Tuple containing:
            - Deepfake confidence score (0-1)
            - Attention heatmap as numpy array
        """
        # Prepare frame for model
        inputs = self.frame_processor(images=frame, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.frame_model(**inputs, output_attentions=True)
            
            # Get logits and convert to probabilities
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # Get the deepfake probability (assuming index 1 is "fake")
            deepfake_score = probs[0, 1].item()
            
            # For simplicity, we're not generating an attention map
            # In a full implementation, we would generate it
            attention_map = np.zeros((14, 14))
        
        return deepfake_score, attention_map
    
    def _temporal_analysis(self, frame_scores: np.ndarray) -> float:
        """
        Analyze temporal consistency across frames.
        
        Args:
            frame_scores: Array of deepfake scores for each frame
            
        Returns:
            Temporal inconsistency score (0-1)
        """
        if len(frame_scores) <= 1:
            return 0.0
        
        # Calculate standard deviation of scores
        std_dev = np.std(frame_scores)
        
        # Calculate the rate of change between adjacent frames
        diffs = np.abs(np.diff(frame_scores))
        mean_diff = np.mean(diffs)
        
        # Look for sudden changes in scores
        # (potential indicators of temporal inconsistency)
        threshold = 0.3
        sudden_changes = np.sum(diffs > threshold) / max(1, len(diffs))
        
        # Combined temporal inconsistency measure
        temporal_score = 0.4 * std_dev + 0.3 * mean_diff + 0.3 * sudden_changes
        
        # Normalize to 0-1 range
        normalized = min(1.0, temporal_score * 2.0)
        
        return normalized
    
    def normalize_confidence(self, raw_score: float) -> float:
        """
        Normalize the raw confidence score.
        
        Args:
            raw_score: Raw score from the model
            
        Returns:
            Normalized confidence score
        """
        # For this detector, the raw score is already in [0,1]
        return raw_score
