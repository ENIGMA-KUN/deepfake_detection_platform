#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face detection module for image deepfake detection.
Implements face detection as a preprocessing step for more focused analysis.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detectors.base_detector import BaseDetector
from detectors.detection_result import (DeepfakeCategory, DetectionResult,
                                       DetectionStatus, MediaType, Region)
from detectors.detector_utils import (identify_deepfake_category,
                                     load_image, measure_execution_time,
                                     normalize_confidence_score,
                                     normalize_image, resize_image)

# Configure logger
logger = logging.getLogger(__name__)


class FaceDetector(BaseDetector):
    """Face detection for deepfake preprocessing."""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """Initialize the face detector.
        
        Args:
            config: Configuration dictionary
            model_path: Path to face detection model (optional)
        """
        # Face detection parameters
        self.confidence_threshold = config.get("face_confidence_threshold", 0.5)
        self.use_gpu = config.get("use_gpu", True)
        self.detector_backend = config.get("face_detector_backend", "opencv")
        self.max_faces = config.get("max_faces", 5)
        
        # Call parent initializer
        super().__init__(config, model_path)
    
    def _initialize(self) -> None:
        """Initialize the face detector model and resources."""
        if self.detector_backend == "opencv":
            self._initialize_opencv()
        elif self.detector_backend == "dlib":
            self._initialize_dlib()
        elif self.detector_backend == "mtcnn":
            self._initialize_mtcnn()
        else:
            self._initialize_opencv()  # Default to OpenCV
    
    def _initialize_opencv(self) -> None:
        """Initialize OpenCV face detector."""
        try:
            import cv2
            
            # Load face detection model
            face_cascade_path = self.model_path
            if face_cascade_path is None:
                # Use default cascade
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            if self.face_cascade.empty():
                raise ValueError(f"Failed to load cascade classifier from {face_cascade_path}")
            
            logger.info(f"OpenCV face detector initialized using {face_cascade_path}")
            
        except ImportError as e:
            logger.error(f"Failed to import OpenCV: {e}")
            raise ImportError("OpenCV (cv2) is required for face detection")
    
    def _initialize_dlib(self) -> None:
        """Initialize dlib face detector."""
        try:
            import dlib
            
            # Use HOG-based detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Load face landmark predictor if path provided
            self.shape_predictor = None
            if self.model_path and os.path.exists(self.model_path):
                self.shape_predictor = dlib.shape_predictor(self.model_path)
            
            logger.info("dlib face detector initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import dlib: {e}")
            raise ImportError("dlib is required for this face detection method")
    
    def _initialize_mtcnn(self) -> None:
        """Initialize MTCNN face detector."""
        try:
            from mtcnn import MTCNN
            
            # Choose device
            device = 'cuda' if self.use_gpu else 'cpu'
            
            # Initialize detector
            self.face_detector = MTCNN(min_face_size=20, scale_factor=0.709)
            
            logger.info("MTCNN face detector initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import MTCNN: {e}")
            raise ImportError("MTCNN is required for this face detection method")
    
    @measure_execution_time
    def detect(self, media_path: str) -> DetectionResult:
        """Run face detection on the provided image.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            DetectionResult with detection results
        """
        # Create a new result object
        result = DetectionResult.create_new(MediaType.IMAGE, os.path.basename(media_path))
        result.model_used = f"FaceDetector_{self.detector_backend}"
        result.status = DetectionStatus.PROCESSING
        
        try:
            # Validate media
            if not self.validate_media(media_path):
                result.status = DetectionStatus.FAILED
                result.metadata["error"] = "Invalid image file"
                return result
            
            # Preprocess image
            image = self.preprocess(media_path)
            
            # Detect faces
            faces = self._detect_faces(image)
            
            # Create regions for detected faces
            regions = self._create_regions_from_faces(faces, image.shape)
            
            # Determine confidence based on number and size of faces
            if regions:
                # More faces with higher individual confidence increase overall confidence
                face_confidences = [region.confidence for region in regions]
                confidence = sum(face_confidences) / len(face_confidences)
                
                # This is just face detection, not deepfake detection yet
                is_deepfake = False
            else:
                confidence = 0.0
                is_deepfake = False
            
            # Update result
            result.is_deepfake = is_deepfake
            result.confidence_score = confidence
            result.regions = regions
            result.status = DetectionStatus.COMPLETED
            
            # Add metadata
            result.metadata["num_faces"] = len(regions)
            result.metadata["image_size"] = f"{image.shape[1]}x{image.shape[0]}"
            
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            result.status = DetectionStatus.FAILED
            result.metadata["error"] = str(e)
        
        return result
    
    def preprocess(self, media_path: str) -> np.ndarray:
        """Preprocess the image for face detection.
        
        Args:
            media_path: Path to the image file
            
        Returns:
            Image as numpy array
        """
        # Load image
        image = load_image(media_path)
        
        # Convert to grayscale for faster detection if using OpenCV
        if self.detector_backend == "opencv":
            import cv2
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return gray
        
        return image
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image using the selected backend.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of face detection results
        """
        if self.detector_backend == "opencv":
            return self._detect_faces_opencv(image)
        elif self.detector_backend == "dlib":
            return self._detect_faces_dlib(image)
        elif self.detector_backend == "mtcnn":
            return self._detect_faces_mtcnn(image)
        else:
            return self._detect_faces_opencv(image)  # Default to OpenCV
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV.
        
        Args:
            image: Grayscale image as numpy array
            
        Returns:
            List of detected faces with coordinates and confidence
        """
        import cv2
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        result = []
        for i, (x, y, w, h) in enumerate(faces):
            # OpenCV doesn't provide confidence scores directly
            # Use a default confidence or calculate based on face size
            confidence = 0.7  # Default confidence
            
            result.append({
                "id": i,
                "box": [x, y, w, h],
                "confidence": confidence
            })
            
            if len(result) >= self.max_faces:
                break
        
        return result
    
    def _detect_faces_dlib(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using dlib.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of detected faces with coordinates and confidence
        """
        import dlib
        
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            gray = dlib.rgb_to_grayscale(image)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_detector(gray, 1)
        
        result = []
        for i, face in enumerate(faces):
            # dlib doesn't provide confidence scores directly
            confidence = 0.8  # Default confidence
            
            # Extract coordinates
            x, y = face.left(), face.top()
            w, h = face.width(), face.height()
            
            result.append({
                "id": i,
                "box": [x, y, w, h],
                "confidence": confidence
            })
            
            if len(result) >= self.max_faces:
                break
        
        return result
    
    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            List of detected faces with coordinates and confidence
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        result = []
        for i, face in enumerate(faces):
            if face['confidence'] >= self.confidence_threshold:
                result.append({
                    "id": i,
                    "box": face['box'],
                    "confidence": face['confidence'],
                    "keypoints": face['keypoints']
                })
            
            if len(result) >= self.max_faces:
                break
        
        return result
    
    def _create_regions_from_faces(self, faces: List[Dict[str, Any]], 
                                  image_shape: Tuple[int, ...]) -> List[Region]:
        """Convert face detection results to Region objects.
        
        Args:
            faces: List of face detection results
            image_shape: Shape of the original image as (height, width, ...)
            
        Returns:
            List of Region objects
        """
        height, width = image_shape[:2]
        
        regions = []
        for face in faces:
            # Extract coordinates
            x, y, w, h = face["box"]
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Skip invalid regions
            if w <= 0 or h <= 0:
                continue
            
            # Normalize coordinates
            norm_x = x / width
            norm_y = y / height
            norm_width = w / width
            norm_height = h / height
            
            regions.append(Region(
                x=norm_x,
                y=norm_y,
                width=norm_width,
                height=norm_height,
                confidence=face["confidence"],
                label=f"face_{face['id']}",
                category=None  # No deepfake category for face detection
            ))
        
        return regions
    
    def postprocess(self, model_output: Any) -> Tuple[bool, float, List[Region]]:
        """Not used for face detection as a standalone detector."""
        raise NotImplementedError("Face detector is not intended to be used as a standalone deepfake detector")
    
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
        
        # Try opening the image to ensure it's valid
        try:
            image = load_image(media_path)
            # Check image dimensions
            if len(image.shape) < 2:
                logger.error(f"Invalid image dimensions: {image.shape}")
                return False
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