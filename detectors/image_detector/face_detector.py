"""
Face detector module for the Deepfake Detection Platform.
Provides functionality to detect and analyze faces in images.
"""
import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from PIL import Image, ImageDraw

class FaceDetector:
    """
    Face detector class for identifying and analyzing faces in images.
    Uses OpenCV's face detection capabilities.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 model_path: Optional[str] = None):
        """
        Initialize the face detector.
        
        Args:
            confidence_threshold: Threshold for face detection confidence
            model_path: Path to face detection model files (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        
        # Initialize face detector
        try:
            # Try using haarcascade face detector
            module_dir = os.path.dirname(os.path.abspath(__file__))
            cascade_path = os.path.join(module_dir, "..", "..", "models", "cache", "haarcascade_frontalface_default.xml")
            
            # Check if cascade file exists, if not, use OpenCV's default path
            if not os.path.exists(cascade_path):
                self.logger.warning(f"Cascade file not found at {cascade_path}, using OpenCV's default")
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise ValueError("Haar cascade classifier not loaded properly")
            
            self.detection_method = "haarcascade"
            self.logger.info("Initialized face detector using Haar Cascade")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Haar Cascade face detector: {str(e)}")
            
            try:
                # Fall back to DNN face detector
                model_path = os.path.join(module_dir, "..", "..", "models", "cache", "deploy.prototxt")
                weights_path = os.path.join(module_dir, "..", "..", "models", "cache", "res10_300x300_ssd_iter_140000.caffemodel")
                
                # If files don't exist, we'll use mock detection in detect_faces
                if os.path.exists(model_path) and os.path.exists(weights_path):
                    self.face_net = cv2.dnn.readNetFromCaffe(model_path, weights_path)
                    self.detection_method = "dnn"
                    self.logger.info("Initialized face detector using DNN")
                else:
                    self.detection_method = "mock"
                    self.logger.warning("Using mock face detection as fallback")
            except Exception as e2:
                self.logger.error(f"Face detection initialization failed: {str(e2)}")
                self.detection_method = "mock"
                self.logger.warning("Using mock face detection as fallback")
    
    def detect_faces(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image file path, numpy array, or PIL Image
            
        Returns:
            List of face dictionaries with keys:
            - 'bbox': (x, y, width, height)
            - 'confidence': Detection confidence
            - 'landmarks': Facial landmarks if available
        """
        # Load and convert image if needed
        if isinstance(image, str):
            # Load from file path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            # Convert PIL Image to OpenCV format
            img = np.array(image.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Use numpy array directly
            img = image.copy()
        else:
            raise ValueError("Unsupported image type")
        
        # Ensure image is in correct format
        if len(img.shape) == 2:
            # Grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # RGBA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = []
        
        # Detect faces using the selected method
        if self.detection_method == "haarcascade":
            # Use Haar Cascade
            face_rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in face_rects:
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,  # Haar cascade doesn't provide confidence scores
                    'landmarks': None
                })
                
        elif self.detection_method == "dnn":
            # Use DNN-based face detector
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (start_x, start_y, end_x, end_y) = box.astype("int")
                    
                    # Calculate width and height
                    width = end_x - start_x
                    height = end_y - start_y
                    
                    faces.append({
                        'bbox': (start_x, start_y, width, height),
                        'confidence': float(confidence),
                        'landmarks': None
                    })
                    
        else:
            # Mock detection - return a single face in the center of the image
            self.logger.warning("Using mock face detection")
            h, w = img.shape[:2]
            face_w = w // 3
            face_h = h // 3
            x = (w - face_w) // 2
            y = (h - face_h) // 2
            
            faces.append({
                'bbox': (x, y, face_w, face_h),
                'confidence': 0.9,  # Mock confidence
                'landmarks': None
            })
        
        self.logger.info(f"Detected {len(faces)} faces in image")
        return faces
    
    def draw_face_boxes(self, image: Union[np.ndarray, Image.Image], 
                       faces: List[Dict[str, Any]],
                       color: tuple = (0, 255, 0),  # Green by default
                       line_width: int = 2) -> Union[np.ndarray, Image.Image]:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: OpenCV image or PIL Image
            faces: List of face dictionaries from detect_faces()
            color: Box color as BGR tuple for OpenCV or RGB tuple for PIL
            line_width: Width of the bounding box lines
            
        Returns:
            Image with face boxes drawn (same type as input)
        """
        if isinstance(image, np.ndarray):
            # OpenCV image
            img_with_boxes = image.copy()
            
            for face in faces:
                x, y, w, h = face['bbox']
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, line_width)
                
                # Draw confidence if available
                if 'confidence' in face and face['confidence'] is not None:
                    conf_text = f"{face['confidence']:.2f}"
                    cv2.putText(img_with_boxes, conf_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return img_with_boxes
            
        elif isinstance(image, Image.Image):
            # PIL Image
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            for face in faces:
                x, y, w, h = face['bbox']
                draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=line_width)
                
                # Draw confidence if available
                if 'confidence' in face and face['confidence'] is not None:
                    conf_text = f"{face['confidence']:.2f}"
                    draw.text((x, y - 15), conf_text, fill=color)
            
            return img_with_boxes
            
        else:
            raise ValueError("Unsupported image type")

    def get_face_encodings(self, image: Union[str, np.ndarray, Image.Image], 
                          faces: Optional[List[Dict[str, Any]]] = None) -> List[np.ndarray]:
        """
        Get face encodings for detected faces.
        This is a placeholder method that would integrate with a face recognition library.
        
        Args:
            image: Image with faces
            faces: Optional list of face dictionaries, if None faces will be detected
            
        Returns:
            List of face encoding vectors
        """
        # This is a placeholder implementation
        # In a real implementation, you would use a face recognition library
        # like face_recognition or a CNN-based approach
        
        if faces is None:
            faces = self.detect_faces(image)
            
        # Create mock encodings (128-dimensional vectors)
        encodings = []
        for _ in faces:
            # Generate a random encoding vector to simulate face features
            encoding = np.random.normal(0, 1, 128)
            # Normalize to unit length
            encoding = encoding / np.linalg.norm(encoding)
            encodings.append(encoding)
            
        self.logger.info(f"Generated {len(encodings)} face encodings")
        return encodings
