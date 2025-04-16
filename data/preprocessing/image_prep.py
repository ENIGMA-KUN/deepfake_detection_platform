"""
Image preprocessing module for deepfake detection.
"""
import os
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
import torch
from facenet_pytorch import MTCNN

class ImagePreprocessor:
    """
    Preprocessor for image data to prepare for deepfake detection.
    Handles operations like face extraction, normalization, and augmentation.
    """
    
    def __init__(self, face_detection_threshold: float = 0.9,
                 target_size: Tuple[int, int] = (224, 224),
                 device: str = None):
        """
        Initialize the image preprocessor.
        
        Args:
            face_detection_threshold: Confidence threshold for face detection
            target_size: Target size for output images (height, width)
            device: Device to use for processing ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set parameters
        self.face_detection_threshold = face_detection_threshold
        self.target_size = target_size
        
        # Initialize face detector
        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, face_detection_threshold],  # MTCNN thresholds
            factor=0.7  # Scale factor for the image pyramid
        )
        
        self.logger.info(f"ImagePreprocessor initialized (device: {self.device})")
    
    def preprocess(self, image_path: str, extract_faces: bool = True) -> Dict[str, Any]:
        """
        Preprocess an image for deepfake detection.
        
        Args:
            image_path: Path to the image file
            extract_faces: Whether to extract faces from the image
            
        Returns:
            Dictionary containing:
            - normalized_image: Full preprocessed image
            - faces: List of extracted faces (if extract_faces=True)
            - face_boxes: List of face bounding boxes (if extract_faces=True)
            - metadata: Additional preprocessing metadata
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be processed
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            # Load image
            image = self._load_image(image_path)
            original_size = image.size
            
            # Normalize full image
            normalized_image = self._normalize_image(image)
            
            result = {
                "normalized_image": normalized_image,
                "metadata": {
                    "original_size": original_size,
                    "target_size": self.target_size,
                }
            }
            
            # Extract faces if requested
            if extract_faces:
                faces, face_boxes = self._extract_faces(image)
                result["faces"] = faces
                result["face_boxes"] = face_boxes
                result["metadata"]["faces_detected"] = len(faces)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If the image cannot be loaded
        """
        try:
            # Open image with PIL
            image = Image.open(image_path).convert('RGB')
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def _normalize_image(self, image: Image.Image) -> np.ndarray:
        """
        Normalize an image for model input.
        
        Args:
            image: PIL Image to normalize
            
        Returns:
            Normalized image as numpy array
        """
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        normalized = img_array.astype(np.float32) / 255.0
        
        # Convert to model expected format
        # For most models: (channels, height, width)
        normalized = normalized.transpose(2, 0, 1)
        
        return normalized
    
    def _extract_faces(self, image: Image.Image) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Extract faces from an image.
        
        Args:
            image: PIL Image to extract faces from
            
        Returns:
            Tuple containing:
            - List of normalized face images as numpy arrays
            - List of face bounding boxes [x1, y1, x2, y2, confidence]
        """
        # Detect faces using MTCNN
        boxes, probs = self.face_detector.detect(image)
        
        if boxes is None:
            return [], []
        
        faces = []
        face_boxes = []
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < self.face_detection_threshold:
                continue
                
            # Extract face region with some margin
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Add margin (10% of face size)
            h, w = y2 - y1, x2 - x1
            margin_h, margin_w = int(0.1 * h), int(0.1 * w)
            
            # Ensure boundaries are within image
            y1_margin = max(0, y1 - margin_h)
            x1_margin = max(0, x1 - margin_w)
            y2_margin = min(image.height, y2 + margin_h)
            x2_margin = min(image.width, x2 + margin_w)
            
            # Crop face region
            face = image.crop((x1_margin, y1_margin, x2_margin, y2_margin))
            
            # Normalize face
            normalized_face = self._normalize_image(face)
            
            faces.append(normalized_face)
            face_boxes.append([x1_margin, y1_margin, x2_margin, y2_margin, prob])
        
        return faces, face_boxes
    
    def apply_transformations(self, image: Union[Image.Image, np.ndarray], 
                             transformations: List[str] = None) -> np.ndarray:
        """
        Apply a series of transformations to an image.
        
        Args:
            image: PIL Image or numpy array to transform
            transformations: List of transformation names to apply
                Options: 'flip', 'rotate', 'color_jitter', 'noise'
            
        Returns:
            Transformed image as numpy array
        """
        if transformations is None:
            transformations = []
            
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            # If the array has shape (C, H, W), transpose to (H, W, C)
            if image.shape[0] == 3 and len(image.shape) == 3:
                image = image.transpose(1, 2, 0)
                
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply transformations
        for transform in transformations:
            if transform == 'flip':
                # Horizontal flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
            elif transform == 'rotate':
                # Random rotation -15 to 15 degrees
                angle = np.random.uniform(-15, 15)
                image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
                
            elif transform == 'color_jitter':
                # Color jitter (brightness, contrast, saturation)
                from PIL import ImageEnhance
                
                # Brightness
                factor = np.random.uniform(0.8, 1.2)
                image = ImageEnhance.Brightness(image).enhance(factor)
                
                # Contrast
                factor = np.random.uniform(0.8, 1.2)
                image = ImageEnhance.Contrast(image).enhance(factor)
                
                # Saturation
                factor = np.random.uniform(0.8, 1.2)
                image = ImageEnhance.Color(image).enhance(factor)
                
            elif transform == 'noise':
                # Add Gaussian noise
                img_array = np.array(image).astype(np.float32)
                noise = np.random.normal(0, 5, img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_img)
        
        # Normalize and return as numpy array
        return self._normalize_image(image)

def preprocess_batch(image_paths: List[str], 
                    processor: Optional[ImagePreprocessor] = None,
                    extract_faces: bool = True,
                    batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Preprocess a batch of images.
    
    Args:
        image_paths: List of paths to image files
        processor: Optional ImagePreprocessor instance
        extract_faces: Whether to extract faces from images
        batch_size: Number of images to process at once
        
    Returns:
        List of preprocessing results for each image
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing batch of {len(image_paths)} images")
    
    # Create processor if not provided
    if processor is None:
        processor = ImagePreprocessor()
    
    results = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        for image_path in batch:
            try:
                result = processor.preprocess(image_path, extract_faces=extract_faces)
                results.append(result)
            except Exception as e:
                logger.error(f"Error preprocessing {image_path}: {str(e)}")
                # Add empty result with error message
                results.append({
                    "error": str(e),
                    "image_path": image_path
                })
        
        logger.debug(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return results
