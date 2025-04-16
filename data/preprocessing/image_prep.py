# data/preprocessing/image_prep.py
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Class for preprocessing image data for deepfake detection."""
    
    def __init__(self, config=None):
        """
        Initialize the image preprocessor.
        
        Args:
            config (dict, optional): Configuration parameters for preprocessing
        """
        self.config = config or {}
        self.target_size = self.config.get('target_size', (224, 224))
        self.normalize = self.config.get('normalize', True)
        self.face_detection = self.config.get('face_detection', False)
        
        # Initialize face detector if enabled
        if self.face_detection:
            self._init_face_detector()
            
        # Default normalization values (ImageNet)
        self.mean = self.config.get('mean', [0.485, 0.456, 0.406])
        self.std = self.config.get('std', [0.229, 0.224, 0.225])
        
        # Create transform pipeline
        self._create_transforms()
    
    def _init_face_detector(self):
        """Initialize the face detector model."""
        try:
            # Load OpenCV's face detector
            face_cascade_path = self.config.get(
                'face_cascade_path', 
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Check if face detector was loaded successfully
            if self.face_cascade.empty():
                logger.warning("Face detector could not be loaded. Face detection disabled.")
                self.face_detection = False
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            self.face_detection = False
    
    def _create_transforms(self):
        """Create the transformation pipeline for preprocessing."""
        transform_list = []
        
        # Resize transform
        transform_list.append(transforms.Resize(self.target_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if enabled
        if self.normalize:
            transform_list.append(transforms.Normalize(self.mean, self.std))
        
        self.transform = transforms.Compose(transform_list)
    
    def preprocess(self, image_path, return_tensor=True):
        """
        Preprocess an image for deepfake detection.
        
        Args:
            image_path (str): Path to the image file
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            preprocessed_image: Preprocessed image as tensor or NumPy array
            meta_info (dict): Additional information about preprocessing
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Store original dimensions
            original_width, original_height = image.size
            meta_info = {
                'original_size': (original_width, original_height),
                'faces_detected': 0,
                'preprocessing_steps': []
            }
            meta_info['preprocessing_steps'].append('load_image')
            
            # Detect faces if enabled
            if self.face_detection:
                image, face_info = self._detect_and_extract_faces(image)
                meta_info.update(face_info)
                meta_info['preprocessing_steps'].append('face_detection')
            
            # Apply transforms
            if return_tensor:
                processed_image = self.transform(image)
                meta_info['preprocessing_steps'].append('transform_to_tensor')
                
                if self.normalize:
                    meta_info['preprocessing_steps'].append('normalize')
            else:
                # Resize without normalization
                image = image.resize(self.target_size)
                processed_image = np.array(image)
                meta_info['preprocessing_steps'].append('resize')
            
            return processed_image, meta_info
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def _detect_and_extract_faces(self, pil_image):
        """
        Detect and extract faces from an image.
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with extracted face or original if no face found
            dict: Information about detected faces
        """
        # Convert PIL image to OpenCV format (RGB to BGR)
        cv_image = np.array(pil_image)
        cv_image = cv_image[:, :, ::-1].copy()
        
        # Detect faces
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_info = {
            'faces_detected': len(faces),
            'face_regions': []
        }
        
        # If faces found, extract the largest one
        if len(faces) > 0:
            # Find largest face by area
            largest_face_idx = np.argmax([w * h for (x, y, w, h) in faces])
            x, y, w, h = faces[largest_face_idx]
            
            # Add padding around face (20%)
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            # Ensure coordinates are within image bounds
            x_start = max(0, x - padding_x)
            y_start = max(0, y - padding_y)
            x_end = min(cv_image.shape[1], x + w + padding_x)
            y_end = min(cv_image.shape[0], y + h + padding_y)
            
            # Extract face region
            face_region = cv_image[y_start:y_end, x_start:x_end]
            
            # Store face location
            face_info['face_regions'].append({
                'x': x_start,
                'y': y_start,
                'width': x_end - x_start,
                'height': y_end - y_start,
                'is_largest': True
            })
            
            # Convert back to PIL image
            face_pil = Image.fromarray(face_region[:, :, ::-1])  # BGR to RGB
            return face_pil, face_info
        
        # If no face found, return original image
        return pil_image, face_info
    
    def batch_preprocess(self, image_paths, return_tensor=True):
        """
        Preprocess a batch of images.
        
        Args:
            image_paths (list): List of paths to image files
            return_tensor (bool): If True, return torch tensor; otherwise NumPy array
            
        Returns:
            list: List of preprocessed images
            list: List of meta info dictionaries
        """
        preprocessed_images = []
        meta_infos = []
        
        for image_path in image_paths:
            try:
                processed_image, meta_info = self.preprocess(image_path, return_tensor)
                preprocessed_images.append(processed_image)
                meta_infos.append(meta_info)
            except Exception as e:
                logger.error(f"Error preprocessing image {image_path}: {e}")
                # Skip failed images
                continue
        
        # If return tensor and we have images, stack them
        if return_tensor and preprocessed_images:
            preprocessed_images = torch.stack(preprocessed_images)
        
        return preprocessed_images, meta_infos
    
    def apply_ela(self, image_path, quality=90, scale=10):
        """
        Apply Error Level Analysis (ELA) to image.
        
        Args:
            image_path (str): Path to the image file
            quality (int): JPEG compression quality for ELA
            scale (int): Scale factor for ELA
            
        Returns:
            numpy.ndarray: ELA processed image
            dict: Meta information
        """
        meta_info = {
            'preprocessing_steps': ['ela'],
            'ela_quality': quality,
            'ela_scale': scale
        }
        
        try:
            # Load original image
            original = Image.open(image_path).convert('RGB')
            
            # Create temporary filename for saving compressed version
            filename, ext = os.path.splitext(image_path)
            compressed_path = f"{filename}_compressed_temp.jpg"
            
            # Save image with specified quality
            original.save(compressed_path, 'JPEG', quality=quality)
            
            # Open compressed image
            compressed = Image.open(compressed_path).convert('RGB')
            
            # Calculate difference and scale
            ela_image = ImageChops.difference(original, compressed)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale_factor = 255.0 / max_diff
            
            # Scale difference image
            ela_image = ImageChops.multiply(ela_image, Image.new('RGB', ela_image.size, (scale_factor, scale_factor, scale_factor)))
            
            # Remove temporary file
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            
            # Convert to numpy array
            ela_array = np.array(ela_image)
            
            return ela_array, meta_info
            
        except Exception as e:
            logger.error(f"Error applying ELA to image {image_path}: {e}")
            # Return original image if ELA fails
            return np.array(Image.open(image_path).convert('RGB')), meta_info