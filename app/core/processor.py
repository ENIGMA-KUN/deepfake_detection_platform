"""
Media processing module for coordinating deepfake detection operations.
"""
import os
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import threading
import hashlib

class MediaProcessor:
    """
    Core processor for handling media detection requests.
    Coordinates between UI, detectors, and result handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the media processor with configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processing_queue = []
        self.results_cache = {}
        self.lock = threading.RLock()
        self.detectors = {}
        
        # Maximum size of the results cache
        self.max_cache_size = config['general'].get('result_cache_size', 100)
        
        # Initialize the temporary directory
        self.temp_dir = config['general'].get('temp_dir', './temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load detector models upon initialization
        try:
            self.load_detectors()
        except Exception as e:
            self.logger.warning(f"Could not load detectors during initialization: {str(e)}")
            self.logger.info("Detectors will be loaded on first detection request")
        
        self.logger.info("MediaProcessor initialized")
        
    def load_detectors(self):
        """
        Load all the detector models based on configuration.
        """
        from detectors.image_detector.vit_detector import ViTImageDetector
        from detectors.audio_detector.wav2vec_detector import Wav2VecAudioDetector
        from detectors.video_detector.genconvit import GenConViTVideoDetector
        
        self.logger.info("Loading detector models...")
        
        # Load image detector
        img_config = self.config['models']['image']
        self.detectors['image'] = ViTImageDetector(
            model_name=img_config['model_name'],
            confidence_threshold=img_config['confidence_threshold']
        )
        
        # Load audio detector
        audio_config = self.config['models']['audio']
        self.detectors['audio'] = Wav2VecAudioDetector(
            model_name=audio_config['model_name'],
            confidence_threshold=audio_config['confidence_threshold']
        )
        
        # Load video detector
        video_config = self.config['models']['video']
        self.detectors['video'] = GenConViTVideoDetector(
            frame_model_name=video_config['frame_model_name'],
            temporal_model_name=video_config['temporal_model_name'],
            confidence_threshold=video_config['confidence_threshold']
        )
        
        self.logger.info("All detector models loaded successfully")
    
    def detect_media(self, media_path: str, media_type: str = None, model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect if the given media is a deepfake.
        
        Args:
            media_path: Path to the media file
            media_type: Type of media (image, audio, video) or None for auto-detection
            model_params: Optional parameters for the model (model_name, confidence_threshold, etc.)
            
        Returns:
            Detection results dictionary
            
        Raises:
            ValueError: If media_type is not valid or media cannot be processed
        """
        # If no model params are provided, use an empty dict
        if model_params is None:
            model_params = {}
            
        # Create a cache key that includes model parameters
        cache_key = f"{media_path}_{hashlib.sha256(str(model_params).encode()).hexdigest()}"
        
        # Check if we already have results for this file with these parameters
        if cache_key in self.results_cache:
            self.logger.info(f"Using cached results for {media_path} with specified parameters")
            return self.results_cache[cache_key]
        
        # If media_type is not provided, attempt to detect it
        if media_type is None:
            media_type = self._detect_media_type(media_path)
        
        if media_type not in self.detectors:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Check if detector models are loaded
        if not self.detectors:
            self.load_detectors()
        
        # Process the media with the appropriate detector
        self.logger.info(f"Processing {media_type} file: {media_path}")
        start_time = time.time()
        
        try:
            detector = self.detectors[media_type]
            
            # Configure the detector with the provided parameters if applicable
            if model_params:
                # Update model name if provided
                if "model_name" in model_params:
                    detector.model_name = model_params["model_name"]
                    # Reset model to force reload with new model name
                    detector.model = None
                
                # Update confidence threshold if provided
                if "confidence_threshold" in model_params:
                    detector.confidence_threshold = model_params["confidence_threshold"]
                    
                self.logger.info(f"Using model: {detector.model_name} with threshold: {detector.confidence_threshold}")
            
            # Detect the media
            result = detector.detect(media_path)
            
            # Add processing time and model info to results
            processing_time = time.time() - start_time
            result['analysis_time'] = processing_time
            result['model'] = detector.model_name
            result['threshold'] = detector.confidence_threshold
            
            # Cache the results
            self._cache_result(cache_key, result)
            
            self.logger.info(f"Processed {media_path} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {media_path}: {str(e)}")
            raise ValueError(f"Failed to process media: {str(e)}")
    
    def _detect_media_type(self, media_path: str) -> str:
        """
        Detect the type of media based on file extension.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            Detected media type: 'image', 'audio', or 'video'
            
        Raises:
            ValueError: If the media type cannot be determined
        """
        _, ext = os.path.splitext(media_path.lower())
        
        # Image extensions
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            return 'image'
        
        # Audio extensions
        elif ext in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            return 'audio'
        
        # Video extensions
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return 'video'
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _cache_result(self, key: str, result: Dict[str, Any]):
        """Cache the detection result."""
        with self.lock:
            # Add to cache
            self.results_cache[key] = result
            
            # If cache exceeds max size, remove oldest entries
            if len(self.results_cache) > self.max_cache_size:
                # Get the oldest entries (first items in dict)
                oldest = list(self.results_cache.keys())[0]
                del self.results_cache[oldest]
                self.logger.debug(f"Removed oldest cache entry: {oldest}")
    
    def get_processing_status(self) -> List[Dict[str, Any]]:
        """
        Get the current status of all processing jobs.
        
        Returns:
            List of processing status dictionaries
        """
        with self.lock:
            return [item.copy() for item in self.processing_queue]
    
    def clear_cache(self):
        """
        Clear the results cache.
        """
        with self.lock:
            self.results_cache.clear()
            self.logger.info("Results cache cleared")
