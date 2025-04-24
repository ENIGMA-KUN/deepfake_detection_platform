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
        from detectors.image_detector.beit_detector import BEITImageDetector
        from detectors.image_detector.deit_detector import DeiTImageDetector
        from detectors.image_detector.swin_detector import SwinImageDetector
        from detectors.ensemble_detector import ImageEnsembleDetector
        from detectors.audio_detector.wav2vec_detector import Wav2VecAudioDetector
        from detectors.video_detector.genconvit import GenConViTVideoDetector
        
        self.logger.info("Loading detector models...")
        
        # Load individual image detectors
        img_config = self.config['models']['image']
        vit_detector = ViTImageDetector(
            model_name="vit",
            confidence_threshold=img_config['confidence_threshold']
        )
        beit_detector = BEITImageDetector(
            model_name="beit",
            confidence_threshold=img_config['confidence_threshold']
        )
        deit_detector = DeiTImageDetector(
            model_name="deit",
            confidence_threshold=img_config['confidence_threshold']
        )
        swin_detector = SwinImageDetector(
            model_name="swin",
            confidence_threshold=img_config['confidence_threshold']
        )
        
        # Create image ensemble detector with all models
        image_ensemble = ImageEnsembleDetector(
            detectors=[vit_detector, beit_detector, deit_detector, swin_detector],
            weights=[0.25, 0.30, 0.20, 0.25],  # Weights can be adjusted based on model performance
            threshold=img_config['confidence_threshold'],
            enable_singularity=True  # Enable advanced features
        )
        
        # Store both individual detectors and ensemble
        self.detectors['image'] = {
            'vit': vit_detector,
            'beit': beit_detector,
            'deit': deit_detector,
            'swin': swin_detector,
            'ensemble': image_ensemble,
            'default': vit_detector  # Default to ViT if not specified
        }
        
        # Load audio detector
        audio_config = self.config['models']['audio']
        self.detectors['audio'] = {
            'wav2vec': Wav2VecAudioDetector(
                model_name=audio_config['model_name'],
                confidence_threshold=audio_config['confidence_threshold']
            ),
            'default': Wav2VecAudioDetector(
                model_name=audio_config['model_name'],
                confidence_threshold=audio_config['confidence_threshold']
            )
        }
        
        # Load video detector
        video_config = self.config['models']['video']
        self.detectors['video'] = {
            'genconvit': GenConViTVideoDetector(
                frame_model_name=video_config['frame_model_name'],
                temporal_model_name=video_config['temporal_model_name'],
                confidence_threshold=video_config['confidence_threshold']
            ),
            'default': GenConViTVideoDetector(
                frame_model_name=video_config['frame_model_name'],
                temporal_model_name=video_config['temporal_model_name'],
                confidence_threshold=video_config['confidence_threshold']
            )
        }
        
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
        
        # Debug logging
        self.logger.info(f"===== DETECTION REQUEST =====")
        self.logger.info(f"Media path: {media_path}")
        self.logger.info(f"Media type: {media_type or 'auto-detect'}")
        self.logger.info(f"Model params: {model_params}")
            
        # Create a cache key that includes model parameters
        cache_key = f"{media_path}_{hashlib.sha256(str(model_params).encode()).hexdigest()}"
        
        # Check if we already have results for this file with these parameters
        if cache_key in self.results_cache:
            self.logger.info(f"Using cached results for {media_path} with specified parameters")
            return self.results_cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Determine the media type if not provided
            if media_type is None:
                media_type = self._detect_media_type(media_path)
                
            self.logger.info(f"Processing {media_type} file: {media_path}")
            
            # Ensure media file exists
            if not os.path.exists(media_path):
                raise ValueError(f"File not found: {media_path}")
            
            # Get the appropriate detector based on media type and model params
            if media_type not in self.detectors:
                raise ValueError(f"Unsupported media type: {media_type}")
                
            # Get model name or use default
            model_name = model_params.get("model_name", "default")
            self.logger.info(f"Selected model: {model_name}")
            
            # Get the detector for the specified model
            media_detectors = self.detectors[media_type]
            
            # Check if we have the requested model, otherwise use default
            if model_name not in media_detectors:
                self.logger.warning(f"Model {model_name} not found, using default")
                detector = media_detectors["default"]
            else:
                detector = media_detectors[model_name]
                
            self.logger.info(f"Using detector: {detector.__class__.__name__}")
            
            # Configure detector based on model params
            confidence_threshold = model_params.get("confidence_threshold", None)
            using_ensemble = model_name == "ensemble"
            
            # Handle singularity mode for ensemble detectors
            if using_ensemble and model_params.get("enable_singularity", False):
                # Enable singularity mode if the detector supports it
                if hasattr(detector, 'enable_singularity'):
                    detector.enable_singularity = True
                    self.logger.info("Enabling Singularity Mode for enhanced analysis")
            
            if confidence_threshold is not None:
                # Apply confidence threshold (ensure it's a float between 0 and 1)
                if isinstance(confidence_threshold, (int, float)):
                    # If it's a percentage (0-100), convert to decimal
                    if confidence_threshold > 1:
                        confidence_threshold = confidence_threshold / 100.0
                    
                    # Update confidence threshold if provided
                    detector.confidence_threshold = confidence_threshold
                
                self.logger.info(f"Using threshold: {confidence_threshold}")
            
            # Detect the media
            self.logger.info(f"Calling detect method on {detector.__class__.__name__}")
            result = detector.detect(media_path)
            self.logger.info(f"Raw detection result: {result}")
            
            # Add processing time and model info to results
            processing_time = time.time() - start_time
            result['analysis_time'] = processing_time
            
            # Set model name based on whether this is an ensemble or single model
            if using_ensemble:
                result['model'] = "ensemble"
                result['ensemble'] = True
                result['model_count'] = len(detector.detectors) if hasattr(detector, 'detectors') else 0
                # Add singularity flag if enabled
                if hasattr(detector, 'enable_singularity') and detector.enable_singularity:
                    result['singularity_mode'] = True
            else:
                result['model'] = model_name
                result['ensemble'] = False
                
            result['threshold'] = detector.confidence_threshold
            
            # Backward-compatibility: add 'is_deepfake' boolean expected by UI
            if 'prediction' in result:
                result['is_deepfake'] = (result['prediction'].upper() == 'DEEPFAKE')
            elif 'confidence' in result:
                result['is_deepfake'] = (result.get('confidence', 0.0) >= detector.confidence_threshold)
                self.logger.info(f"Determining is_deepfake: confidence {result.get('confidence', 0.0)} >= threshold {detector.confidence_threshold} = {result['is_deepfake']}")
            else:
                # Fallback – assume authentic
                self.logger.warning("No confidence or prediction in result, defaulting to authentic")
                result['is_deepfake'] = False
            
            # Cache the results
            self._cache_result(cache_key, result)
            
            self.logger.info(f"Processed {media_path} in {processing_time:.2f}s")
            self.logger.info(f"Final result: is_deepfake={result.get('is_deepfake')}, confidence={result.get('confidence')}")
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
