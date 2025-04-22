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
        
        # Auto-detect media type if not specified
        if media_type is None:
            media_type = self._detect_media_type(media_path)
            self.logger.info(f"Auto-detected media type: {media_type}")
        
        # Load detectors if they haven't been loaded yet
        if not self.detectors:
            self.logger.info("Detectors not loaded, loading now...")
            self.load_detectors()
            
        # Validate media_type after ensuring detectors are loaded
        if media_type not in self.detectors:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Process the media with the appropriate detector
        self.logger.info(f"Processing {media_type} file: {media_path}")
        start_time = time.time()
        
        try:
            # Check if we need to use an ensemble detector
            using_ensemble = False
            model_name = model_params.get("model_name", None)
            confidence_threshold = model_params.get("confidence_threshold", 0.5)
            
            if model_name and model_name.lower() == "ensemble":
                self.logger.info(f"Using ensemble detector for {media_type} analysis")
                using_ensemble = True
                
                # Import the appropriate ensemble detector based on media type
                if media_type == "image":
                    from detectors.image_detector.ensemble import ImageEnsembleDetector
                    from detectors.image_detector.vit_detector import ViTImageDetector
                    from detectors.image_detector.deit_detector import DeiTImageDetector
                    from detectors.image_detector.beit_detector import BEITImageDetector
                    from detectors.image_detector.swin_detector import SwinImageDetector
                    
                    # Create individual detector instances
                    self.logger.info("Creating image ensemble with multiple detectors")
                    vit_detector = ViTImageDetector(model_name="google/vit-base-patch16-224", confidence_threshold=confidence_threshold)
                    deit_detector = DeiTImageDetector(model_name="facebook/deit-base-distilled-patch16-224", confidence_threshold=confidence_threshold)
                    beit_detector = BEITImageDetector(model_name="microsoft/beit-base-patch16-224", confidence_threshold=confidence_threshold)
                    swin_detector = SwinImageDetector(model_name="microsoft/swin-base-patch4-window7-224", confidence_threshold=confidence_threshold)
                    
                    # Create and use the ensemble detector
                    detector = ImageEnsembleDetector(
                        detectors=[vit_detector, deit_detector, beit_detector, swin_detector],
                        threshold=confidence_threshold,
                        enable_singularity=True
                    )
                    
                elif media_type == "audio":
                    from detectors.audio_detector.ensemble import AudioEnsembleDetector
                    from detectors.audio_detector.wav2vec_detector import Wav2VecAudioDetector
                    from detectors.audio_detector.xlsr_detector import XLSRAudioDetector
                    
                    # Create individual detector instances
                    self.logger.info("Creating audio ensemble with multiple detectors")
                    wav2vec_detector = Wav2VecAudioDetector(model_name="facebook/wav2vec2-large-960h", confidence_threshold=confidence_threshold)
                    xlsr_detector = XLSRAudioDetector(model_name="facebook/wav2vec2-large-xlsr-53", confidence_threshold=confidence_threshold)
                    
                    # Create and use the ensemble detector
                    detector = AudioEnsembleDetector(
                        detectors=[wav2vec_detector, xlsr_detector],
                        threshold=confidence_threshold,
                        enable_singularity=True
                    )
                    
                elif media_type == "video":
                    from detectors.video_detector.ensemble import VideoEnsembleDetector
                    from detectors.video_detector.genconvit import GenConViTVideoDetector
                    from detectors.video_detector.timesformer import TimeSformerVideoDetector
                    from detectors.video_detector.video_swin import VideoSwinDetector
                    
                    # Create individual detector instances
                    self.logger.info("Creating video ensemble with multiple detectors")
                    genconvit_detector = GenConViTVideoDetector(
                        frame_model_name="google/vit-base-patch16-224", 
                        confidence_threshold=confidence_threshold
                    )
                    timesformer_detector = TimeSformerVideoDetector(
                        model_name="facebook/timesformer-base-finetuned-k400",
                        confidence_threshold=confidence_threshold
                    )
                    video_swin_detector = VideoSwinDetector(
                        model_name="microsoft/swin-base-patch4-window7-224",
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Create and use the ensemble detector
                    detector = VideoEnsembleDetector(
                        detectors=[genconvit_detector, timesformer_detector, video_swin_detector],
                        threshold=confidence_threshold,
                        enable_singularity=True
                    )
            else:
                # Use the regular detector for this media type
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
            
            # Set model name based on whether this is an ensemble or single model
            if using_ensemble:
                result['model'] = "ensemble"
                result['ensemble'] = True
                result['model_count'] = len(detector.detectors) if hasattr(detector, 'detectors') else 0
            else:
                result['model'] = detector.model_name
                result['ensemble'] = False
                
            result['threshold'] = confidence_threshold
            
            # Backward-compatibility: add 'is_deepfake' boolean expected by UI
            if 'prediction' in result:
                result['is_deepfake'] = (result['prediction'].upper() == 'DEEPFAKE')
            elif 'confidence' in result:
                result['is_deepfake'] = (result.get('confidence', 0.0) >= confidence_threshold)
            else:
                # Fallback – assume authentic
                result['is_deepfake'] = False
            
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
