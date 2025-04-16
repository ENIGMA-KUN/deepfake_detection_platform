"""
File handler module for Deepfake Detection Platform.

This module contains utilities for file validation, saving, and loading.
It handles different media types (images, audio, and video) and provides
functions for checking file types, sizes, and integrity, as well as
saving and loading files for processing.
"""

import os
import mimetypes
import magic
import hashlib
from pathlib import Path
import logging
import yaml
import cv2
import numpy as np
from PIL import Image
import librosa
import moviepy.editor as mp
from typing import Dict, Tuple, List, Union, Optional

# Initialize logger
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)

config = load_config()

# File type validation
def is_valid_file_type(file_path: str, allowed_types: List[str]) -> bool:
    """
    Check if the file type is allowed.
    
    Args:
        file_path: Path to the file
        allowed_types: List of allowed mime types
        
    Returns:
        bool: True if file type is allowed, False otherwise
    """
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    logger.info(f"File {file_path} has type {file_type}")
    return file_type in allowed_types

# File size validation
def is_valid_file_size(file_path: str, max_size_mb: int) -> bool:
    """
    Check if the file size is under the maximum allowed size.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        bool: True if file size is valid, False otherwise
    """
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    logger.info(f"File {file_path} has size {file_size:.2f} MB")
    return file_size <= max_size_mb

# File integrity check
def check_file_integrity(file_path: str) -> str:
    """
    Calculate SHA-256 hash of the file to check integrity.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: SHA-256 hash of the file
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Media-specific validation functions
def validate_image(file_path: str) -> Tuple[bool, str]:
    """
    Validate an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    try:
        # Check if file type is allowed
        allowed_types = config.get("allowed_image_types", [
            "image/jpeg", "image/png", "image/gif", "image/bmp"
        ])
        if not is_valid_file_type(file_path, allowed_types):
            return False, f"Invalid image format. Allowed formats: {', '.join(t.split('/')[-1].upper() for t in allowed_types)}"
        
        # Check file size
        max_size = config.get("max_image_size_mb", 10)
        if not is_valid_file_size(file_path, max_size):
            return False, f"Image size exceeds the maximum allowed size of {max_size} MB"
        
        # Try to open the image to verify it's not corrupted
        try:
            img = Image.open(file_path)
            img.verify()  # Verify that it's an image
            img.close()
            
            img = Image.open(file_path)
            img.load()  # Load it to make sure it's not truncated
            
            # Check dimensions
            width, height = img.size
            min_dim = config.get("min_image_dimension", 32)
            max_dim = config.get("max_image_dimension", 4096)
            if width < min_dim or height < min_dim:
                return False, f"Image dimensions too small. Minimum dimension: {min_dim}px"
            if width > max_dim or height > max_dim:
                return False, f"Image dimensions too large. Maximum dimension: {max_dim}px"
                
            return True, "Image validation successful"
        except Exception as e:
            return False, f"Image appears to be corrupted: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error validating image {file_path}: {str(e)}")
        return False, f"Image validation failed: {str(e)}"

def validate_audio(file_path: str) -> Tuple[bool, str]:
    """
    Validate an audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    try:
        # Check if file type is allowed
        allowed_types = config.get("allowed_audio_types", [
            "audio/wav", "audio/mp3", "audio/mpeg", "audio/flac"
        ])
        if not is_valid_file_type(file_path, allowed_types):
            return False, f"Invalid audio format. Allowed formats: {', '.join(t.split('/')[-1].upper() for t in allowed_types)}"
        
        # Check file size
        max_size = config.get("max_audio_size_mb", 50)
        if not is_valid_file_size(file_path, max_size):
            return False, f"Audio size exceeds the maximum allowed size of {max_size} MB"
        
        # Try to load the audio to verify it's not corrupted
        try:
            audio, sr = librosa.load(file_path, sr=None, duration=5)  # Load just first 5 seconds to check
            
            # Check duration
            duration = librosa.get_duration(y=audio, sr=sr)
            min_duration = config.get("min_audio_duration_sec", 1)
            max_duration = config.get("max_audio_duration_sec", 600)  # 10 minutes
            
            if duration < min_duration:
                return False, f"Audio duration too short. Minimum duration: {min_duration} seconds"
            
            # Estimate full duration without loading the entire file
            file_size = os.path.getsize(file_path)
            bytes_per_sec = file_size / duration  # Based on the 5 seconds we loaded
            estimated_duration = file_size / bytes_per_sec
            
            if estimated_duration > max_duration:
                return False, f"Audio duration too long. Maximum duration: {max_duration} seconds"
                
            return True, "Audio validation successful"
        except Exception as e:
            return False, f"Audio appears to be corrupted: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error validating audio {file_path}: {str(e)}")
        return False, f"Audio validation failed: {str(e)}"

def validate_video(file_path: str) -> Tuple[bool, str]:
    """
    Validate a video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    try:
        # Check if file type is allowed
        allowed_types = config.get("allowed_video_types", [
            "video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"
        ])
        if not is_valid_file_type(file_path, allowed_types):
            return False, f"Invalid video format. Allowed formats: {', '.join(t.split('/')[-1].upper() for t in allowed_types)}"
        
        # Check file size
        max_size = config.get("max_video_size_mb", 500)
        if not is_valid_file_size(file_path, max_size):
            return False, f"Video size exceeds the maximum allowed size of {max_size} MB"
        
        # Try to open the video to verify it's not corrupted
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, "Could not open video file"
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Check dimensions
            min_dim = config.get("min_video_dimension", 128)
            max_dim = config.get("max_video_dimension", 1920)
            if width < min_dim or height < min_dim:
                cap.release()
                return False, f"Video dimensions too small. Minimum dimension: {min_dim}px"
            if width > max_dim or height > max_dim:
                cap.release()
                return False, f"Video dimensions too large. Maximum dimension: {max_dim}px"
            
            # Check duration
            duration = frame_count / fps if fps > 0 else 0
            min_duration = config.get("min_video_duration_sec", 1)
            max_duration = config.get("max_video_duration_sec", 300)  # 5 minutes
            
            if duration < min_duration:
                cap.release()
                return False, f"Video duration too short. Minimum duration: {min_duration} seconds"
            if duration > max_duration:
                cap.release()
                return False, f"Video duration too long. Maximum duration: {max_duration} seconds"
            
            # Check if video can be read
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False, "Could not read video frames"
            
            cap.release()
            return True, "Video validation successful"
        except Exception as e:
            return False, f"Video appears to be corrupted: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error validating video {file_path}: {str(e)}")
        return False, f"Video validation failed: {str(e)}"

def validate_file(file_path: str, media_type: str = None) -> Tuple[bool, str, str]:
    """
    Validate a file based on its media type.
    
    Args:
        file_path: Path to the file
        media_type: Type of media ('image', 'audio', 'video'). If None, will be detected.
        
    Returns:
        Tuple[bool, str, str]: (is_valid, message, detected_media_type)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist", ""
    
    if media_type is None:
        # Try to detect media type based on mime type
        mime = magic.Magic(mime=True)
        file_mime = mime.from_file(file_path)
        
        if file_mime.startswith('image/'):
            media_type = 'image'
        elif file_mime.startswith('audio/'):
            media_type = 'audio'
        elif file_mime.startswith('video/'):
            media_type = 'video'
        else:
            return False, f"Unsupported media type: {file_mime}", ""
    
    # Validate based on media type
    if media_type == 'image':
        is_valid, message = validate_image(file_path)
    elif media_type == 'audio':
        is_valid, message = validate_audio(file_path)
    elif media_type == 'video':
        is_valid, message = validate_video(file_path)
    else:
        return False, f"Unknown media type: {media_type}", ""
    
    return is_valid, message, media_type

# File saving functions
def get_storage_path(media_type: str) -> Path:
    """
    Get the storage path for a specific media type.
    
    Args:
        media_type: Type of media ('image', 'audio', 'video')
        
    Returns:
        Path: Directory path for storage
    """
    base_dir = Path(config.get("storage_base_dir", "storage"))
    if not base_dir.is_absolute():
        # Make relative to the project root
        base_dir = Path(__file__).parent.parent.parent / base_dir
    
    media_dir = base_dir / media_type
    os.makedirs(media_dir, exist_ok=True)
    
    return media_dir

def generate_unique_filename(original_filename: str, media_type: str) -> str:
    """
    Generate a unique filename for storage.
    
    Args:
        original_filename: Original filename
        media_type: Type of media
        
    Returns:
        str: Unique filename
    """
    timestamp = int(os.path.getmtime(original_filename) if os.path.exists(original_filename) 
                   else os.path.getctime(original_filename))
    file_hash = hashlib.md5(f"{original_filename}_{timestamp}".encode()).hexdigest()[:8]
    extension = os.path.splitext(original_filename)[1]
    
    return f"{media_type}_{timestamp}_{file_hash}{extension}"

def save_uploaded_file(file_path: str, media_type: str = None) -> Tuple[bool, str, Dict]:
    """
    Save an uploaded file to the appropriate storage location.
    
    Args:
        file_path: Path to the temporary uploaded file
        media_type: Type of media ('image', 'audio', 'video'). If None, will be detected.
        
    Returns:
        Tuple[bool, str, Dict]: (success, message, file_info)
    """
    try:
        # Validate the file first
        is_valid, message, detected_media_type = validate_file(file_path, media_type)
        if not is_valid:
            return False, message, {}
        
        # Use detected media type if not provided
        if media_type is None:
            media_type = detected_media_type
        
        # Generate unique filename and determine storage path
        unique_filename = generate_unique_filename(file_path, media_type)
        storage_dir = get_storage_path(media_type)
        target_path = storage_dir / unique_filename
        
        # Copy the file to the storage location
        import shutil
        shutil.copy2(file_path, target_path)
        
        # Generate file metadata
        file_size = os.path.getsize(target_path) / (1024 * 1024)  # Size in MB
        file_hash = check_file_integrity(target_path)
        
        # Get media-specific metadata
        media_metadata = {}
        if media_type == 'image':
            img = Image.open(target_path)
            media_metadata = {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            }
            img.close()
        elif media_type == 'audio':
            audio, sr = librosa.load(target_path, sr=None, duration=10)
            duration = librosa.get_duration(y=audio, sr=sr)
            media_metadata = {
                'sample_rate': sr,
                'duration': duration,
                'channels': 1 if len(audio.shape) == 1 else audio.shape[1]
            }
        elif media_type == 'video':
            cap = cv2.VideoCapture(str(target_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            media_metadata = {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
        
        # Compile file information
        file_info = {
            'original_filename': os.path.basename(file_path),
            'stored_filename': unique_filename,
            'storage_path': str(target_path),
            'media_type': media_type,
            'file_size_mb': file_size,
            'hash': file_hash,
            'timestamp': os.path.getmtime(target_path),
            'metadata': media_metadata
        }
        
        logger.info(f"File saved successfully: {target_path}")
        return True, "File saved successfully", file_info
        
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {str(e)}")
        return False, f"Error saving file: {str(e)}", {}

# File loading functions
def load_file(file_info: Dict) -> Tuple[bool, str, Union[np.ndarray, np.ndarray, mp.VideoFileClip]]:
    """
    Load a file based on its file_info for processing.
    
    Args:
        file_info: Dictionary containing file information
        
    Returns:
        Tuple[bool, str, Union[np.ndarray, np.ndarray, mp.VideoFileClip]]: 
            (success, message, loaded_content)
    """
    try:
        file_path = file_info.get('storage_path')
        if not file_path or not os.path.exists(file_path):
            return False, "File not found", None
        
        media_type = file_info.get('media_type')
        if not media_type:
            return False, "Media type not specified", None
        
        # Load based on media type
        if media_type == 'image':
            # Load image using OpenCV for processing
            img = cv2.imread(file_path)
            if img is None:
                return False, "Failed to load image", None
            return True, "Image loaded successfully", img
            
        elif media_type == 'audio':
            # Load audio using librosa
            audio, sr = librosa.load(file_path, sr=None)
            if audio is None:
                return False, "Failed to load audio", None
            return True, "Audio loaded successfully", (audio, sr)
            
        elif media_type == 'video':
            # Load video using moviepy
            try:
                video = mp.VideoFileClip(file_path)
                return True, "Video loaded successfully", video
            except Exception as e:
                # Fallback to OpenCV if moviepy fails
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    return False, f"Failed to load video: {str(e)}", None
                return True, "Video loaded successfully (OpenCV)", cap
            
        else:
            return False, f"Unsupported media type: {media_type}", None
            
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return False, f"Error loading file: {str(e)}", None

def get_file_info(file_id: str) -> Dict:
    """
    Retrieve file information by ID from the database or file system.
    
    Args:
        file_id: Unique file identifier
        
    Returns:
        Dict: File information dictionary
    """
    # Implementation depends on how file info is stored
    # This is a placeholder that would be replaced with actual database or file system lookup
    # For now, we'll assume file_id is the same as the stored filename
    
    # Search in all media directories
    for media_type in ['image', 'audio', 'video']:
        storage_dir = get_storage_path(media_type)
        for file_path in storage_dir.glob(f"*{file_id}*"):
            # Found a matching file, create a file_info dictionary
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            file_hash = check_file_integrity(file_path)
            
            return {
                'original_filename': file_path.name,
                'stored_filename': file_path.name,
                'storage_path': str(file_path),
                'media_type': media_type,
                'file_size_mb': file_size,
                'hash': file_hash,
                'timestamp': os.path.getmtime(file_path)
            }
    
    return {}

# Temporary file management
class TempFileManager:
    """
    Manages temporary files created during processing.
    
    This class provides utilities for creating, tracking, and cleaning up
    temporary files used in the detection process.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the temporary file manager.
        
        Args:
            base_dir: Base directory for temporary files. If None, uses system temp directory.
        """
        if base_dir is None:
            import tempfile
            self.base_dir = Path(tempfile.gettempdir()) / "deepfake_detection"
        else:
            self.base_dir = Path(base_dir)
        
        # Create the base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Keep track of created temporary files
        self.temp_files = []
        
        logger.info(f"Temporary file manager initialized with base directory: {self.base_dir}")
    
    def create_temp_dir(self, prefix: str = "") -> Path:
        """
        Create a temporary directory for processing.
        
        Args:
            prefix: Prefix for the directory name
            
        Returns:
            Path: Path to the created temporary directory
        """
        import uuid
        dir_name = f"{prefix}_{uuid.uuid4().hex}"
        temp_dir = self.base_dir / dir_name
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def create_temp_file(self, prefix: str = "", suffix: str = "", directory: Optional[Path] = None) -> Path:
        """
        Create a temporary file.
        
        Args:
            prefix: Prefix for the filename
            suffix: Suffix for the filename (e.g., file extension)
            directory: Directory to create the file in. If None, uses the base temp directory.
            
        Returns:
            Path: Path to the created temporary file
        """
        import uuid
        
        if directory is None:
            directory = self.base_dir
        
        os.makedirs(directory, exist_ok=True)
        
        file_name = f"{prefix}_{uuid.uuid4().hex}{suffix}"
        temp_file = directory / file_name
        
        # Track the file for cleanup
        self.temp_files.append(temp_file)
        
        return temp_file
    
    def create_temp_copy(self, file_path: str, directory: Optional[Path] = None) -> Path:
        """
        Create a temporary copy of a file.
        
        Args:
            file_path: Path to the original file
            directory: Directory to create the copy in. If None, uses the base temp directory.
            
        Returns:
            Path: Path to the temporary copy
        """
        import shutil
        
        # Get the file extension
        _, ext = os.path.splitext(file_path)
        
        # Create a temporary file with the same extension
        temp_file = self.create_temp_file(prefix="copy", suffix=ext, directory=directory)
        
        # Copy the file
        shutil.copy2(file_path, temp_file)
        
        return temp_file
    
    def cleanup_file(self, file_path: Path) -> bool:
        """
        Remove a specific temporary file.
        
        Args:
            file_path: Path to the temporary file
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if file_path.exists():
                if file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                
                logger.info(f"Cleaned up temporary file: {file_path}")
                
                # Remove from tracking list
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {file_path}: {str(e)}")
            return False
    
    def cleanup_all(self) -> Tuple[int, int]:
        """
        Remove all tracked temporary files.
        
        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0
        
        # Create a copy of the list to avoid modification during iteration
        temp_files_copy = self.temp_files.copy()
        
        for file_path in temp_files_copy:
            if self.cleanup_file(file_path):
                success_count += 1
            else:
                failure_count += 1
        
        logger.info(f"Cleaned up {success_count} temporary files, {failure_count} failures")
        
        return success_count, failure_count
    
    def __del__(self):
        """Destructor to ensure cleanup when the object is garbage collected."""
        try:
            self.cleanup_all()
        except:
            pass  # Suppress errors during garbage collection

def get_temp_file_manager() -> TempFileManager:
    """
    Get a singleton instance of the temporary file manager.
    
    Returns:
        TempFileManager: Instance of the temporary file manager
    """
    if not hasattr(get_temp_file_manager, 'instance'):
        temp_dir = config.get("temp_dir", None)
        get_temp_file_manager.instance = TempFileManager(temp_dir)
    
    return get_temp_file_manager.instance

# Batch file operations
class BatchProcessor:
    """
    Handles batch operations for multiple files.
    
    This class provides utilities for processing multiple files in batch mode,
    tracking progress, and aggregating results.
    """
    
    def __init__(self, max_batch_size: int = 10):
        """
        Initialize the batch processor.
        
        Args:
            max_batch_size: Maximum number of files to process in a batch
        """
        self.max_batch_size = max_batch_size
        self.temp_manager = get_temp_file_manager()
        self.current_batch = []
        self.results = {}
        
    def add_file(self, file_path: str, media_type: str = None) -> bool:
        """
        Add a file to the current batch.
        
        Args:
            file_path: Path to the file
            media_type: Type of media ('image', 'audio', 'video'). If None, will be detected.
            
        Returns:
            bool: True if file was added successfully, False otherwise
        """
        if len(self.current_batch) >= self.max_batch_size:
            logger.warning(f"Batch is full (max size: {self.max_batch_size}). Cannot add more files.")
            return False
        
        # Validate the file first
        is_valid, message, detected_media_type = validate_file(file_path, media_type)
        if not is_valid:
            logger.warning(f"Invalid file {file_path}: {message}")
            return False
        
        # Use detected media type if not provided
        if media_type is None:
            media_type = detected_media_type
        
        # Add to batch
        self.current_batch.append({
            'file_path': file_path,
            'media_type': media_type,
            'status': 'pending',
            'message': 'Added to batch',
            'timestamp': os.path.getmtime(file_path)
        })
        
        logger.info(f"Added file to batch: {file_path} (media type: {media_type})")
        return True
    
    def add_files(self, file_paths: List[str], media_type: str = None) -> Tuple[int, int]:
        """
        Add multiple files to the current batch.
        
        Args:
            file_paths: List of file paths
            media_type: Type of media for all files. If None, will be detected individually.
            
        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0
        
        for file_path in file_paths:
            if self.add_file(file_path, media_type):
                success_count += 1
            else:
                failure_count += 1
                
            # Check if batch is full
            if len(self.current_batch) >= self.max_batch_size:
                logger.warning("Batch is full. Stopping file addition.")
                break
        
        logger.info(f"Added {success_count} files to batch, {failure_count} failures")
        return success_count, failure_count
    
    def save_batch(self) -> Dict:
        """
        Save all files in the current batch.
        
        Returns:
            Dict: Dictionary with file IDs as keys and file info as values
        """
        results = {}
        
        for i, file_info in enumerate(self.current_batch):
            file_path = file_info['file_path']
            media_type = file_info['media_type']
            
            # Update status
            self.current_batch[i]['status'] = 'processing'
            
            # Save the file
            success, message, saved_info = save_uploaded_file(file_path, media_type)
            
            # Update batch info
            self.current_batch[i]['status'] = 'saved' if success else 'failed'
            self.current_batch[i]['message'] = message
            
            if success:
                # Use the stored filename as the key
                file_id = saved_info['stored_filename']
                results[file_id] = saved_info
            
            logger.info(f"Batch file {i+1}/{len(self.current_batch)}: {message}")
        
        # Store results
        self.results.update(results)
        
        return results
    
    def group_by_media_type(self) -> Dict[str, List[Dict]]:
        """
        Group batch files by media type.
        
        Returns:
            Dict[str, List[Dict]]: Dictionary with media types as keys and lists of file info as values
        """
        grouped = {
            'image': [],
            'audio': [],
            'video': []
        }
        
        for file_info in self.current_batch:
            media_type = file_info['media_type']
            if media_type in grouped:
                grouped[media_type].append(file_info)
        
        return grouped
    
    def clear_batch(self) -> None:
        """Clear the current batch."""
        self.current_batch = []
        logger.info("Batch cleared")
    
    def get_batch_stats(self) -> Dict:
        """
        Get statistics about the current batch.
        
        Returns:
            Dict: Statistics dictionary
        """
        grouped = self.group_by_media_type()
        
        stats = {
            'total_files': len(self.current_batch),
            'image_count': len(grouped['image']),
            'audio_count': len(grouped['audio']),
            'video_count': len(grouped['video']),
            'status_counts': {}
        }
        
        # Count by status
        status_counts = {}
        for file_info in self.current_batch:
            status = file_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        stats['status_counts'] = status_counts
        
        return stats

def process_directory(directory_path: str, recursive: bool = False, media_type: str = None) -> Dict:
    """
    Process all supported files in a directory.
    
    Args:
        directory_path: Path to the directory
        recursive: Whether to process subdirectories recursively
        media_type: Force specific media type for all files
        
    Returns:
        Dict: Results dictionary
    """
    if not os.path.isdir(directory_path):
        logger.error(f"Not a directory: {directory_path}")
        return {'error': 'Not a directory', 'processed_files': 0}
    
    # Get file extensions for allowed media types
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    audio_extensions = ['.wav', '.mp3', '.flac']
    video_extensions = ['.mp4', '.avi', '.mov']
    
    # Combine all allowed extensions
    allowed_extensions = set()
    if media_type is None or media_type == 'image':
        allowed_extensions.update(image_extensions)
    if media_type is None or media_type == 'audio':
        allowed_extensions.update(audio_extensions)
    if media_type is None or media_type == 'video':
        allowed_extensions.update(video_extensions)
    
    # Function to process files
    def get_files(dir_path):
        file_list = []
        try:
            for item in os.listdir(dir_path):
                full_path = os.path.join(dir_path, item)
                if os.path.isfile(full_path):
                    _, ext = os.path.splitext(full_path)
                    if ext.lower() in allowed_extensions:
                        file_list.append(full_path)
                elif recursive and os.path.isdir(full_path):
                    file_list.extend(get_files(full_path))
        except Exception as e:
            logger.error(f"Error accessing directory {dir_path}: {str(e)}")
        return file_list
    
    # Get all files
    files = get_files(directory_path)
    logger.info(f"Found {len(files)} files to process in {directory_path}")
    
    # Process files in batches
    batch_processor = BatchProcessor()
    
    # Split files into batches
    batches = [files[i:i+batch_processor.max_batch_size] 
               for i in range(0, len(files), batch_processor.max_batch_size)]
    
    all_results = {}
    
    for i, batch_files in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch_files)} files)")
        
        # Add files to batch
        batch_processor.add_files(batch_files, media_type)
        
        # Save the batch
        batch_results = batch_processor.save_batch()
        all_results.update(batch_results)
        
        # Clear for next batch
        batch_processor.clear_batch()
    
    return {
        'processed_files': len(files),
        'successful_files': len(all_results),
        'failed_files': len(files) - len(all_results),
        'results': all_results
    }