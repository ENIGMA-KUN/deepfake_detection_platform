"""
File handling utilities for the Deepfake Detection Platform.

This module provides file validation, media type detection, and file
management functionality.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import mimetypes
import shutil

from detectors.base_detector import MediaType
from app.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

class FileHandler:
    """File handling and validation for the Deepfake Detection Platform."""
    
    DEFAULT_ALLOWED_EXTENSIONS = {
        MediaType.IMAGE.value: ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
        MediaType.AUDIO.value: ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
        MediaType.VIDEO.value: ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    }
    
    MIME_TYPE_MAP = {
        'image': MediaType.IMAGE,
        'audio': MediaType.AUDIO,
        'video': MediaType.VIDEO
    }
    
    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    def __init__(
        self,
        upload_dir: str,
        allowed_extensions: Optional[Dict[str, list]] = None,
        max_file_size: Optional[int] = None
    ):
        """
        Initialize the file handler.
        
        Args:
            upload_dir: Directory for temporary file storage
            allowed_extensions: Dictionary of allowed file extensions by media type
            max_file_size: Maximum allowed file size in bytes
        """
        self.upload_dir = Path(upload_dir)
        self.allowed_extensions = allowed_extensions or self.DEFAULT_ALLOWED_EXTENSIONS
        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE
        
        # Initialize upload directory
        self._init_upload_dir()
        
        # Initialize mime type detection
        self._init_mime_types()
        
        logger.info(
            f"FileHandler initialized with upload_dir={upload_dir}, "
            f"max_file_size={self.max_file_size/1024/1024:.1f}MB"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Upload directory initialized: {self.upload_dir}")
        except Exception as e:
            logger.error(f"Error creating upload directory: {e}")
            raise
    
    def _init_mime_types(self):
        """Initialize mime type detection."""
        # Add common mime types
        mimetypes.init()
        
        # Add custom mime types
        mimetypes.add_type('audio/wav', '.wav')
        mimetypes.add_type('audio/flac', '.flac')
        mimetypes.add_type('video/mkv', '.mkv')
        mimetypes.add_type('video/webm', '.webm')
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate a file against size and type restrictions.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.is_file():
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            # Check file size
            size = path.stat().st_size
            if size > self.max_file_size:
                logger.warning(
                    f"File too large: {size/1024/1024:.1f}MB > "
                    f"{self.max_file_size/1024/1024:.1f}MB"
                )
                return False
            
            # Check file extension
            extension = path.suffix.lower()
            media_type = self.detect_media_type(file_path)
            
            if media_type is None:
                logger.warning(f"Unsupported file type: {extension}")
                return False
            
            if extension not in self.allowed_extensions[media_type.value]:
                logger.warning(f"Invalid extension for {media_type.value}: {extension}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False
    
    def detect_media_type(self, file_path: str) -> Optional[MediaType]:
        """
        Detect the media type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MediaType enum value or None if unknown
        """
        try:
            # Use python-magic for mime type detection
            mime = magic.from_file(file_path, mime=True)
            
            if mime.startswith('image/'):
                return MediaType.IMAGE
            elif mime.startswith('audio/'):
                return MediaType.AUDIO
            elif mime.startswith('video/'):
                return MediaType.VIDEO
            else:
                logger.warning(f"Unknown mime type for {file_path}: {mime}")
                return None
                
        except Exception as e:
            logger.error(f"Error detecting media type for {file_path}: {e}")
            return None
    
    def save_uploaded_file(self, file_path: str, save_path: Optional[str] = None) -> Optional[str]:
        """
        Save an uploaded file to the upload directory.
        
        Args:
            file_path: Path to the source file
            save_path: Optional specific path to save to (within upload_dir)
            
        Returns:
            Path where the file was saved, or None if failed
        """
        try:
            src_path = Path(file_path)
            
            if save_path:
                dst_path = self.upload_dir / save_path
            else:
                # Generate unique filename if needed
                dst_path = self.upload_dir / src_path.name
                counter = 1
                while dst_path.exists():
                    stem = src_path.stem
                    suffix = src_path.suffix
                    dst_path = self.upload_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            # Create parent directories if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            logger.info(f"Saved uploaded file to {dst_path}")
            
            return str(dst_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file {file_path}: {e}")
            return None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            media_type = self.detect_media_type(file_path)
            
            return {
                'name': path.name,
                'path': str(path),
                'size': stat.st_size,
                'size_mb': stat.st_size / 1024 / 1024,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'media_type': media_type.value if media_type else None,
                'extension': path.suffix.lower(),
                'mime_type': magic.from_file(file_path, mime=True)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}