"""
Model caching system for the Deepfake Detection Platform.

This module provides utilities for caching models to improve loading times
and reduce memory usage, with support for LRU (Least Recently Used) eviction
and size-based limits.
"""

import os
import time
import json
import shutil
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import OrderedDict

# Configure logging
logger = logging.getLogger(__name__)

class ModelCacheEntry:
    """
    Represents a cached model entry.
    
    This class stores information about a cached model, including
    its key, size, access time, and reference to the model object.
    """
    
    def __init__(self, key: str, model: Any, size_bytes: int = 0):
        """
        Initialize a cache entry.
        
        Args:
            key: Cache key
            model: Model object
            size_bytes: Size of the model in bytes (0 for unknown)
        """
        self.key = key
        self.model = model
        self.size_bytes = size_bytes
        self.last_access = time.time()
        self.access_count = 1
    
    def touch(self) -> None:
        """
        Update the last access time and access count.
        """
        self.last_access = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'key': self.key,
            'size_bytes': self.size_bytes,
            'last_access': self.last_access,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model: Any) -> 'ModelCacheEntry':
        """
        Create a cache entry from a dictionary.
        
        Args:
            data: Dictionary representation
            model: Model object
            
        Returns:
            Cache entry instance
        """
        entry = cls(data['key'], model, data['size_bytes'])
        entry.last_access = data['last_access']
        entry.access_count = data['access_count']
        return entry

class ModelCache:
    """
    Cache for loaded models to improve performance.
    
    This class provides a mechanism for caching models in memory
    with support for LRU (Least Recently Used) eviction, size-based
    limits, and persistence.
    """
    
    def __init__(
        self,
        max_entries: int = 10,
        max_size_mb: int = 1024,  # 1GB default
        cache_dir: str = 'models/cache',
        metadata_file: str = 'cache_metadata.json'
    ):
        """
        Initialize the model cache.
        
        Args:
            max_entries: Maximum number of models to cache (0 for unlimited)
            max_size_mb: Maximum total size in MB (0 for unlimited)
            cache_dir: Directory for model files
            metadata_file: Filename for cache metadata
        """
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb > 0 else 0
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, metadata_file)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Dictionary to store cached models (ordered by insertion)
        self.cached_models = OrderedDict()
        
        # Set to track models that should be kept in cache
        self.pinned_models = set()
        
        # Track current cache size
        self.current_size_bytes = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Load existing cache metadata if available
        self._load_metadata()
        
        logger.info(f"Model cache initialized: max entries={max_entries}, max size={max_size_mb}MB")
    
    def _load_metadata(self) -> None:
        """
        Load cache metadata from file.
        """
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load pinned models
                self.pinned_models = set(metadata.get('pinned_models', []))
                
                # Update cache size
                self.current_size_bytes = metadata.get('total_size_bytes', 0)
                
                logger.info(f"Loaded cache metadata: {len(self.pinned_models)} pinned models, "
                           f"{self.current_size_bytes/1024/1024:.2f}MB total size")
            else:
                logger.info("No cache metadata found, starting with empty cache")
                
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
            # Start with empty cache in case of error
            self.pinned_models = set()
            self.current_size_bytes = 0
    
    def _save_metadata(self) -> None:
        """
        Save cache metadata to file.
        """
        try:
            # Collect metadata
            metadata = {
                'timestamp': time.time(),
                'max_entries': self.max_entries,
                'max_size_bytes': self.max_size_bytes,
                'total_size_bytes': self.current_size_bytes,
                'pinned_models': list(self.pinned_models),
                'entries': {}
            }
            
            # Add entries metadata (not the models themselves)
            for key, entry in self.cached_models.items():
                metadata['entries'][key] = entry.to_dict()
            
            # Save to file
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug("Cache metadata saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a model from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached model or None if not found
        """
        with self.lock:
            if key in self.cached_models:
                # Update access time and count
                entry = self.cached_models[key]
                entry.touch()
                
                # Move to end of OrderedDict to update LRU order
                self.cached_models.move_to_end(key)
                
                logger.debug(f"Cache hit for key: {key}")
                return entry.model
            else:
                logger.debug(f"Cache miss for key: {key}")
                return None
    
    def put(self, key: str, model: Any, size_bytes: int = 0) -> bool:
        """
        Add a model to the cache.
        
        Args:
            key: Cache key
            model: Model to cache
            size_bytes: Size of the model in bytes
            
        Returns:
            True if added successfully, False otherwise
        """
        with self.lock:
            # Check if model is already in cache
            if key in self.cached_models:
                # Update existing entry
                old_entry = self.cached_models[key]
                old_size = old_entry.size_bytes
                
                # Update size tracking
                self.current_size_bytes -= old_size
                self.current_size_bytes += size_bytes
                
                # Create new entry with updated model and size
                entry = ModelCacheEntry(key, model, size_bytes)
                entry.access_count = old_entry.access_count + 1
                
                # Replace in cache
                self.cached_models[key] = entry
                self.cached_models.move_to_end(key)
                
                logger.debug(f"Updated existing cache entry: {key}")
                
            else:
                # Create new entry
                entry = ModelCacheEntry(key, model, size_bytes)
                
                # Check size limit before adding
                if self.max_size_bytes > 0 and size_bytes > self.max_size_bytes:
                    logger.warning(f"Model size ({size_bytes/1024/1024:.2f}MB) exceeds "
                                 f"cache limit ({self.max_size_bytes/1024/1024:.2f}MB)")
                    return False
                
                # Make room in cache if needed
                self._ensure_capacity(size_bytes)
                
                # Add to cache
                self.cached_models[key] = entry
                self.current_size_bytes += size_bytes
                
                logger.debug(f"Added new cache entry: {key} "
                            f"({size_bytes/1024/1024:.2f}MB, "
                            f"total: {self.current_size_bytes/1024/1024:.2f}MB)")
            
            # Save metadata
            self._save_metadata()
            
            return True
    
    def _ensure_capacity(self, required_bytes: int) -> None:
        """
        Ensure there is enough capacity for a new model.
        
        This will evict models based on LRU policy if necessary.
        
        Args:
            required_bytes: Size of new model in bytes
        """
        # Check entry count limit
        while (self.max_entries > 0 and 
              len(self.cached_models) >= self.max_entries and
              self.cached_models):
            # Find a non-pinned model to evict
            for key in list(self.cached_models.keys()):
                if key not in self.pinned_models:
                    self._evict(key)
                    break
            else:
                # All models are pinned, can't evict any
                logger.warning("Cannot evict any models: all are pinned")
                break
        
        # Check size limit
        while (self.max_size_bytes > 0 and 
              self.current_size_bytes + required_bytes > self.max_size_bytes and
              self.cached_models):
            # Find a non-pinned model to evict
            for key in list(self.cached_models.keys()):
                if key not in self.pinned_models:
                    self._evict(key)
                    break
            else:
                # All models are pinned, can't evict any
                logger.warning("Cannot free enough space: all models are pinned")
                break
    
    def _evict(self, key: str) -> bool:
        """
        Evict a model from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if evicted, False otherwise
        """
        if key in self.cached_models:
            # Get entry before removing
            entry = self.cached_models[key]
            
            # Remove from cache
            del self.cached_models[key]
            
            # Update size tracking
            self.current_size_bytes -= entry.size_bytes
            
            logger.debug(f"Evicted model from cache: {key} "
                        f"({entry.size_bytes/1024/1024:.2f}MB)")
            return True
        return False
    
    def remove(self, key: str) -> bool:
        """
        Explicitly remove a model from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if removed, False if not found
        """
        with self.lock:
            # Check if model is pinned
            if key in self.pinned_models:
                # Unpin first
                self.unpin(key)
            
            result = self._evict(key)
            
            # Save metadata if something was removed
            if result:
                self._save_metadata()
            
            return result
    
    def pin(self, key: str) -> bool:
        """
        Pin a model to prevent it from being evicted.
        
        Args:
            key: Cache key
            
        Returns:
            True if pinned, False if not found
        """
        with self.lock:
            if key in self.cached_models:
                self.pinned_models.add(key)
                logger.debug(f"Pinned model in cache: {key}")
                
                # Save metadata
                self._save_metadata()
                
                return True
            return False
    
    def unpin(self, key: str) -> bool:
        """
        Unpin a model to allow it to be evicted.
        
        Args:
            key: Cache key
            
        Returns:
            True if unpinned, False if not found or not pinned
        """
        with self.lock:
            if key in self.pinned_models:
                self.pinned_models.remove(key)
                logger.debug(f"Unpinned model in cache: {key}")
                
                # Save metadata
                self._save_metadata()
                
                return True
            return False
    
    def clear(self, include_pinned: bool = False) -> int:
        """
        Clear the cache.
        
        Args:
            include_pinned: Whether to clear pinned models as well
            
        Returns:
            Number of models cleared
        """
        with self.lock:
            if include_pinned:
                # Clear everything
                count = len(self.cached_models)
                self.cached_models.clear()
                self.pinned_models.clear()
                self.current_size_bytes = 0
                
                logger.info(f"Cleared {count} models from cache (including pinned)")
                
            else:
                # Clear only non-pinned models
                keys_to_remove = [key for key in self.cached_models if key not in self.pinned_models]
                count = len(keys_to_remove)
                
                for key in keys_to_remove:
                    self._evict(key)
                
                logger.info(f"Cleared {count} non-pinned models from cache")
            
            # Save metadata
            self._save_metadata()
            
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            # Count models by type
            model_types = {}
            for key in self.cached_models:
                # Extract model type from key (assuming key format: name_type_hash)
                parts = key.split('_')
                if len(parts) >= 2:
                    model_type = parts[-2]
                    model_types[model_type] = model_types.get(model_type, 0) + 1
            
            # Calculate hit ratio if available
            hit_ratio = 0.0
            
            return {
                'count': len(self.cached_models),
                'max_entries': self.max_entries,
                'size_mb': self.current_size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024 if self.max_size_bytes > 0 else 0,
                'pinned_count': len(self.pinned_models),
                'model_types': model_types,
                'hit_ratio': hit_ratio
            }
    
    def get_cached_keys(self) -> List[str]:
        """
        Get list of currently cached keys.
        
        Returns:
            List of cache keys
        """
        with self.lock:
            return list(self.cached_models.keys())
    
    def get_pinned_keys(self) -> List[str]:
        """
        Get list of currently pinned keys.
        
        Returns:
            List of pinned keys
        """
        with self.lock:
            return list(self.pinned_models)

class ModelCacheManager:
    """
    Manager for model cache operations.
    
    This class provides a high-level interface for model caching,
    with support for memory and disk caching.
    """
    
    def __init__(
        self,
        memory_cache_size: int = 5,  # Number of models in memory
        disk_cache_dir: str = 'models/cache',
        disk_cache_size_mb: int = 5120  # 5GB disk cache
    ):
        """
        Initialize the model cache manager.
        
        Args:
            memory_cache_size: Maximum number of models in memory
            disk_cache_dir: Directory for disk cache
            disk_cache_size_mb: Maximum disk cache size in MB
        """
        # Initialize memory cache
        self.memory_cache = ModelCache(
            max_entries=memory_cache_size,
            max_size_mb=0,  # No explicit size limit for memory cache
            cache_dir=disk_cache_dir,
            metadata_file='memory_cache_metadata.json'
        )
        
        # Track disk cache location and size
        self.disk_cache_dir = disk_cache_dir
        self.disk_cache_size_mb = disk_cache_size_mb
        
        # Create disk cache directory if it doesn't exist
        os.makedirs(disk_cache_dir, exist_ok=True)
        
        # Ensure disk cache size is respected
        self._cleanup_disk_cache()
        
        logger.info(f"Model cache manager initialized: "
                   f"memory_cache_size={memory_cache_size}, "
                   f"disk_cache_size={disk_cache_size_mb}MB")
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """
        Get a model from memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached model or None if not found
        """
        return self.memory_cache.get(key)
    
    def put_in_memory(self, key: str, model: Any, size_bytes: int = 0) -> bool:
        """
        Add a model to memory cache.
        
        Args:
            key: Cache key
            model: Model to cache
            size_bytes: Size of the model in bytes
            
        Returns:
            True if added successfully, False otherwise
        """
        return self.memory_cache.put(key, model, size_bytes)
    
    def is_in_disk_cache(self, key: str) -> bool:
        """
        Check if a model exists in disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if found, False otherwise
        """
        cache_path = self._get_disk_cache_path(key)
        return os.path.exists(cache_path)
    
    def get_disk_cache_path(self, key: str) -> str:
        """
        Get the disk cache path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cached model file
        """
        return self._get_disk_cache_path(key)
    
    def save_to_disk(self, key: str, model_file: str) -> bool:
        """
        Save a model file to disk cache.
        
        Args:
            key: Cache key
            model_file: Path to model file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure disk cache size limit
            self._ensure_disk_capacity(os.path.getsize(model_file))
            
            # Copy file to cache
            cache_path = self._get_disk_cache_path(key)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            shutil.copy2(model_file, cache_path)
            
            logger.debug(f"Saved model to disk cache: {key} -> {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to disk cache: {str(e)}")
            return False
    
    def remove_from_disk(self, key: str) -> bool:
        """
        Remove a model from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if removed, False if not found
        """
        try:
            cache_path = self._get_disk_cache_path(key)
            
            if os.path.exists(cache_path):
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                else:
                    os.remove(cache_path)
                
                logger.debug(f"Removed model from disk cache: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing model from disk cache: {str(e)}")
            return False
    
    def clear_memory_cache(self) -> int:
        """
        Clear the memory cache.
        
        Returns:
            Number of models cleared
        """
        return self.memory_cache.clear()
    
    def clear_disk_cache(self) -> int:
        """
        Clear the disk cache.
        
        Returns:
            Number of files cleared
        """
        try:
            # Count files before clearing
            count = 0
            for root, dirs, files in os.walk(self.disk_cache_dir):
                count += len(files)
            
            # Remove files (but keep the directory structure)
            for root, dirs, files in os.walk(self.disk_cache_dir):
                for file in files:
                    if file != 'memory_cache_metadata.json':  # Keep metadata
                        os.remove(os.path.join(root, file))
            
            logger.info(f"Cleared {count} files from disk cache")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing disk cache: {str(e)}")
            return 0
    
    def _get_disk_cache_path(self, key: str) -> str:
        """
        Get the disk cache path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cached model file
        """
        # Create a directory structure based on key
        # Using the first two chars of hash as directory to avoid too many files in one dir
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_subdir = os.path.join(self.disk_cache_dir, key_hash[:2])
        
        # Create cache filename from key
        safe_key = key.replace('/', '_').replace('\\', '_')
        filename = f"{safe_key}_{key_hash[2:]}.bin"
        
        return os.path.join(cache_subdir, filename)
    
    def _get_disk_cache_size(self) -> int:
        """
        Get total size of disk cache in bytes.
        
        Returns:
            Size in bytes
        """
        total_size = 0
        
        for root, dirs, files in os.walk(self.disk_cache_dir):
            for file in files:
                if file != 'memory_cache_metadata.json':  # Exclude metadata
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        
        return total_size
    
    def _cleanup_disk_cache(self) -> None:
        """
        Clean up disk cache to respect size limit.
        """
        # Get current size
        current_size = self._get_disk_cache_size()
        max_size = self.disk_cache_size_mb * 1024 * 1024
        
        if current_size <= max_size:
            # Already within limit
            return
        
        logger.info(f"Disk cache size ({current_size/1024/1024:.2f}MB) "
                   f"exceeds limit ({max_size/1024/1024:.2f}MB). Cleaning up...")
        
        # Collect file info
        files_info = []
        
        for root, dirs, files in os.walk(self.disk_cache_dir):
            for file in files:
                if file != 'memory_cache_metadata.json':  # Exclude metadata
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    size = os.path.getsize(file_path)
                    files_info.append((file_path, mtime, size))
        
        # Sort by modification time (oldest first)
        files_info.sort(key=lambda x: x[1])
        
        # Remove files until size is below limit
        removed_size = 0
        removed_count = 0
        
        for file_path, _, size in files_info:
            if current_size - removed_size <= max_size:
                break
            
            try:
                os.remove(file_path)
                removed_size += size
                removed_count += 1
                logger.debug(f"Removed from disk cache: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")
        
        logger.info(f"Removed {removed_count} files ({removed_size/1024/1024:.2f}MB) from disk cache")
    
    def _ensure_disk_capacity(self, required_bytes: int) -> None:
        """
        Ensure there is enough capacity in disk cache.
        
        Args:
            required_bytes: Size of new model in bytes
        """
        # Get current size
        current_size = self._get_disk_cache_size()
        max_size = self.disk_cache_size_mb * 1024 * 1024
        
        # Check if we need to make room
        if current_size + required_bytes <= max_size:
            # Already enough space
            return
        
        # Calculate how much space to free
        to_free = current_size + required_bytes - max_size
        
        logger.info(f"Need to free {to_free/1024/1024:.2f}MB from disk cache")
        
        # Collect file info
        files_info = []
        
        for root, dirs, files in os.walk(self.disk_cache_dir):
            for file in files:
                if file != 'memory_cache_metadata.json':  # Exclude metadata
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    size = os.path.getsize(file_path)
                    files_info.append((file_path, mtime, size))
        
        # Sort by modification time (oldest first)
        files_info.sort(key=lambda x: x[1])
        
        # Remove files until enough space is freed
        freed_space = 0
        removed_count = 0
        
        for file_path, _, size in files_info:
            if freed_space >= to_free:
                break
            
            try:
                os.remove(file_path)
                freed_space += size
                removed_count += 1
                logger.debug(f"Removed from disk cache: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")
        
        logger.info(f"Freed {freed_space/1024/1024:.2f}MB by removing {removed_count} files")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        memory_stats = self.memory_cache.get_stats()
        
        disk_size = self._get_disk_cache_size()
        
        # Count files in disk cache
        file_count = 0
        for root, dirs, files in os.walk(self.disk_cache_dir):
            for file in files:
                if file != 'memory_cache_metadata.json':  # Exclude metadata
                    file_count += 1
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': {
                'count': file_count,
                'size_mb': disk_size / 1024 / 1024,
                'max_size_mb': self.disk_cache_size_mb,
                'usage_percent': (disk_size / (self.disk_cache_size_mb * 1024 * 1024)) * 100 if self.disk_cache_size_mb > 0 else 0
            }
        }