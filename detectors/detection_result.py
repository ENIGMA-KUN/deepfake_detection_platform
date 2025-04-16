#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection result module for the Deepfake Detection Platform.
Defines data structures for storing and manipulating detection results.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np


class MediaType(Enum):
    """Enum for supported media types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class DetectionStatus(Enum):
    """Enum for detection status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DeepfakeCategory(Enum):
    """Enum for deepfake categories."""
    FACE_SWAP = "face_swap"
    FACE_MANIPULATION = "face_manipulation"
    GAN_GENERATED = "gan_generated"
    VOICE_CLONING = "voice_cloning"
    AUDIO_SYNTHESIS = "audio_synthesis"
    VIDEO_MANIPULATION = "video_manipulation"
    UNKNOWN = "unknown"


@dataclass
class Region:
    """Data class for representing a region of interest in media."""
    x: float  # Normalized x-coordinate (0.0 to 1.0)
    y: float  # Normalized y-coordinate (0.0 to 1.0)
    width: float  # Normalized width (0.0 to 1.0)
    height: float  # Normalized height (0.0 to 1.0)
    confidence: float  # Confidence score (0.0 to 1.0)
    label: str  # Label for the region
    category: Optional[DeepfakeCategory] = None  # Category of manipulation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert region to dictionary."""
        result = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "label": self.label
        }
        
        if self.category:
            result["category"] = self.category.value
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Region':
        """Create Region from dictionary."""
        category = None
        if "category" in data and data["category"]:
            try:
                category = DeepfakeCategory(data["category"])
            except ValueError:
                category = DeepfakeCategory.UNKNOWN
                
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            confidence=data["confidence"],
            label=data["label"],
            category=category
        )


@dataclass
class TimeSegment:
    """Data class for representing a time segment in audio or video."""
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    confidence: float  # Confidence score (0.0 to 1.0)
    label: str  # Label for the segment
    category: Optional[DeepfakeCategory] = None  # Category of manipulation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert time segment to dictionary."""
        result = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "label": self.label
        }
        
        if self.category:
            result["category"] = self.category.value
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSegment':
        """Create TimeSegment from dictionary."""
        category = None
        if "category" in data and data["category"]:
            try:
                category = DeepfakeCategory(data["category"])
            except ValueError:
                category = DeepfakeCategory.UNKNOWN
                
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            confidence=data["confidence"],
            label=data["label"],
            category=category
        )


@dataclass
class AudioFeatures:
    """Data class for storing audio-specific detection features."""
    spectrogram_anomalies: List[TimeSegment] = field(default_factory=list)
    voice_inconsistencies: List[TimeSegment] = field(default_factory=list)
    synthesis_markers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audio features to dictionary."""
        return {
            "spectrogram_anomalies": [s.to_dict() for s in self.spectrogram_anomalies],
            "voice_inconsistencies": [v.to_dict() for v in self.voice_inconsistencies],
            "synthesis_markers": self.synthesis_markers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioFeatures':
        """Create AudioFeatures from dictionary."""
        return cls(
            spectrogram_anomalies=[TimeSegment.from_dict(s) for s in data.get("spectrogram_anomalies", [])],
            voice_inconsistencies=[TimeSegment.from_dict(v) for v in data.get("voice_inconsistencies", [])],
            synthesis_markers=data.get("synthesis_markers", [])
        )


@dataclass
class VideoFeatures:
    """Data class for storing video-specific detection features."""
    temporal_inconsistencies: List[TimeSegment] = field(default_factory=list)
    frame_anomalies: Dict[int, List[Region]] = field(default_factory=dict)
    sync_issues: List[TimeSegment] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert video features to dictionary."""
        return {
            "temporal_inconsistencies": [t.to_dict() for t in self.temporal_inconsistencies],
            "frame_anomalies": {str(k): [r.to_dict() for r in v] for k, v in self.frame_anomalies.items()},
            "sync_issues": [s.to_dict() for s in self.sync_issues]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoFeatures':
        """Create VideoFeatures from dictionary."""
        frame_anomalies = {}
        for frame_idx, regions in data.get("frame_anomalies", {}).items():
            frame_anomalies[int(frame_idx)] = [Region.from_dict(r) for r in regions]
            
        return cls(
            temporal_inconsistencies=[TimeSegment.from_dict(t) for t in data.get("temporal_inconsistencies", [])],
            frame_anomalies=frame_anomalies,
            sync_issues=[TimeSegment.from_dict(s) for s in data.get("sync_issues", [])]
        )


@dataclass
class DetectionResult:
    """Data class for storing comprehensive detection results."""
    id: str  # Unique identifier for the detection
    media_type: MediaType  # Type of media analyzed
    filename: str  # Original filename
    timestamp: datetime  # When the detection was performed
    is_deepfake: bool  # True if deepfake detected, False otherwise
    confidence_score: float  # Overall confidence score (0.0 to 1.0)
    regions: List[Region] = field(default_factory=list)  # List of regions of interest (for images)
    time_segments: List[TimeSegment] = field(default_factory=list)  # List of time segments (for audio/video)
    audio_features: Optional[AudioFeatures] = None  # Audio-specific features
    video_features: Optional[VideoFeatures] = None  # Video-specific features
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    execution_time: float = 0.0  # Time taken for detection in seconds
    model_used: str = ""  # Name/version of the model used
    status: DetectionStatus = DetectionStatus.PENDING  # Current status of the detection
    categories: List[DeepfakeCategory] = field(default_factory=list)  # List of detected deepfake categories
    
    @classmethod
    def create_new(cls, media_type: MediaType, filename: str) -> 'DetectionResult':
        """Create a new detection result with default values.
        
        Args:
            media_type: Type of media being analyzed
            filename: Original filename
            
        Returns:
            New DetectionResult instance with default values
        """
        return cls(
            id=str(uuid.uuid4()),
            media_type=media_type,
            filename=filename,
            timestamp=datetime.now(),
            is_deepfake=False,
            confidence_score=0.0,
            status=DetectionStatus.PENDING
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the detection result to a dictionary."""
        result = {
            "id": self.id,
            "media_type": self.media_type.value,
            "filename": self.filename,
            "timestamp": self.timestamp.isoformat(),
            "is_deepfake": self.is_deepfake,
            "confidence_score": self.confidence_score,
            "regions": [r.to_dict() for r in self.regions],
            "time_segments": [t.to_dict() for t in self.time_segments],
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "model_used": self.model_used,
            "status": self.status.value,
            "categories": [c.value for c in self.categories]
        }
        
        if self.audio_features:
            result["audio_features"] = self.audio_features.to_dict()
            
        if self.video_features:
            result["video_features"] = self.video_features.to_dict()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create DetectionResult from dictionary."""
        # Convert basic fields
        media_type = MediaType(data["media_type"])
        status = DetectionStatus(data["status"])
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        # Convert regions and time segments
        regions = [Region.from_dict(r) for r in data.get("regions", [])]
        time_segments = [TimeSegment.from_dict(t) for t in data.get("time_segments", [])]
        
        # Convert categories
        categories = []
        for category in data.get("categories", []):
            try:
                categories.append(DeepfakeCategory(category))
            except ValueError:
                categories.append(DeepfakeCategory.UNKNOWN)
        
        # Convert audio and video features if present
        audio_features = None
        if "audio_features" in data:
            audio_features = AudioFeatures.from_dict(data["audio_features"])
            
        video_features = None
        if "video_features" in data:
            video_features = VideoFeatures.from_dict(data["video_features"])
        
        return cls(
            id=data["id"],
            media_type=media_type,
            filename=data["filename"],
            timestamp=timestamp,
            is_deepfake=data["is_deepfake"],
            confidence_score=data["confidence_score"],
            regions=regions,
            time_segments=time_segments,
            audio_features=audio_features,
            video_features=video_features,
            metadata=data.get("metadata", {}),
            execution_time=data.get("execution_time", 0.0),
            model_used=data.get("model_used", ""),
            status=status,
            categories=categories
        )
    
    def save_to_file(self, directory: str) -> str:
        """Save detection result to a JSON file.
        
        Args:
            directory: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        filename = f"{self.id}.json"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DetectionResult':
        """Load detection result from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded DetectionResult instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file isn't valid JSON
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return cls.from_dict(data)