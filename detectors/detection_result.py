#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection result structures for the Deepfake Detection Platform.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time

from detectors.base_detector import MediaType, DetectionStatus


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
class DetectionRegion:
    """Region of interest in detected media."""
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    width: float  # Normalized width (0-1)
    height: float  # Normalized height (0-1)
    confidence_score: float  # Confidence score for this region
    detection_type: str  # Type of detection (e.g., "face", "artifact", "splice")
    metadata: Optional[Dict[str, Any]] = None  # Additional region metadata


@dataclass
class TimeSegment:
    """Time segment in audio/video detection."""
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    confidence_score: float  # Confidence score for this segment
    detection_type: str  # Type of detection (e.g., "voice", "sync", "motion")
    metadata: Optional[Dict[str, Any]] = None  # Additional segment metadata


@dataclass
class DetectionResult:
    """Standardized detection result structure."""
    file_path: str  # Path to the analyzed file
    media_type: MediaType  # Type of media analyzed
    is_deepfake: bool  # Whether the media is classified as a deepfake
    confidence_score: float  # Overall confidence score (0-1)
    status: DetectionStatus  # Detection status
    detector_name: str  # Name of the detector that produced this result
    regions: List[DetectionRegion]  # List of detected regions
    time_segments: List[TimeSegment]  # List of time segments (for audio/video)
    features: Dict[str, Any]  # Extracted features used for detection
    metadata: Dict[str, Any]  # Additional metadata about the detection
    timestamp: float  # When the detection was performed
    
    def __init__(
        self,
        file_path: str,
        media_type: MediaType,
        is_deepfake: bool,
        confidence_score: float,
        detector_name: str,
        regions: Optional[List[DetectionRegion]] = None,
        time_segments: Optional[List[TimeSegment]] = None,
        features: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: DetectionStatus = DetectionStatus.COMPLETED
    ):
        """
        Initialize a detection result.
        
        Args:
            file_path: Path to the analyzed file
            media_type: Type of media analyzed
            is_deepfake: Whether the media is classified as a deepfake
            confidence_score: Overall confidence score (0-1)
            detector_name: Name of the detector that produced this result
            regions: List of detected regions (optional)
            time_segments: List of time segments (optional)
            features: Extracted features used for detection (optional)
            metadata: Additional metadata about the detection (optional)
            status: Detection status (defaults to COMPLETED)
        """
        self.file_path = file_path
        self.media_type = media_type
        self.is_deepfake = is_deepfake
        self.confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to 0-1
        self.detector_name = detector_name
        self.regions = regions or []
        self.time_segments = time_segments or []
        self.features = features or {}
        self.metadata = metadata or {}
        self.status = status
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the detection result to a dictionary."""
        return {
            "file_path": self.file_path,
            "media_type": self.media_type.value,
            "is_deepfake": self.is_deepfake,
            "confidence_score": self.confidence_score,
            "status": self.status.value,
            "detector_name": self.detector_name,
            "regions": [vars(r) for r in self.regions],
            "time_segments": [vars(t) for t in self.time_segments],
            "features": self.features,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Create a detection result from a dictionary."""
        regions = [DetectionRegion(**r) for r in data.get("regions", [])]
        time_segments = [TimeSegment(**t) for t in data.get("time_segments", [])]
        
        return cls(
            file_path=data["file_path"],
            media_type=MediaType(data["media_type"]),
            is_deepfake=data["is_deepfake"],
            confidence_score=data["confidence_score"],
            detector_name=data["detector_name"],
            regions=regions,
            time_segments=time_segments,
            features=data.get("features", {}),
            metadata=data.get("metadata", {}),
            status=DetectionStatus(data.get("status", DetectionStatus.COMPLETED.value))
        )
    
    def merge_results(self, other: "DetectionResult") -> "DetectionResult":
        """
        Merge another detection result into this one.
        Useful for combining results from multiple detectors.
        
        Args:
            other: Another detection result to merge with this one
            
        Returns:
            A new merged detection result
        """
        if self.media_type != other.media_type:
            raise ValueError("Cannot merge results from different media types")
        
        if self.file_path != other.file_path:
            raise ValueError("Cannot merge results from different files")
        
        # Combine detections using weighted confidence scores
        total_confidence = self.confidence_score + other.confidence_score
        if total_confidence > 0:
            weight1 = self.confidence_score / total_confidence
            weight2 = other.confidence_score / total_confidence
            merged_is_deepfake = (
                self.is_deepfake if self.confidence_score > other.confidence_score 
                else other.is_deepfake
            )
            merged_confidence = max(self.confidence_score, other.confidence_score)
        else:
            weight1 = weight2 = 0.5
            merged_is_deepfake = self.is_deepfake or other.is_deepfake
            merged_confidence = 0.0
        
        # Merge metadata with weights
        merged_metadata = {
            "weight1": weight1,
            "weight2": weight2,
            "detector1": self.detector_name,
            "detector2": other.detector_name,
            **self.metadata,
            **other.metadata
        }
        
        # Create merged result
        return DetectionResult(
            file_path=self.file_path,
            media_type=self.media_type,
            is_deepfake=merged_is_deepfake,
            confidence_score=merged_confidence,
            detector_name=f"{self.detector_name}+{other.detector_name}",
            regions=self.regions + other.regions,
            time_segments=self.time_segments + other.time_segments,
            features={**self.features, **other.features},
            metadata=merged_metadata
        )