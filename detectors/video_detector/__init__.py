#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video deepfake detector module.
Contains implementations for detecting deepfakes in videos.
"""

from detectors.base_detector import MediaType
from detectors.detector_factory import DetectorRegistry

# Import detector implementations
from .genconvit import GenConVitDetector
from .frame_analyzer import FrameAnalyzer

# Register detectors with the registry
DetectorRegistry.register(MediaType.VIDEO)(GenConVitDetector)

__all__ = ['GenConVitDetector', 'FrameAnalyzer']