#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image deepfake detector module.
Contains implementations for detecting deepfakes in images.
"""

from detectors.base_detector import MediaType
from detectors.detector_factory import DetectorRegistry

# Import detector implementations
from .vit_detector import VitDetector

# Register detectors with the registry
DetectorRegistry.register(MediaType.IMAGE)(VitDetector)

__all__ = ['VitDetector']