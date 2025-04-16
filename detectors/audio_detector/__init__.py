#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio deepfake detector module.
Contains implementations for detecting deepfakes in audio.
"""

from detectors.base_detector import MediaType
from detectors.detector_factory import DetectorRegistry

# Import detector implementations
from .wav2vec_detector import Wav2VecDetector
from .spectrogram_analyzer import SpectrogramAnalyzer

# Register detectors with the registry
DetectorRegistry.register(MediaType.AUDIO)(Wav2VecDetector)

__all__ = ['Wav2VecDetector', 'SpectrogramAnalyzer']