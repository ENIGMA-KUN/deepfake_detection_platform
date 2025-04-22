"""Stub implementation of X3D video deepfake detector."""
from __future__ import annotations

import random
from typing import Dict, Any

from detectors.base_detector import BaseDetector


class X3DVideoDetector(BaseDetector):
    """Mock X3D video detector returning random confidence for testing."""

    def __init__(self, model_name: str = "facebook/x3d", confidence_threshold: float = 0.5):
        super().__init__(model_name=model_name, confidence_threshold=confidence_threshold)

    def load(self) -> None:  # noqa: D401
        """Mock load the X3D model."""
        self.model = "MockX3DModel"

    def detect(self, media_path: str) -> Dict[str, Any]:  # noqa: D401
        """Return a fake prediction with random confidence."""
        self._validate_media(media_path)
        if self.model is None:
            self.load()

        confidence = self.normalize_confidence(random.random())
        is_deepfake = confidence > self.confidence_threshold

        metadata: Dict[str, Any] = {
            "media_type": "video",
            "details": {"model": self.model_name, "note": "Mock X3D inference"},
            "timestamp": None,
            "analysis_time": 0.0,
        }
        return self.format_result(is_deepfake, confidence, metadata)

    # Compatibility alias
    predict = detect
