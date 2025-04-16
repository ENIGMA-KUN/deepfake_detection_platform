#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for the detection result module.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime

from detectors.detection_result import (AudioFeatures, DeepfakeCategory,
                                      DetectionResult, DetectionStatus,
                                      MediaType, Region, TimeSegment,
                                      VideoFeatures)


class TestDetectionResult(unittest.TestCase):
    """Test cases for the detection result data structures."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample region
        self.region = Region(
            x=0.1,
            y=0.2,
            width=0.3,
            height=0.4,
            confidence=0.8,
            label="fake_face",
            category=DeepfakeCategory.FACE_SWAP
        )
        
        # Create a sample time segment
        self.time_segment = TimeSegment(
            start_time=1.5,
            end_time=3.0,
            confidence=0.75,
            label="voice_clone",
            category=DeepfakeCategory.VOICE_CLONING
        )
        
        # Create sample audio features
        self.audio_features = AudioFeatures(
            spectrogram_anomalies=[self.time_segment],
            voice_inconsistencies=[],
            synthesis_markers=["neural_vocoder", "voice_artifacts"]
        )
        
        # Create sample video features
        self.video_features = VideoFeatures(
            temporal_inconsistencies=[self.time_segment],
            frame_anomalies={10: [self.region], 20: [self.region]},
            sync_issues=[]
        )
        
        # Create a sample detection result
        self.result = DetectionResult(
            id="test-result-123",
            media_type=MediaType.VIDEO,
            filename="test_video.mp4",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            is_deepfake=True,
            confidence_score=0.85,
            regions=[self.region],
            time_segments=[self.time_segment],
            audio_features=self.audio_features,
            video_features=self.video_features,
            metadata={"duration": 15.5, "resolution": "1920x1080"},
            execution_time=2.3,
            model_used="genconvit-v1",
            status=DetectionStatus.COMPLETED,
            categories=[DeepfakeCategory.FACE_SWAP, DeepfakeCategory.VOICE_CLONING]
        )
    
    def test_region_to_dict(self):
        """Test conversion of Region to dictionary."""
        region_dict = self.region.to_dict()
        
        self.assertEqual(region_dict["x"], 0.1)
        self.assertEqual(region_dict["y"], 0.2)
        self.assertEqual(region_dict["width"], 0.3)
        self.assertEqual(region_dict["height"], 0.4)
        self.assertEqual(region_dict["confidence"], 0.8)
        self.assertEqual(region_dict["label"], "fake_face")
        self.assertEqual(region_dict["category"], "face_swap")
    
    def test_region_from_dict(self):
        """Test creation of Region from dictionary."""
        region_dict = {
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.4,
            "confidence": 0.8,
            "label": "fake_face",
            "category": "face_swap"
        }
        
        region = Region.from_dict(region_dict)
        
        self.assertEqual(region.x, 0.1)
        self.assertEqual(region.y, 0.2)
        self.assertEqual(region.width, 0.3)
        self.assertEqual(region.height, 0.4)
        self.assertEqual(region.confidence, 0.8)
        self.assertEqual(region.label, "fake_face")
        self.assertEqual(region.category, DeepfakeCategory.FACE_SWAP)
    
    def test_time_segment_to_dict(self):
        """Test conversion of TimeSegment to dictionary."""
        segment_dict = self.time_segment.to_dict()
        
        self.assertEqual(segment_dict["start_time"], 1.5)
        self.assertEqual(segment_dict["end_time"], 3.0)
        self.assertEqual(segment_dict["confidence"], 0.75)
        self.assertEqual(segment_dict["label"], "voice_clone")
        self.assertEqual(segment_dict["category"], "voice_cloning")
    
    def test_time_segment_from_dict(self):
        """Test creation of TimeSegment from dictionary."""
        segment_dict = {
            "start_time": 1.5,
            "end_time": 3.0,
            "confidence": 0.75,
            "label": "voice_clone",
            "category": "voice_cloning"
        }
        
        segment = TimeSegment.from_dict(segment_dict)
        
        self.assertEqual(segment.start_time, 1.5)
        self.assertEqual(segment.end_time, 3.0)
        self.assertEqual(segment.confidence, 0.75)
        self.assertEqual(segment.label, "voice_clone")
        self.assertEqual(segment.category, DeepfakeCategory.VOICE_CLONING)
    
    def test_audio_features_to_dict(self):
        """Test conversion of AudioFeatures to dictionary."""
        features_dict = self.audio_features.to_dict()
        
        self.assertEqual(len(features_dict["spectrogram_anomalies"]), 1)
        self.assertEqual(len(features_dict["voice_inconsistencies"]), 0)
        self.assertEqual(features_dict["synthesis_markers"], ["neural_vocoder", "voice_artifacts"])
        
        segment_dict = features_dict["spectrogram_anomalies"][0]
        self.assertEqual(segment_dict["start_time"], 1.5)
        self.assertEqual(segment_dict["end_time"], 3.0)
    
    def test_audio_features_from_dict(self):
        """Test creation of AudioFeatures from dictionary."""
        features_dict = {
            "spectrogram_anomalies": [
                {
                    "start_time": 1.5,
                    "end_time": 3.0,
                    "confidence": 0.75,
                    "label": "voice_clone",
                    "category": "voice_cloning"
                }
            ],
            "voice_inconsistencies": [],
            "synthesis_markers": ["neural_vocoder", "voice_artifacts"]
        }
        
        features = AudioFeatures.from_dict(features_dict)
        
        self.assertEqual(len(features.spectrogram_anomalies), 1)
        self.assertEqual(len(features.voice_inconsistencies), 0)
        self.assertEqual(features.synthesis_markers, ["neural_vocoder", "voice_artifacts"])
        
        segment = features.spectrogram_anomalies[0]
        self.assertEqual(segment.start_time, 1.5)
        self.assertEqual(segment.end_time, 3.0)
    
    def test_video_features_to_dict(self):
        """Test conversion of VideoFeatures to dictionary."""
        features_dict = self.video_features.to_dict()
        
        self.assertEqual(len(features_dict["temporal_inconsistencies"]), 1)
        self.assertEqual(len(features_dict["frame_anomalies"]), 2)
        self.assertEqual(len(features_dict["sync_issues"]), 0)
        
        self.assertIn("10", features_dict["frame_anomalies"])
        self.assertIn("20", features_dict["frame_anomalies"])
        
        frame10_regions = features_dict["frame_anomalies"]["10"]
        self.assertEqual(len(frame10_regions), 1)
        self.assertEqual(frame10_regions[0]["x"], 0.1)
    
    def test_video_features_from_dict(self):
        """Test creation of VideoFeatures from dictionary."""
        features_dict = {
            "temporal_inconsistencies": [
                {
                    "start_time": 1.5,
                    "end_time": 3.0,
                    "confidence": 0.75,
                    "label": "voice_clone",
                    "category": "voice_cloning"
                }
            ],
            "frame_anomalies": {
                "10": [
                    {
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.3,
                        "height": 0.4,
                        "confidence": 0.8,
                        "label": "fake_face",
                        "category": "face_swap"
                    }
                ],
                "20": [
                    {
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.3,
                        "height": 0.4,
                        "confidence": 0.8,
                        "label": "fake_face",
                        "category": "face_swap"
                    }
                ]
            },
            "sync_issues": []
        }
        
        features = VideoFeatures.from_dict(features_dict)
        
        self.assertEqual(len(features.temporal_inconsistencies), 1)
        self.assertEqual(len(features.frame_anomalies), 2)
        self.assertEqual(len(features.sync_issues), 0)
        
        self.assertIn(10, features.frame_anomalies)
        self.assertIn(20, features.frame_anomalies)
        
        frame10_regions = features.frame_anomalies[10]
        self.assertEqual(len(frame10_regions), 1)
        self.assertEqual(frame10_regions[0].x, 0.1)
    
    def test_detection_result_to_dict(self):
        """Test conversion of DetectionResult to dictionary."""
        result_dict = self.result.to_dict()
        
        self.assertEqual(result_dict["id"], "test-result-123")
        self.assertEqual(result_dict["media_type"], "video")
        self.assertEqual(result_dict["filename"], "test_video.mp4")
        self.assertEqual(result_dict["timestamp"], "2023-01-01T12:00:00")
        self.assertEqual(result_dict["is_deepfake"], True)
        self.assertEqual(result_dict["confidence_score"], 0.85)
        self.assertEqual(len(result_dict["regions"]), 1)
        self.assertEqual(len(result_dict["time_segments"]), 1)
        self.assertEqual(result_dict["metadata"]["duration"], 15.5)
        self.assertEqual(result_dict["execution_time"], 2.3)
        self.assertEqual(result_dict["model_used"], "genconvit-v1")
        self.assertEqual(result_dict["status"], "completed")
        self.assertEqual(result_dict["categories"], ["face_swap", "voice_cloning"])
        
        self.assertIn("audio_features", result_dict)
        self.assertIn("video_features", result_dict)
    
    def test_detection_result_from_dict(self):
        """Test creation of DetectionResult from dictionary."""
        result_dict = {
            "id": "test-result-123",
            "media_type": "video",
            "filename": "test_video.mp4",
            "timestamp": "2023-01-01T12:00:00",
            "is_deepfake": True,
            "confidence_score": 0.85,
            "regions": [
                {
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.3,
                    "height": 0.4,
                    "confidence": 0.8,
                    "label": "fake_face",
                    "category": "face_swap"
                }
            ],
            "time_segments": [
                {
                    "start_time": 1.5,
                    "end_time": 3.0,
                    "confidence": 0.75,
                    "label": "voice_clone",
                    "category": "voice_cloning"
                }
            ],
            "audio_features": {
                "spectrogram_anomalies": [
                    {
                        "start_time": 1.5,
                        "end_time": 3.0,
                        "confidence": 0.75,
                        "label": "voice_clone",
                        "category": "voice_cloning"
                    }
                ],
                "voice_inconsistencies": [],
                "synthesis_markers": ["neural_vocoder", "voice_artifacts"]
            },
            "video_features": {
                "temporal_inconsistencies": [
                    {
                        "start_time": 1.5,
                        "end_time": 3.0,
                        "confidence": 0.75,
                        "label": "voice_clone",
                        "category": "voice_cloning"
                    }
                ],
                "frame_anomalies": {
                    "10": [
                        {
                            "x": 0.1,
                            "y": 0.2,
                            "width": 0.3,
                            "height": 0.4,
                            "confidence": 0.8,
                            "label": "fake_face",
                            "category": "face_swap"
                        }
                    ],
                    "20": [
                        {
                            "x": 0.1,
                            "y": 0.2,
                            "width": 0.3,
                            "height": 0.4,
                            "confidence": 0.8,
                            "label": "fake_face",
                            "category": "face_swap"
                        }
                    ]
                },
                "sync_issues": []
            },
            "metadata": {"duration": 15.5, "resolution": "1920x1080"},
            "execution_time": 2.3,
            "model_used": "genconvit-v1",
            "status": "completed",
            "categories": ["face_swap", "voice_cloning"]
        }
        
        result = DetectionResult.from_dict(result_dict)
        
        self.assertEqual(result.id, "test-result-123")
        self.assertEqual(result.media_type, MediaType.VIDEO)
        self.assertEqual(result.filename, "test_video.mp4")
        self.assertEqual(result.timestamp, datetime(2023, 1, 1, 12, 0, 0))
        self.assertEqual(result.is_deepfake, True)
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(len(result.regions), 1)
        self.assertEqual(len(result.time_segments), 1)
        self.assertEqual(result.metadata["duration"], 15.5)
        self.assertEqual(result.execution_time, 2.3)
        self.assertEqual(result.model_used, "genconvit-v1")
        self.assertEqual(result.status, DetectionStatus.COMPLETED)
        self.assertEqual(result.categories, [DeepfakeCategory.FACE_SWAP, DeepfakeCategory.VOICE_CLONING])
        
        self.assertIsNotNone(result.audio_features)
        self.assertIsNotNone(result.video_features)
    
    def test_create_new(self):
        """Test creation of new detection result."""
        result = DetectionResult.create_new(MediaType.IMAGE, "test.jpg")
        
        self.assertIsNotNone(result.id)
        self.assertEqual(result.media_type, MediaType.IMAGE)
        self.assertEqual(result.filename, "test.jpg")
        self.assertIsNotNone(result.timestamp)
        self.assertEqual(result.is_deepfake, False)
        self.assertEqual(result.confidence_score, 0.0)
        self.assertEqual(result.status, DetectionStatus.PENDING)
    
    def test_save_and_load(self):
        """Test saving and loading detection result to/from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the result to a file
            filepath = self.result.save_to_file(temp_dir)
            
            # Verify the file exists
            self.assertTrue(os.path.exists(filepath))
            
            # Load the result from the file
            loaded_result = DetectionResult.load_from_file(filepath)
            
            # Verify the loaded result matches the original
            self.assertEqual(loaded_result.id, self.result.id)
            self.assertEqual(loaded_result.media_type, self.result.media_type)
            self.assertEqual(loaded_result.filename, self.result.filename)
            self.assertEqual(loaded_result.timestamp, self.result.timestamp)
            self.assertEqual(loaded_result.is_deepfake, self.result.is_deepfake)
            self.assertEqual(loaded_result.confidence_score, self.result.confidence_score)
            self.assertEqual(len(loaded_result.regions), len(self.result.regions))
            self.assertEqual(len(loaded_result.time_segments), len(self.result.time_segments))
            self.assertEqual(loaded_result.metadata, self.result.metadata)
            self.assertEqual(loaded_result.execution_time, self.result.execution_time)
            self.assertEqual(loaded_result.model_used, self.result.model_used)
            self.assertEqual(loaded_result.status, self.result.status)
            self.assertEqual(loaded_result.categories, self.result.categories)


if __name__ == '__main__':
    unittest.main()