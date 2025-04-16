import unittest
import json
from datetime import datetime
import re

from reports.report_data import (
    ReportData,
    DetectorResult,
    VisualizationData,
    TimelineMarker,
    MediaFile
)

class TestReportData(unittest.TestCase):
    """Tests for the report data structures."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a sample report data for testing
        self.report_data = ReportData(
            report_title="Test Deepfake Detection Report",
            media_type="image",
            media_path="/path/to/test_image.jpg",
            filename="test_image.jpg",
            verdict="authentic",
            confidence=95.5,
            analysis_duration=1.23
        )
        
        # Add some detector results
        self.report_data.add_detector_result(DetectorResult(
            name="ViT Detector",
            description="Vision Transformer based detector",
            confidence=96.0,
            verdict="authentic",
            metadata={"model": "google/vit-base-patch16-224"}
        ))
        self.report_data.add_detector_result(DetectorResult(
            name="ELA Detector",
            description="Error Level Analysis detector",
            confidence=95.0,
            verdict="authentic",
            metadata={"quality": 90}
        ))
        
        # Add some visualizations
        self.report_data.add_visualization(VisualizationData(
            type="heatmap",
            path="/path/to/heatmap.png",
            description="Detection heatmap",
            metadata={"colormap": "viridis"}
        ))
        
        # Add metadata
        self.report_data.metadata = {
            "original_size": (1920, 1080),
            "file_size": "2.5 MB",
            "format": "JPEG"
        }
        
        # Create a sample batch report
        self.batch_report = ReportData(
            report_title="Batch Analysis Report",
            batch_analysis=True
        )
        
        # Add some files to the batch report
        self.batch_report.add_file(MediaFile(
            name="file1.jpg",
            type="image",
            verdict="authentic",
            confidence=95.0,
            detailed_report_url="/reports/file1.html"
        ))
        self.batch_report.add_file(MediaFile(
            name="file2.mp4",
            type="video",
            verdict="deepfake",
            confidence=87.5,
            detailed_report_url="/reports/file2.html"
        ))
        self.batch_report.add_file(MediaFile(
            name="file3.wav",
            type="audio",
            verdict="authentic",
            confidence=92.0,
            detailed_report_url="/reports/file3.html"
        ))
    
    def test_report_data_initialization(self):
        """Test basic initialization of ReportData."""
        self.assertEqual(self.report_data.report_title, "Test Deepfake Detection Report")
        self.assertEqual(self.report_data.media_type, "image")
        self.assertEqual(self.report_data.verdict, "authentic")
        self.assertAlmostEqual(self.report_data.confidence, 95.5)
        
        # Test that report_id is a valid UUID string
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        self.assertTrue(uuid_pattern.match(self.report_data.report_id))
        
        # Test that analysis_date is a valid datetime string
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
        self.assertTrue(date_pattern.match(self.report_data.analysis_date))
    
    def test_detector_results(self):
        """Test detector results in ReportData."""
        self.assertEqual(len(self.report_data.detectors), 2)
        self.assertEqual(self.report_data.detectors[0].name, "ViT Detector")
        self.assertEqual(self.report_data.detectors[1].name, "ELA Detector")
        
        # Test adding a detector result
        self.report_data.add_detector_result(DetectorResult(
            name="Face Detector",
            description="Face detection and analysis",
            confidence=90.0,
            verdict="authentic",
            metadata={}
        ))
        self.assertEqual(len(self.report_data.detectors), 3)
        self.assertEqual(self.report_data.detectors[2].name, "Face Detector")
    
    def test_visualizations(self):
        """Test visualizations in ReportData."""
        self.assertEqual(len(self.report_data.visualizations), 1)
        self.assertEqual(self.report_data.visualizations[0].type, "heatmap")
        
        # Test adding a visualization
        self.report_data.add_visualization(VisualizationData(
            type="spectrogram",
            path="/path/to/spectrogram.png",
            description="Audio spectrogram",
            metadata={}
        ))
        self.assertEqual(len(self.report_data.visualizations), 2)
        self.assertEqual(self.report_data.visualizations[1].type, "spectrogram")
    
    def test_timeline_markers(self):
        """Test timeline markers in ReportData."""
        # Initially no timeline markers
        self.assertEqual(len(self.report_data.timeline_markers), 0)
        
        # Test adding a timeline marker
        self.report_data.add_timeline_marker(TimelineMarker(
            position=25.0,
            width=5.0,
            description="Suspicious frame",
            metadata={}
        ))
        self.assertEqual(len(self.report_data.timeline_markers), 1)
        self.assertEqual(self.report_data.timeline_markers[0].position, 25.0)
    
    def test_verdict_and_confidence_update(self):
        """Test automatic update of verdict and confidence."""
        # Create a new report
        report = ReportData()
        
        # Add detector results with conflicting verdicts
        report.add_detector_result(DetectorResult(
            name="Detector 1",
            description="Test detector 1",
            confidence=70.0,
            verdict="authentic",
            metadata={}
        ))
        report.add_detector_result(DetectorResult(
            name="Detector 2",
            description="Test detector 2",
            confidence=80.0,
            verdict="deepfake",
            metadata={}
        ))
        
        # The overall verdict should be 'deepfake' since it has higher confidence
        self.assertEqual(report.verdict, "deepfake")
        self.assertAlmostEqual(report.confidence, 80.0)
        
        # Add another detector that tilts the balance
        report.add_detector_result(DetectorResult(
            name="Detector 3",
            description="Test detector 3",
            confidence=90.0,
            verdict="authentic",
            metadata={}
        ))
        
        # Now the overall verdict should be 'authentic'
        self.assertEqual(report.verdict, "authentic")
        self.assertAlmostEqual(report.confidence, 80.0)  # (70 + 90) / 3 = 53.33
    
    def test_batch_analysis(self):
        """Test batch analysis in ReportData."""
        # Test batch statistics
        self.assertEqual(self.batch_report.total_files, 3)
        self.assertEqual(self.batch_report.authentic_files, 2)
        self.assertEqual(self.batch_report.deepfake_files, 1)
        self.assertAlmostEqual(self.batch_report.authentic_percentage, 66.66666666666666)
        self.assertAlmostEqual(self.batch_report.deepfake_percentage, 33.33333333333333)
        self.assertAlmostEqual(self.batch_report.average_confidence, 91.5)
        
        # Test adding another file
        self.batch_report.add_file(MediaFile(
            name="file4.jpg",
            type="image",
            verdict="deepfake",
            confidence=78.0,
            detailed_report_url="/reports/file4.html"
        ))
        
        # Check updated statistics
        self.assertEqual(self.batch_report.total_files, 4)
        self.assertEqual(self.batch_report.authentic_files, 2)
        self.assertEqual(self.batch_report.deepfake_files, 2)
        self.assertAlmostEqual(self.batch_report.authentic_percentage, 50.0)
        self.assertAlmostEqual(self.batch_report.deepfake_percentage, 50.0)
        self.assertAlmostEqual(self.batch_report.average_confidence, 88.125)
    
    def test_to_dict_and_from_dict(self):
        """Test conversion to and from dictionary."""
        # Convert to dictionary
        report_dict = self.report_data.to_dict()
        
        # Check that the dictionary contains the expected keys
        self.assertIn('report_id', report_dict)
        self.assertIn('report_title', report_dict)
        self.assertIn('verdict', report_dict)
        self.assertIn('detectors', report_dict)
        self.assertIn('visualizations', report_dict)
        
        # Convert back from dictionary
        report_from_dict = ReportData.from_dict(report_dict)
        
        # Check that the reconstructed report matches the original
        self.assertEqual(report_from_dict.report_title, self.report_data.report_title)
        self.assertEqual(report_from_dict.media_type, self.report_data.media_type)
        self.assertEqual(report_from_dict.verdict, self.report_data.verdict)
        self.assertEqual(len(report_from_dict.detectors), len(self.report_data.detectors))
        self.assertEqual(report_from_dict.detectors[0].name, self.report_data.detectors[0].name)
    
    def test_to_json_and_from_json(self):
        """Test conversion to and from JSON."""
        # Convert to JSON
        json_str = self.report_data.to_json()
        
        # Check that the JSON is valid
        self.assertIsInstance(json_str, str)
        json_data = json.loads(json_str)
        self.assertIsInstance(json_data, dict)
        
        # Convert back from JSON
        report_from_json = ReportData.from_json(json_str)
        
        # Check that the reconstructed report matches the original
        self.assertEqual(report_from_json.report_title, self.report_data.report_title)
        self.assertEqual(report_from_json.media_type, self.report_data.media_type)
        self.assertEqual(report_from_json.verdict, self.report_data.verdict)
        
        # Test batch report to JSON and back
        batch_json = self.batch_report.to_json()
        batch_from_json = ReportData.from_json(batch_json)
        
        self.assertEqual(batch_from_json.total_files, self.batch_report.total_files)
        self.assertEqual(batch_from_json.authentic_files, self.batch_report.authentic_files)
        self.assertEqual(len(batch_from_json.files), len(self.batch_report.files))
        self.assertEqual(batch_from_json.files[0].name, self.batch_report.files[0].name)