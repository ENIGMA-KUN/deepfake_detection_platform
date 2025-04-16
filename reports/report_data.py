from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import uuid
import json

@dataclass
class DetectorResult:
    """Data structure to represent a detector's result."""
    
    name: str
    description: str
    confidence: float  # 0-100 scale
    verdict: str  # 'authentic' or 'deepfake'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence,
            'verdict': self.verdict,
            'metadata': self.metadata
        }

@dataclass
class VisualizationData:
    """Data structure to represent visualization data."""
    
    type: str  # e.g., 'heatmap', 'spectrogram', 'frames'
    path: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type,
            'path': self.path,
            'description': self.description,
            'metadata': self.metadata
        }

@dataclass
class TimelineMarker:
    """Data structure to represent a timeline marker for video analysis."""
    
    position: float  # Position in percentage (0-100)
    width: float  # Width in percentage (0-100)
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position,
            'width': self.width,
            'description': self.description,
            'metadata': self.metadata
        }

@dataclass
class MediaFile:
    """Data structure to represent a media file in a batch analysis."""
    
    name: str
    type: str
    verdict: str
    confidence: float
    detailed_report_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'verdict': self.verdict,
            'confidence': self.confidence,
            'detailed_report_url': self.detailed_report_url,
            'metadata': self.metadata
        }

@dataclass
class ReportData:
    """Data structure to represent the report data."""
    
    # General report information
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_title: str = "Deepfake Detection Report"
    analysis_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Analysis results
    media_type: Optional[str] = None  # 'image', 'audio', or 'video'
    media_path: Optional[str] = None
    filename: Optional[str] = None
    
    # Verdict and confidence
    verdict: str = "unknown"  # 'authentic', 'deepfake', or 'unknown'
    confidence: float = 0.0  # 0-100 scale
    
    # Analysis duration
    analysis_duration: float = 0.0  # in seconds
    
    # Detector results
    detectors: List[DetectorResult] = field(default_factory=list)
    
    # Technical metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Visualizations
    visualizations: List[VisualizationData] = field(default_factory=list)
    
    # Timeline markers (for video analysis)
    timeline_markers: List[TimelineMarker] = field(default_factory=list)
    
    # Batch analysis
    batch_analysis: bool = False
    total_files: int = 0
    authentic_files: int = 0
    deepfake_files: int = 0
    authentic_percentage: float = 0.0
    deepfake_percentage: float = 0.0
    average_confidence: float = 0.0
    files: List[MediaFile] = field(default_factory=list)
    
    # URLs
    detailed_report_url: Optional[str] = None
    
    def calculate_batch_statistics(self) -> None:
        """Calculate statistics for batch analysis."""
        if not self.batch_analysis or not self.files:
            return
        
        self.total_files = len(self.files)
        self.authentic_files = sum(1 for file in self.files if file.verdict == 'authentic')
        self.deepfake_files = self.total_files - self.authentic_files
        
        self.authentic_percentage = (self.authentic_files / self.total_files) * 100 if self.total_files > 0 else 0
        self.deepfake_percentage = (self.deepfake_files / self.total_files) * 100 if self.total_files > 0 else 0
        
        self.average_confidence = sum(file.confidence for file in self.files) / self.total_files if self.total_files > 0 else 0
    
    def add_detector_result(self, detector: DetectorResult) -> None:
        """Add a detector result."""
        self.detectors.append(detector)
        self._update_verdict_and_confidence()
    
    def add_visualization(self, visualization: VisualizationData) -> None:
        """Add a visualization."""
        self.visualizations.append(visualization)
    
    def add_timeline_marker(self, marker: TimelineMarker) -> None:
        """Add a timeline marker."""
        self.timeline_markers.append(marker)
    
    def add_file(self, file: MediaFile) -> None:
        """Add a file to batch analysis."""
        self.batch_analysis = True
        self.files.append(file)
        self.calculate_batch_statistics()
    
    def _update_verdict_and_confidence(self) -> None:
        """Update the overall verdict and confidence based on detector results."""
        if not self.detectors:
            return
        
        # Calculate weighted confidence for each verdict
        authentic_confidence = 0.0
        deepfake_confidence = 0.0
        
        for detector in self.detectors:
            if detector.verdict == 'authentic':
                authentic_confidence += detector.confidence
            elif detector.verdict == 'deepfake':
                deepfake_confidence += detector.confidence
        
        # Determine the verdict based on which has higher confidence
        if deepfake_confidence > authentic_confidence:
            self.verdict = 'deepfake'
            self.confidence = deepfake_confidence / len(self.detectors)
        else:
            self.verdict = 'authentic'
            self.confidence = authentic_confidence / len(self.detectors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'report_id': self.report_id,
            'report_title': self.report_title,
            'analysis_date': self.analysis_date,
            'verdict': self.verdict,
            'confidence': self.confidence,
            'analysis_duration': self.analysis_duration,
            'detectors': [detector.to_dict() for detector in self.detectors],
            'metadata': self.metadata,
            'visualizations': [viz.to_dict() for viz in self.visualizations],
            'timeline_markers': [marker.to_dict() for marker in self.timeline_markers],
            'batch_analysis': self.batch_analysis
        }
        
        # Add single file analysis data if not batch
        if not self.batch_analysis:
            data.update({
                'media_type': self.media_type,
                'media_path': self.media_path,
                'filename': self.filename,
                'detailed_report_url': self.detailed_report_url
            })
        # Add batch analysis data if batch
        else:
            data.update({
                'total_files': self.total_files,
                'authentic_files': self.authentic_files,
                'deepfake_files': self.deepfake_files,
                'authentic_percentage': self.authentic_percentage,
                'deepfake_percentage': self.deepfake_percentage,
                'average_confidence': self.average_confidence,
                'files': [file.to_dict() for file in self.files]
            })
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportData':
        """Create a ReportData instance from a dictionary."""
        report = cls(
            report_id=data.get('report_id', str(uuid.uuid4())),
            report_title=data.get('report_title', "Deepfake Detection Report"),
            analysis_date=data.get('analysis_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            verdict=data.get('verdict', "unknown"),
            confidence=data.get('confidence', 0.0),
            analysis_duration=data.get('analysis_duration', 0.0),
            media_type=data.get('media_type'),
            media_path=data.get('media_path'),
            filename=data.get('filename'),
            metadata=data.get('metadata', {}),
            batch_analysis=data.get('batch_analysis', False),
            detailed_report_url=data.get('detailed_report_url')
        )
        
        # Add detector results
        for detector_data in data.get('detectors', []):
            report.detectors.append(DetectorResult(
                name=detector_data.get('name', ""),
                description=detector_data.get('description', ""),
                confidence=detector_data.get('confidence', 0.0),
                verdict=detector_data.get('verdict', "unknown"),
                metadata=detector_data.get('metadata', {})
            ))
        
        # Add visualizations
        for viz_data in data.get('visualizations', []):
            report.visualizations.append(VisualizationData(
                type=viz_data.get('type', ""),
                path=viz_data.get('path', ""),
                description=viz_data.get('description', ""),
                metadata=viz_data.get('metadata', {})
            ))
        
        # Add timeline markers
        for marker_data in data.get('timeline_markers', []):
            report.timeline_markers.append(TimelineMarker(
                position=marker_data.get('position', 0.0),
                width=marker_data.get('width', 0.0),
                description=marker_data.get('description', ""),
                metadata=marker_data.get('metadata', {})
            ))
        
        # Add files for batch analysis
        if report.batch_analysis:
            for file_data in data.get('files', []):
                report.files.append(MediaFile(
                    name=file_data.get('name', ""),
                    type=file_data.get('type', ""),
                    verdict=file_data.get('verdict', "unknown"),
                    confidence=file_data.get('confidence', 0.0),
                    detailed_report_url=file_data.get('detailed_report_url', ""),
                    metadata=file_data.get('metadata', {})
                ))
            
            # Update batch statistics
            report.calculate_batch_statistics()
        
        return report
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ReportData':
        """Create a ReportData instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)