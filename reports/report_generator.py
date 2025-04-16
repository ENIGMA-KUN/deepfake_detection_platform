"""
Report generator for the Deepfake Detection Platform.
Processes detection results and generates reports using templates.
"""

import os
import json
import datetime
import jinja2
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from detectors.detection_result import DetectionResult


class ReportGenerator:
    """
    Generates detailed and summary reports from detection results.
    Supports HTML output format with the ability to export to PDF and CSV.
    """
    
    def __init__(self, template_dir: str = "reports/templates"):
        """
        Initialize the report generator.
        
        Args:
            template_dir: Directory containing the report templates
        """
        self.template_dir = template_dir
        self.env = self._setup_jinja_env()
        
    def _setup_jinja_env(self) -> jinja2.Environment:
        """
        Set up the Jinja2 environment for template rendering.
        
        Returns:
            Configured Jinja2 environment
        """
        # Create Jinja2 environment with the template directory
        template_loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
        env = jinja2.Environment(
            loader=template_loader,
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Add custom filters if needed
        env.filters['format_timestamp'] = lambda ts: datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        env.filters['percentage'] = lambda val: f"{val * 100:.2f}%"
        
        return env
    
    def generate_detailed_report(self, result: DetectionResult, output_path: str) -> str:
        """
        Generate a detailed report for a detection result.
        
        Args:
            result: The detection result to include in the report
            output_path: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        # Load the detailed report template
        template = self.env.get_template('detailed_report.html')
        
        # Prepare the template data
        template_data = self._prepare_detailed_template_data(result)
        
        # Generate the report content
        report_content = template.render(**template_data)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the report to the output path
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return output_path
    
    def generate_summary_report(self, results: List[DetectionResult], output_path: str) -> str:
        """
        Generate a summary report for multiple detection results.
        
        Args:
            results: List of detection results to include in the report
            output_path: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        # Load the summary report template
        template = self.env.get_template('summary_report.html')
        
        # Prepare the template data
        template_data = self._prepare_summary_template_data(results)
        
        # Generate the report content
        report_content = template.render(**template_data)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the report to the output path
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return output_path
    
    def _prepare_detailed_template_data(self, result: DetectionResult) -> Dict[str, Any]:
        """
        Prepare data for the detailed report template.
        
        Args:
            result: Detection result to process
            
        Returns:
            Dictionary with template variables
        """
        # Basic report information
        template_data = {
            'report_title': f"Deepfake Detection Report: {os.path.basename(result.file_path)}",
            'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_path': result.file_path,
            'file_name': os.path.basename(result.file_path),
            'media_type': result.media_type,
            'detection_time': result.timestamp,
            'confidence_score': result.confidence_score,
            'is_deepfake': result.is_deepfake,
            'detector_used': result.detector_name,
            'regions_of_interest': result.regions_of_interest,
            'metadata': result.metadata,
            'analysis_details': result.analysis_details
        }
        
        # Media-specific information
        if result.media_type == 'image':
            template_data.update(self._prepare_image_template_data(result))
        elif result.media_type == 'audio':
            template_data.update(self._prepare_audio_template_data(result))
        elif result.media_type == 'video':
            template_data.update(self._prepare_video_template_data(result))
        
        return template_data
    
    def _prepare_summary_template_data(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """
        Prepare data for the summary report template.
        
        Args:
            results: List of detection results to process
            
        Returns:
            Dictionary with template variables
        """
        # Group results by media type
        image_results = [r for r in results if r.media_type == 'image']
        audio_results = [r for r in results if r.media_type == 'audio']
        video_results = [r for r in results if r.media_type == 'video']
        
        # Calculate summary statistics
        total_count = len(results)
        deepfake_count = sum(1 for r in results if r.is_deepfake)
        authentic_count = total_count - deepfake_count
        
        avg_confidence = sum(r.confidence_score for r in results) / total_count if total_count > 0 else 0
        
        # Template data
        template_data = {
            'report_title': "Deepfake Detection Summary Report",
            'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_count': total_count,
            'deepfake_count': deepfake_count,
            'authentic_count': authentic_count,
            'deepfake_percentage': (deepfake_count / total_count * 100) if total_count > 0 else 0,
            'avg_confidence': avg_confidence,
            'image_results': image_results,
            'audio_results': audio_results,
            'video_results': video_results,
            'results': results
        }
        
        return template_data
    
    def _prepare_image_template_data(self, result: DetectionResult) -> Dict[str, Any]:
        """
        Prepare image-specific template data.
        
        Args:
            result: Image detection result
            
        Returns:
            Dictionary with template variables
        """
        # Extract image-specific information
        image_data = {
            'has_faces': any(region.get('type') == 'face' for region in result.regions_of_interest),
            'face_count': sum(1 for region in result.regions_of_interest if region.get('type') == 'face'),
            'manipulation_heatmap': result.visualization_data.get('heatmap_path', ''),
            'ela_visualization': result.visualization_data.get('ela_path', ''),
            'face_confidence_scores': [
                {'region_id': i, 'score': region.get('confidence', 0), 'bbox': region.get('bbox', [])}
                for i, region in enumerate(result.regions_of_interest)
                if region.get('type') == 'face'
            ]
        }
        
        return image_data
    
    def _prepare_audio_template_data(self, result: DetectionResult) -> Dict[str, Any]:
        """
        Prepare audio-specific template data.
        
        Args:
            result: Audio detection result
            
        Returns:
            Dictionary with template variables
        """
        # Extract audio-specific information
        audio_data = {
            'spectrogram_path': result.visualization_data.get('spectrogram_path', ''),
            'waveform_path': result.visualization_data.get('waveform_path', ''),
            'duration': result.metadata.get('duration', 0),
            'sample_rate': result.metadata.get('sample_rate', 0),
            'channels': result.metadata.get('channels', 0),
            'temporal_anomalies': [
                {'start_time': anomaly.get('start_time', 0), 
                 'end_time': anomaly.get('end_time', 0),
                 'confidence': anomaly.get('confidence', 0),
                 'description': anomaly.get('description', '')}
                for anomaly in result.analysis_details.get('temporal_anomalies', [])
            ]
        }
        
        return audio_data
    
    def _prepare_video_template_data(self, result: DetectionResult) -> Dict[str, Any]:
        """
        Prepare video-specific template data.
        
        Args:
            result: Video detection result
            
        Returns:
            Dictionary with template variables
        """
        # Extract video-specific information
        video_data = {
            'duration': result.metadata.get('duration', 0),
            'frame_count': result.metadata.get('frame_count', 0),
            'fps': result.metadata.get('fps', 0),
            'resolution': result.metadata.get('resolution', ''),
            'keyframe_paths': result.visualization_data.get('keyframe_paths', []),
            'temporal_analysis_path': result.visualization_data.get('temporal_analysis_path', ''),
            'sync_analysis_result': result.analysis_details.get('sync_analysis', {}),
            'frame_analysis': [
                {'frame_number': frame.get('frame_number', 0),
                 'timestamp': frame.get('timestamp', 0),
                 'confidence': frame.get('confidence', 0),
                 'manipulation_type': frame.get('manipulation_type', ''),
                 'regions': frame.get('regions', [])}
                for frame in result.analysis_details.get('frame_analysis', [])
            ]
        }
        
        return video_data