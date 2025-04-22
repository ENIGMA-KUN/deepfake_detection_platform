"""
Singularity Manager module for managing advanced detection modes.

This module manages all Singularity Modes - specialized algorithms that
combine multiple detection models with adaptive weighting and enhanced
visualization to achieve superior detection performance.
"""
from typing import Dict, List, Any, Optional
import logging

from app.core.singularity_modes import (
    SingularityMode, 
    VisualSentinel, 
    AcousticGuardian, 
    TemporalOracle
)

class SingularityManager:
    """
    Manager for Singularity Modes - advanced ensemble detection capabilities.
    
    Responsible for registering, tracking, and applying Singularity Modes
    for different media types. Acts as a central hub for enhanced detection.
    """
    
    def __init__(self):
        """Initialize the Singularity Manager with empty mode registry."""
        self.modes: Dict[str, Dict[str, SingularityMode]] = {
            'image': {},
            'audio': {},
            'video': {}
        }
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Singularity Manager")
        
        # Register default Singularity Modes
        self.register_default_modes()
    
    def register_default_modes(self):
        """Register the standard Singularity Modes for each media type."""
        # Image mode: Visual Sentinel
        visual_sentinel = VisualSentinel()
        self.register_mode('image', visual_sentinel)
        
        # Audio mode: Acoustic Guardian
        acoustic_guardian = AcousticGuardian()
        self.register_mode('audio', acoustic_guardian)
        
        # Video mode: Temporal Oracle
        temporal_oracle = TemporalOracle()
        self.register_mode('video', temporal_oracle)
        
        self.logger.info(f"Registered default Singularity Modes: {', '.join([m.name for m in [visual_sentinel, acoustic_guardian, temporal_oracle]])}")
    
    def register_mode(self, media_type: str, mode: SingularityMode):
        """
        Register a Singularity Mode for a specific media type.
        
        Args:
            media_type: Type of media ('image', 'audio', 'video')
            mode: SingularityMode instance to register
        
        Raises:
            ValueError: If media_type is invalid or mode is not a SingularityMode
        """
        if media_type not in self.modes:
            raise ValueError(f"Invalid media type: {media_type}. Must be one of: {', '.join(self.modes.keys())}")
        
        if not isinstance(mode, SingularityMode):
            raise ValueError(f"Mode must be a SingularityMode instance, got {type(mode)}")
        
        self.modes[media_type][mode.name] = mode
        self.logger.info(f"Registered {mode.name} for {media_type} media")
    
    def get_available_modes(self, media_type: str) -> List[Dict[str, str]]:
        """
        Get all available Singularity Modes for a specific media type.
        
        Args:
            media_type: Type of media ('image', 'audio', 'video')
            
        Returns:
            List of dictionaries with mode name and description
            
        Raises:
            ValueError: If media_type is invalid
        """
        if media_type not in self.modes:
            raise ValueError(f"Invalid media type: {media_type}. Must be one of: {', '.join(self.modes.keys())}")
        
        return [
            {
                'name': mode.name,
                'description': mode.description
            }
            for mode in self.modes[media_type].values()
        ]
    
    def apply_mode(self, media_type: str, mode_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a specific Singularity Mode to enhance detection results.
        
        Args:
            media_type: Type of media ('image', 'audio', 'video')
            mode_name: Name of the Singularity Mode to apply
            result: Original detection result to enhance
            
        Returns:
            Enhanced detection result
            
        Raises:
            ValueError: If media_type or mode_name is invalid
        """
        if media_type not in self.modes:
            raise ValueError(f"Invalid media type: {media_type}. Must be one of: {', '.join(self.modes.keys())}")
        
        if not result:
            self.logger.warning("Cannot apply Singularity Mode to empty result")
            return result
            
        # If mode_name is not specified, use default for this media type
        if not mode_name and len(self.modes[media_type]) > 0:
            mode_name = next(iter(self.modes[media_type].keys()))
            
        if mode_name not in self.modes[media_type]:
            available_modes = ', '.join(self.modes[media_type].keys())
            self.logger.warning(f"Invalid mode name: {mode_name}. Available modes: {available_modes}")
            return result
            
        mode = self.modes[media_type][mode_name]
        self.logger.info(f"Applying {mode.name} to {media_type} detection result")
        
        try:
            enhanced_result = mode.process(result)
            enhanced_result['singularity_applied'] = True
            enhanced_result['singularity_mode'] = mode_name
            return enhanced_result
        except Exception as e:
            self.logger.error(f"Error applying {mode.name}: {str(e)}")
            # Return original result on error
            return result
