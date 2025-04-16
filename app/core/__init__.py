#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core module for the Deepfake Detection Platform.
Contains the main processing logic and workflow management.
"""

from app.core.processor import (BatchProcessor, MediaProcessor, ProcessingStatus,
                               ProcessingTask)
from app.core.queue_manager import ProgressTracker, QueueManager
from app.core.result_handler import (ReportGenerator, ResultFormat, ResultHandler)

__all__ = [
    'MediaProcessor',
    'BatchProcessor',
    'ProcessingStatus',
    'ProcessingTask',
    'QueueManager',
    'ProgressTracker',
    'ResultHandler',
    'ResultFormat',
    'ReportGenerator'
]