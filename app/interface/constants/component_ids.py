"""
Component ID constants for the deepfake detection platform.

This module provides a central place to define all component IDs used in the application.
This helps prevent mismatched component IDs in callbacks and ensures consistency.

Usage:
    from app.interface.constants.component_ids import UPLOAD_IMAGE, ANALYZE_IMAGE_BUTTON
    
    # In component definition
    upload_component = dcc.Upload(id=UPLOAD_IMAGE)
    
    # In callback
    @app.callback(
        Output(IMAGE_RESULT_CONTAINER, 'children'),
        Input(ANALYZE_IMAGE_BUTTON, 'n_clicks')
    )
"""

# Tab IDs
TAB_HOME = 'tab-home'
TAB_IMAGE = 'tab-image'
TAB_AUDIO = 'tab-audio'
TAB_VIDEO = 'tab-video'
TAB_REPORTS = 'tab-reports'
TAB_SETTINGS = 'tab-settings'
TABS = 'tabs'  # Main tabs component

# Upload Components
UPLOAD_IMAGE = 'upload-image'
UPLOAD_AUDIO = 'upload-audio'
UPLOAD_VIDEO = 'upload-video'
UPLOAD_OUTPUT = 'upload-output'

# Upload Output Stores
IMAGE_UPLOAD_OUTPUT = 'image-upload-output'
AUDIO_UPLOAD_OUTPUT = 'audio-upload-output'
VIDEO_UPLOAD_OUTPUT = 'video-upload-output'

# Analysis Buttons
ANALYZE_IMAGE_BUTTON = 'analyze-image-button'
ANALYZE_AUDIO_BUTTON = 'analyze-audio-button'
ANALYZE_VIDEO_BUTTON = 'analyze-video-button'
CLEAR_IMAGE_BUTTON = 'clear-image-button'
CLEAR_AUDIO_BUTTON = 'clear-audio-button'
CLEAR_VIDEO_BUTTON = 'clear-video-button'

# Home Navigation Buttons
HOME_IMAGE_BUTTON = 'home-image-button'
HOME_AUDIO_BUTTON = 'home-audio-button'
HOME_VIDEO_BUTTON = 'home-video-button'

# Model Selection Dropdowns
IMAGE_MODEL_DROPDOWN = 'image-model-dropdown'
AUDIO_MODEL_DROPDOWN = 'audio-model-dropdown'
VIDEO_MODEL_DROPDOWN = 'video-model-dropdown'
MODEL_DROPDOWN_OUTPUT = 'model-dropdown-output'

# Threshold Sliders
IMAGE_THRESHOLD_SLIDER = 'image-threshold-slider'
AUDIO_THRESHOLD_SLIDER = 'audio-threshold-slider'
VIDEO_THRESHOLD_SLIDER = 'video-threshold-slider'
THRESHOLD_OUTPUT = 'threshold-output'

# Results Containers
IMAGE_RESULT_CONTAINER = 'image-result-container'
AUDIO_RESULT_CONTAINER = 'audio-result-container'
VIDEO_RESULT_CONTAINER = 'video-result-container'
RESULT_CONFIDENCE_GAUGE = 'result-confidence-gauge'
RESULT_AUTHENTIC_LABEL = 'result-authentic-label'
RESULT_DEEPFAKE_LABEL = 'result-deepfake-label'
RESULT_TIMESTAMP = 'result-timestamp'
RESULT_ANALYSIS_TIME = 'result-analysis-time'

# Results Sections
IMAGE_RESULTS_SECTION = 'image-results-section'
AUDIO_RESULTS_SECTION = 'audio-results-section'
VIDEO_RESULTS_SECTION = 'video-results-section'

# Analysis Modes
IMAGE_ANALYSIS_MODE = 'image-analysis-mode'
AUDIO_ANALYSIS_MODE = 'audio-analysis-mode'
VIDEO_ANALYSIS_MODE = 'video-analysis-mode'

# Visualization Components
IMAGE_HEATMAP = 'image-heatmap'
AUDIO_SPECTROGRAM = 'audio-spectrogram'
VIDEO_FRAME_GRID = 'video-frame-grid'
VIDEO_TIMELINE = 'video-timeline'
FACE_BOUNDING_BOXES = 'face-bounding-boxes'
ANOMALY_REGIONS = 'anomaly-regions'

# Model Technical Details
VIT_TECHNICAL_COLLAPSE = 'vit-technical-collapse'
BEIT_TECHNICAL_COLLAPSE = 'beit-technical-collapse'
DEIT_TECHNICAL_COLLAPSE = 'deit-technical-collapse'
SWIN_TECHNICAL_COLLAPSE = 'swin-technical-collapse'
WAV2VEC_TECHNICAL_COLLAPSE = 'wav2vec-technical-collapse'
XLSR_TECHNICAL_COLLAPSE = 'xlsr-technical-collapse'
MAMBA_TECHNICAL_COLLAPSE = 'mamba-technical-collapse'
TIMESFORMER_TECHNICAL_COLLAPSE = 'timesformer-technical-collapse'
GENCONVIT_TECHNICAL_COLLAPSE = 'genconvit-technical-collapse'
VIDEO_SWIN_TECHNICAL_COLLAPSE = 'video-swin-technical-collapse'

# Technical Detail Toggles
TOGGLE_VIT_DETAILS = 'toggle-vit-details'
TOGGLE_BEIT_DETAILS = 'toggle-beit-details'
TOGGLE_DEIT_DETAILS = 'toggle-deit-details'
TOGGLE_SWIN_DETAILS = 'toggle-swin-details'
TOGGLE_WAV2VEC_DETAILS = 'toggle-wav2vec-details'
TOGGLE_XLSR_DETAILS = 'toggle-xlsr-details'
TOGGLE_MAMBA_DETAILS = 'toggle-mamba-details'
TOGGLE_TIMESFORMER_DETAILS = 'toggle-timesformer-details'
TOGGLE_GENCONVIT_DETAILS = 'toggle-genconvit-details'
TOGGLE_VIDEO_SWIN_DETAILS = 'toggle-video-swin-details'

# About Sections
IMAGE_ABOUT_COLLAPSE = 'image-about-collapse'
AUDIO_ABOUT_COLLAPSE = 'audio-about-collapse'
VIDEO_ABOUT_COLLAPSE = 'video-about-collapse'
TOGGLE_IMAGE_ABOUT = 'toggle-image-about'
TOGGLE_AUDIO_ABOUT = 'toggle-audio-about'
TOGGLE_VIDEO_ABOUT = 'toggle-video-about'

# Export/Report Components
EXPORT_REPORT_BUTTON = 'export-report-button'
DOWNLOAD_JSON_BUTTON = 'download-json-button'
REPORT_PREVIEW = 'report-preview'
REPORT_EXPORT_STATUS = 'report-export-status'
EXPORT_REPORT_PDF = 'export-report-pdf'
DOWNLOAD_REPORT = 'download-report'
DOWNLOAD_REPORT_VIS = 'download-report-vis'
DELETE_REPORT = 'delete-report'

# Report Details
REPORT_LIST = 'report-list'
REPORT_TITLE = 'report-title'
REPORT_DATE = 'report-date'
REPORT_DATE_RANGE = 'report-date-range'
REPORT_FILENAME = 'report-filename'
REPORT_MEDIA_TYPE = 'report-media-type'
REPORT_VERDICT = 'report-verdict'
REPORT_CONFIDENCE = 'report-confidence'
REPORT_DETAILS = 'report-details'
REPORT_DETAILS_VIEW = 'report-details-view'
REPORT_VISUALIZATION = 'report-visualization'
REPORT_FILTER_MEDIA = 'report-filter-media'
REPORT_FILTER_RESULT = 'report-filter-result'
REPORT_SEARCH = 'report-search'
REPORT_SORT = 'report-sort'
NO_REPORT_SELECTED = 'no-report-selected'

# Report Dynamic Items (prefix for generated IDs)
REPORT_ITEM_PREFIX = 'report-item-'

# View Report Buttons
IMAGE_VIEW_REPORT = 'image-view-report'
AUDIO_VIEW_REPORT = 'audio-view-report'
VIDEO_VIEW_REPORT = 'video-view-report'

# State Management
APP_STATE_STORE = 'app-state-store'
ANALYSIS_STATUS = 'analysis-status'
ERROR_TOAST = 'error-toast'
SUCCESS_TOAST = 'success-toast'
MAIN_CONTAINER = 'main-container'
REPORTS_TAB_CONTENT = 'reports-tab-content'
SELECTED_REPORT_STORE = 'selected-report-store'

# Upload Stores
IMAGE_UPLOAD_STORE = 'image-upload-store'
AUDIO_UPLOAD_STORE = 'audio-upload-store'
VIDEO_UPLOAD_STORE = 'video-upload-store'

# Settings Components
API_KEY_INPUT = 'api-key-input'
API_KEY_SAVE_BUTTON = 'api-key-save-button'
THEME_TOGGLE = 'theme-toggle'
CACHE_CLEAR_BUTTON = 'cache-clear-button'

# Known mismatched IDs (for reference during migration)
# Once fixed, these should be removed
LEGACY_COMPONENT_IDS = {
    # Upload Components
    'upload-image-data': UPLOAD_IMAGE,
    'upload-audio-component': UPLOAD_AUDIO,
    'upload-video-component': UPLOAD_VIDEO,
    
    # Result Containers
    'image-result': IMAGE_RESULT_CONTAINER,
    'audio-result-div': AUDIO_RESULT_CONTAINER,
    'video-result-display': VIDEO_RESULT_CONTAINER,
    
    # Analysis Controls
    'analyze-img-btn': ANALYZE_IMAGE_BUTTON,
    'audio-analyze-button': ANALYZE_AUDIO_BUTTON,
    'video-analyze-button': ANALYZE_VIDEO_BUTTON,
    
    # Model Dropdowns
    'dropdown-image-model': IMAGE_MODEL_DROPDOWN,
    'dropdown-audio-model': AUDIO_MODEL_DROPDOWN,
    'dropdown-video-model': VIDEO_MODEL_DROPDOWN,
    
    # Threshold Sliders
    'image-confidence-threshold-slider': IMAGE_THRESHOLD_SLIDER,
    'audio-confidence-threshold-slider': AUDIO_THRESHOLD_SLIDER,
    'video-confidence-threshold-slider': VIDEO_THRESHOLD_SLIDER,
    
    # Upload Outputs
    'output-image-upload': IMAGE_UPLOAD_OUTPUT,
    'output-audio-upload': AUDIO_UPLOAD_OUTPUT,
    'output-video-upload': VIDEO_UPLOAD_OUTPUT,
    
    # Upload Stores
    'uploaded-image-store': IMAGE_UPLOAD_STORE,
    'uploaded-audio-store': AUDIO_UPLOAD_STORE,
    'uploaded-video-store': VIDEO_UPLOAD_STORE,
    
    # Report Items  
    'report-item-rep_001': f"{REPORT_ITEM_PREFIX}001",
    'report-item-rep_002': f"{REPORT_ITEM_PREFIX}002",
    'report-item-rep_003': f"{REPORT_ITEM_PREFIX}003",
}


def get_legacy_id_mapping():
    """
    Returns a mapping of legacy component IDs to their new standardized versions.
    Used during transition to help locate mismatched IDs.
    
    Returns:
        dict: Mapping of old IDs to new IDs
    """
    return LEGACY_COMPONENT_IDS

def get_all_component_ids():
    """
    Returns a list of all standardized component IDs.
    
    Returns:
        list: All component IDs defined in this module
    """
    # Get all uppercase variables which are component IDs
    return [value for name, value in globals().items() 
            if name.isupper() and isinstance(value, str) and name != 'LEGACY_COMPONENT_IDS']

def get_report_item_id(report_id):
    """
    Generate a standardized report item ID for a specific report.
    
    Args:
        report_id: Unique identifier for the report
        
    Returns:
        str: Standardized report item ID
    """
    return f"{REPORT_ITEM_PREFIX}{report_id}"
