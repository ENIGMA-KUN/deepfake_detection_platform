"""
Web application interface for the Deepfake Detection Platform.
"""
import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import yaml
from typing import Dict, Any

from app.core.processor import MediaProcessor
from app.interface.layout_manager import create_app_layout
from app.interface.callback_manager import register_callbacks
from app.utils.visualization import VisualizationManager

def create_app(processor: MediaProcessor, config: Dict[str, Any]) -> dash.Dash:
    """
    Create the Dash application instance.
    
    Args:
        processor: Media processor instance
        config: Application configuration
        
    Returns:
        Dash application instance
    """
    # Initialize Dash app with Tron-inspired theme
    assets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    # Extract UI-related config
    debug_mode = config['general'].get('debug_mode', False)
    
    # Create the app instance
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],  # Dark theme as a base
        assets_folder=assets_folder,
        title="Deepfake Detection Platform",
        suppress_callback_exceptions=True,
        prevent_initial_callbacks=False  # Explicit setting to avoid attribute error
    )
    
    # Configure Flask server limits (important for large base64 uploads via dcc.Upload)
    # Accept up to 25 MB unless overridden in config.ui.max_upload_size_mb
    default_upload_mb = 25
    max_upload_mb = config.get('ui', {}).get('max_upload_size_mb', default_upload_mb)
    try:
        app.server.config['MAX_CONTENT_LENGTH'] = int(max_upload_mb) * 1024 * 1024
    except Exception:
        # Fallback – shouldn’t block app startup if config type mismatch
        app.server.config['MAX_CONTENT_LENGTH'] = default_upload_mb * 1024 * 1024

    # Attach the processor and config as custom attributes
    app._processor = processor
    app._app_config = config
    
    # Define custom CSS for Tron-inspired theme
    app.index_string = load_custom_index_string()
    
    # Create and set the app layout
    app.layout = create_app_layout(app, processor, config)
    
    # Register all callbacks
    register_callbacks(app)
    
    return app

def load_custom_index_string() -> str:
    """
    Load the custom HTML index string with Tron-inspired CSS.
    
    Returns:
        HTML string with custom CSS
    """
    return '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background-color: #0F1924;
                    font-family: 'Roboto', sans-serif;
                    color: #E0F7FF;
                }
                
                h1, h2, h3, h4, h5, h6 {
                    font-family: 'Orbitron', sans-serif;
                }
                
                h4, h5 {
                    background: linear-gradient(90deg, #FFFFFF 0%, #B3E5FC 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
                    font-weight: 700;
                    letter-spacing: 0.5px;
                }
                
                .card {
                    background-color: #1E2A38;
                    border: 1px solid #1E5F75;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(30, 95, 117, 0.2);
                }
                
                .nav-link {
                    color: #8FBCBB;
                }
                
                .nav-link.active {
                    color: #4CC9F0 !important;
                    border-bottom: 2px solid #4CC9F0;
                    background-color: rgba(76, 201, 240, 0.1) !important;
                }
                
                .form-control, .form-select {
                    background-color: #2D3748;
                    border: 1px solid #1E5F75;
                    color: #E0F7FF;
                }
                
                .form-control:focus, .form-select:focus {
                    background-color: #2D3748;
                    border-color: #4CC9F0;
                    box-shadow: 0 0 0 0.25rem rgba(76, 201, 240, 0.25);
                    color: #E0F7FF;
                }
                
                .btn-primary {
                    background-color: #4CC9F0;
                    border-color: #4CC9F0;
                }
                
                .btn-primary:hover {
                    background-color: #3EAFDB;
                    border-color: #3EAFDB;
                }
                
                .btn-danger {
                    background-color: #F72585;
                    border-color: #F72585;
                }
                
                .btn-danger:hover {
                    background-color: #E01D77;
                    border-color: #E01D77;
                }
                
                .btn-outline-info {
                    border-color: #4CC9F0;
                    color: #4CC9F0;
                }
                
                .btn-outline-info:hover {
                    background-color: #4CC9F0;
                    color: #0F1924;
                }
                
                .btn-outline-secondary {
                    border-color: #8FBCBB;
                    color: #8FBCBB;
                }
                
                .btn-outline-secondary:hover {
                    background-color: #8FBCBB;
                    color: #0F1924;
                }
                
                .text-glow {
                    text-shadow: 0 0 10px rgba(76, 201, 240, 0.5);
                }
                
                .animate-glow {
                    animation: glow 2s infinite alternate;
                }
                
                @keyframes glow {
                    0% {
                        box-shadow: 0 0 5px rgba(76, 201, 240, 0.5);
                    }
                    100% {
                        box-shadow: 0 0 20px rgba(76, 201, 240, 0.8);
                    }
                }
                
                /* Custom tabs */
                .custom-tabs .nav-item .nav-link {
                    color: #4CC9F0;
                    border: none;
                    border-bottom: 2px solid transparent;
                    background-color: transparent;
                    transition: all 0.3s;
                    margin-right: 10px;
                    padding: 10px 15px;
                }
                
                .custom-tabs .nav-item .nav-link:hover {
                    color: #7DF9FF;
                    border-bottom: 2px solid rgba(76, 201, 240, 0.3);
                }
                
                .custom-tabs .nav-item .nav-link.active {
                    color: #7DF9FF;
                    border-bottom: 2px solid #4CC9F0;
                    background-color: rgba(76, 201, 240, 0.1);
                }
                
                /* Upload container */
                .upload-container {
                    width: 100%;
                    height: 150px;
                    border-width: 2px;
                    border-style: dashed;
                    border-radius: 5px;
                    text-align: center;
                    padding: 20px;
                    background-color: rgba(76, 201, 240, 0.05);
                    border-color: #4CC9F0;
                    transition: all 0.3s;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }
                
                .upload-container:hover {
                    background-color: rgba(76, 201, 240, 0.1);
                    border-color: #7DF9FF;
                }
                
                /* Loading spinner */
                .loading-spinner {
                    border: 4px solid rgba(76, 201, 240, 0.1);
                    border-radius: 50%;
                    border-top: 4px solid #4CC9F0;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .loading-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 30px 0;
                }
                
                .loading-text {
                    margin-top: 20px;
                    color: #4CC9F0;
                }
                
                /* Results styling */
                .results-header {
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }
                
                .results-title {
                    margin: 0;
                    margin-right: 15px;
                }
                
                .results-verdict {
                    padding: 5px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                    font-family: 'Orbitron', sans-serif;
                    letter-spacing: 1px;
                }
                
                .verdict-authentic {
                    background-color: rgba(16, 172, 132, 0.2);
                    border: 1px solid #10ac84;
                    color: #10ac84;
                }
                
                .verdict-deepfake {
                    background-color: rgba(247, 37, 133, 0.2);
                    border: 1px solid #F72585;
                    color: #F72585;
                }
                
                /* Confidence meter */
                .confidence-meter {
                    width: 100%;
                    height: 10px;
                    background-color: rgba(76, 201, 240, 0.2);
                    border-radius: 5px;
                    margin: 10px 0;
                    position: relative;
                }
                
                .confidence-fill {
                    height: 100%;
                    border-radius: 5px;
                    background: linear-gradient(90deg, #4CC9F0 0%, #F72585 100%);
                    width: 0%; /* Will be set dynamically */
                    transition: width 1s ease-in-out;
                }
                
                .confidence-threshold {
                    position: absolute;
                    top: -5px;
                    height: 20px;
                    width: 2px;
                    background-color: rgba(255, 255, 255, 0.7);
                    left: 50%; /* Default threshold at 0.5 */
                }
                
                /* Visualization container */
                .visualization-container {
                    background-color: rgba(76, 201, 240, 0.05);
                    border: 1px solid rgba(76, 201, 240, 0.2);
                    border-radius: 5px;
                    padding: 15px;
                }
                
                /* About card */
                .about-card {
                    background-color: rgba(76, 201, 240, 0.05);
                }
            </style>
            {%scripts%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''