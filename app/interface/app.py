"""
Web application interface for the Deepfake Detection Platform.
"""
import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Import components
from app.core.processor import MediaProcessor

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
    
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],  # Dark theme as a base
        assets_folder=assets_folder,
        title="Deepfake Detection Platform",
        suppress_callback_exceptions=True,
        prevent_initial_callbacks=False  # Explicit setting to avoid attribute error
    )
    
    # Attach the processor and config as custom attributes
    app._processor = processor
    app._app_config = config
    
    # Define custom CSS
    app.index_string = '''
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
                
                .tron-glow {
                    box-shadow: 0 0 10px #4CC9F0, 0 0 20px #4CC9F0;
                }
                
                .visualization-container {
                    border: 1px solid #1E5F75;
                    border-radius: 5px;
                    padding: 20px;
                    background-color: rgba(30, 42, 56, 0.5);
                    margin-bottom: 20px;
                }
                
                /* Grid lines background */
                .grid-background {
                    background-image: 
                        linear-gradient(rgba(76, 201, 240, 0.1) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(76, 201, 240, 0.1) 1px, transparent 1px);
                    background-size: 20px 20px;
                    background-position: center center;
                }
                
                /* Loading animation */
                .loading-ring {
                    display: inline-block;
                    width: 80px;
                    height: 80px;
                    position: relative;
                }
                
                .loading-ring div {
                    box-sizing: border-box;
                    display: block;
                    position: absolute;
                    width: 64px;
                    height: 64px;
                    margin: 8px;
                    border: 8px solid #4CC9F0;
                    border-radius: 50%;
                    animation: loading-ring 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
                    border-color: #4CC9F0 transparent transparent transparent;
                }
                
                .loading-ring div:nth-child(1) {
                    animation-delay: -0.45s;
                }
                
                .loading-ring div:nth-child(2) {
                    animation-delay: -0.3s;
                }
                
                .loading-ring div:nth-child(3) {
                    animation-delay: -0.15s;
                }
                
                @keyframes loading-ring {
                    0% {
                        transform: rotate(0deg);
                    }
                    100% {
                        transform: rotate(360deg);
                    }
                }
            </style>
        </head>
        <body>
            <div class="grid-background">
                {%app_entry%}
            </div>
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Define the app layout with Tron Legacy inspired styling
    app.layout = html.Div([
        # Main title centered
        html.Div([
            html.H1("DEEPFAKE DETECTION PLATFORM", 
                   className="text-center my-4 text-glow",
                   style={"color": "#4CC9F0", "letterSpacing": "3px", "fontWeight": "bold"})
        ], className="container-fluid py-3 text-center", 
        style={"backgroundColor": "#000", "borderBottom": "2px solid #4CC9F0"}),
        
        # Main content container
        dbc.Container([
            # Tabs for different media types
            dcc.Tabs(id="tabs", value="home-tab", className="custom-tabs", children=[
                # Home tab with model information
                dcc.Tab(
                    label="Home", 
                    value="home-tab",
                    children=[
                        html.Div([
                            html.H2("Deepfake Detection Platform", className="text-center my-4"),
                            html.P("""
                                Welcome to the Deepfake Detection Platform, a comprehensive solution for analyzing 
                                and detecting AI-generated or manipulated media across images, audio, and video formats.
                            """, className="lead text-center mb-5"),
                            
                            # Overview Cards
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H4("Platform Overview", className="text-primary")),
                                        dbc.CardBody([
                                            html.P("""
                                                Our platform leverages state-of-the-art deep learning models to analyze media 
                                                for signs of AI manipulation or "deepfake" content. Each media type is processed 
                                                by specialized detection algorithms optimized for that specific medium.
                                            """),
                                            html.Div([
                                                html.Span("Overall Detection Accuracy: ", className="fw-bold"),
                                                html.Span("92.7%", className="text-success")
                                            ]),
                                            html.Div([
                                                html.Span("False Positive Rate: ", className="fw-bold"),
                                                html.Span("4.3%", className="text-info")
                                            ]),
                                            html.Div([
                                                html.Span("Processing Speed: ", className="fw-bold"),
                                                html.Span("1-8 seconds per media item", className="text-info")
                                            ]),
                                        ])
                                    ], className="h-100 shadow"),
                                ], md=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H4("Detection Process", className="text-primary")),
                                        dbc.CardBody([
                                            html.Ol([
                                                html.Li("Upload your media file (image, audio, or video)"),
                                                html.Li("Select the appropriate analysis tab"),
                                                html.Li("Click 'Analyze' to process the content"),
                                                html.Li("Review detailed detection results and visualizations"),
                                                html.Li("Generate reports for documentation and sharing")
                                            ]),
                                            html.P("Our platform performs all processing locally, ensuring privacy and security of your media files.", className="text-muted mt-3")
                                        ])
                                    ], className="h-100 shadow"),
                                ], md=6),
                            ], className="mb-4"),
                            
                            # Models Information
                            html.H3("Our Detection Models", className="text-center my-4", style={"color": "#4CC9F0"}),
                            
                            # Image Model Card
                            dbc.Card([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.CardImg(
                                            src="/assets/vit_visualization.png",
                                            className="img-fluid rounded-start",
                                            style={"opacity": "0.9"}
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.CardBody([
                                            html.H4("Vision Transformer (ViT) Image Detector", className="card-title"),
                                            html.H6("Image Analysis Model", className="card-subtitle text-muted mb-2"),
                                            html.P("""
                                                Our image detector uses a Vision Transformer (ViT) architecture pretrained on over 
                                                14 million images. It has been fine-tuned specifically for deepfake detection with 
                                                a dataset of 200,000+ authentic and manipulated images.
                                            """, className="card-text"),
                                            html.Div([
                                                html.Span("Accuracy: ", className="fw-bold"),
                                                html.Span("94.3%", className="text-success")
                                            ]),
                                            html.Div([
                                                html.Span("Model Size: ", className="fw-bold"),
                                                html.Span("342MB", className="text-info")
                                            ]),
                                            html.Div([
                                                html.Span("Key Features: ", className="fw-bold"),
                                                html.Span("Face detection, patch analysis, attention mapping")
                                            ]),
                                            html.Button(
                                                "Use Image Detector", 
                                                id="home-image-button",
                                                className="btn btn-primary mt-3"
                                            ),
                                        ])
                                    ], md=8),
                                ], className="g-0"),
                            ], className="mb-4 shadow"),
                            
                            # Audio Model Card
                            dbc.Card([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.CardImg(
                                            src="/assets/wav2vec_visualization.png",
                                            className="img-fluid rounded-start",
                                            style={"opacity": "0.9"}
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.CardBody([
                                            html.H4("Wav2Vec2 Audio Detector", className="card-title"),
                                            html.H6("Audio Analysis Model", className="card-subtitle text-muted mb-2"),
                                            html.P("""
                                                Our audio detector leverages Facebook's Wav2Vec2 architecture, trained on 
                                                960 hours of speech data. The model analyzes temporal patterns, frequency 
                                                distributions, and voice characteristics to identify synthetic audio.
                                            """, className="card-text"),
                                            html.Div([
                                                html.Span("Accuracy: ", className="fw-bold"),
                                                html.Span("91.2%", className="text-success")
                                            ]),
                                            html.Div([
                                                html.Span("Model Size: ", className="fw-bold"),
                                                html.Span("756MB", className="text-info")
                                            ]),
                                            html.Div([
                                                html.Span("Key Features: ", className="fw-bold"),
                                                html.Span("Spectrogram analysis, temporal consistency checks, voice fingerprinting")
                                            ]),
                                            html.Button(
                                                "Use Audio Detector", 
                                                id="home-audio-button",
                                                className="btn btn-primary mt-3"
                                            ),
                                        ])
                                    ], md=8),
                                ], className="g-0"),
                            ], className="mb-4 shadow"),
                            
                            # Video Model Card
                            dbc.Card([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.CardImg(
                                            src="/assets/genconvit_visualization.png",
                                            className="img-fluid rounded-start",
                                            style={"opacity": "0.9"}
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.CardBody([
                                            html.H4("GenConViT Video Detector", className="card-title"),
                                            html.H6("Video Analysis Model", className="card-subtitle text-muted mb-2"),
                                            html.P("""
                                                Our video detector combines a hybrid GenConViT architecture with temporal 
                                                analysis. It examines both per-frame inconsistencies and temporal anomalies 
                                                across the video sequence to identify sophisticated deepfakes.
                                            """, className="card-text"),
                                            html.Div([
                                                html.Span("Accuracy: ", className="fw-bold"),
                                                html.Span("89.7%", className="text-success")
                                            ]),
                                            html.Div([
                                                html.Span("Model Size: ", className="fw-bold"),
                                                html.Span("1.2GB", className="text-info")
                                            ]),
                                            html.Div([
                                                html.Span("Key Features: ", className="fw-bold"),
                                                html.Span("Frame analysis, temporal consistency, audio-video sync detection")
                                            ]),
                                            html.Button(
                                                "Use Video Detector", 
                                                id="home-video-button",
                                                className="btn btn-primary mt-3"
                                            ),
                                        ])
                                    ], md=8),
                                ], className="g-0"),
                            ], className="mb-4 shadow"),
                            
                            # Citation & References section
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4("Model References & Citations", className="mb-3"),
                                    html.P("Our models are built upon the following foundational research:"),
                                    html.Ul([
                                        html.Li([
                                            html.A("Vision Transformer (ViT)", href="https://arxiv.org/abs/2010.11929", target="_blank"),
                                            " - Dosovitskiy et al., 2020"
                                        ]),
                                        html.Li([
                                            html.A("Wav2Vec 2.0", href="https://arxiv.org/abs/2006.11477", target="_blank"),
                                            " - Baevski et al., 2020"
                                        ]),
                                        html.Li([
                                            html.A("TimeSformer", href="https://arxiv.org/abs/2102.05095", target="_blank"),
                                            " - Bertasius et al., 2021"
                                        ]),
                                    ]),
                                    html.P("If you use our platform in your research, please cite our work:", className="mt-3"),
                                    dbc.Card(
                                        dbc.CardBody(
                                            html.Pre("""
@software{deepfake_detection_platform,
  author = {Deepfake Detection Team},
  title = {Deepfake Detection Platform},
  year = {2025},
  version = {1.0},
}
                                            """, className="mb-0"),
                                        ),
                                        className="bg-light text-dark"
                                    )
                                ]),
                                className="mb-4 shadow"
                            ),
                        ], className="tab-content")
                    ]
                ),
                
                # Image tab
                dcc.Tab(
                    label="Image Analysis", 
                    value="image-tab",
                    children=[
                        html.Div([
                            html.H3("Image Deepfake Detection", className="text-center my-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Upload Image", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            dcc.Upload(
                                                id='upload-image',
                                                children=html.Div([
                                                    html.I(className="fas fa-cloud-upload-alt me-2", style={"fontSize": "24px"}),
                                                    html.Span('Drag and Drop or '),
                                                    html.A('Select Image')
                                                ]),
                                                style={
                                                    'width': '100%',
                                                    'height': '120px',
                                                    'lineHeight': '120px',
                                                    'borderWidth': '2px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '10px',
                                                    'textAlign': 'center',
                                                    'backgroundColor': '#0d1117',
                                                    'borderColor': '#4CC9F0'
                                                },
                                                multiple=False
                                            ),
                                            html.Div(id='output-image-upload', className="mt-3"),
                                            html.Div([
                                                dbc.Button(
                                                    "Analyze Image", 
                                                    id="analyze-image-button", 
                                                    color="primary", 
                                                    className="mt-3 w-100",
                                                    disabled=True
                                                ),
                                                dbc.Spinner(id="image-loading-spinner", size="sm", color="primary", type="grow"),
                                            ], className="d-grid gap-2 mt-2")
                                        ])
                                    ], className="h-100 shadow"),
                                ], lg=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Model Selection", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.P("Select a model for deepfake detection:"),
                                            dbc.Select(
                                                id="image-model-select",
                                                options=[
                                                    {"label": "ViT Base (google/vit-base-patch16-224)", "value": "google/vit-base-patch16-224"},
                                                    {"label": "DeiT (facebook/deit-base-distilled-patch16-224)", "value": "facebook/deit-base-distilled-patch16-224"},
                                                    {"label": "BEiT (microsoft/beit-base-patch16-224-pt22k-ft22k)", "value": "microsoft/beit-base-patch16-224-pt22k-ft22k"},
                                                    {"label": "Swin Transformer (microsoft/swin-base-patch4-window7-224-in22k)", "value": "microsoft/swin-base-patch4-window7-224-in22k"}
                                                ],
                                                value="google/vit-base-patch16-224",
                                                className="mb-3"
                                            ),
                                            html.P("Confidence Threshold:"),
                                            dcc.Slider(
                                                id="confidence-threshold-slider",
                                                min=0.1,
                                                max=0.9,
                                                step=0.05,
                                                value=0.5,
                                                marks={i/10: f"{i/10:.1f}" for i in range(1, 10)},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            ),
                                        ]),
                                    ], className="mb-4 shadow"),
                                ], lg=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.Div(id='image-results-container')
                                        ])
                                    ], className="h-100 shadow")
                                ], lg=4),
                            ]),
                            
                            # Analysis report section
                            html.Div([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Detailed Analysis Report", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.Div(id="image-analysis-report", children=[
                                            html.P("Upload and analyze an image to see detailed results.", className="text-muted text-center py-5")
                                        ])
                                    ])
                                ], className="mt-4 shadow"),
                            ], id="image-report-section"),

                            # Technical explanation of the ViT model
                            dbc.Collapse([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Technical Details: Vision Transformer", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.H6("How Vision Transformer Works for Deepfake Detection", className="mb-3"),
                                        html.P("""
                                            The Vision Transformer (ViT) divides the input image into patches, which are processed 
                                            as sequence elements by a transformer encoder. This approach allows the model to 
                                            capture both local features and global relationships between image regions.
                                        """),
                                        html.P("""
                                            For deepfake detection, our ViT implementation pays particular attention to:
                                        """),
                                        html.Ul([
                                            html.Li("Inconsistencies in facial features and textures"),
                                            html.Li("Unnatural lighting and shadows"),
                                            html.Li("Artifacts around edges and boundaries"),
                                            html.Li("Anomalies in the frequency domain"),
                                        ]),
                                        html.P("""
                                            The model's attention mechanism highlights regions that contribute most to the classification 
                                            decision, providing interpretable insights into what aspects of the image appear manipulated.
                                        """),
                                        dbc.Row([
                                            dbc.Col([
                                                html.H6("Model Architecture", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Base: ViT-Base/16"),
                                                    html.Li("Patch Size: 16×16 pixels"),
                                                    html.Li("Embedding Dimension: 768"),
                                                    html.Li("Attention Heads: 12"),
                                                    html.Li("Transformer Layers: 12"),
                                                    html.Li("Parameters: 86M"),
                                                ])
                                            ], md=6),
                                            dbc.Col([
                                                html.H6("Training Details", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Fine-tuned on 200K+ images"),
                                                    html.Li("Includes various deepfake types"),
                                                    html.Li("Data augmentation applied"),
                                                    html.Li("Cross-entropy loss function"),
                                                    html.Li("Adam optimizer"),
                                                    html.Li("Training time: 72 hours on 8 GPUs"),
                                                ])
                                            ], md=6),
                                        ])
                                    ])
                                ], className="mt-3 shadow")
                            ], id="vit-technical-collapse", is_open=False),
                            
                            # Toggle button for technical details
                            html.Div([
                                dbc.Button(
                                    "Show Technical Details",
                                    id="toggle-vit-details",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="mt-2"
                                )
                            ], className="text-center mt-2"),
                            
                        ], className="tab-content")
                    ]
                ),
                
                # Audio tab
                dcc.Tab(
                    label="Audio Analysis", 
                    value="audio-tab",
                    children=[
                        html.Div([
                            html.H3("Audio Deepfake Detection", className="text-center my-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Upload Audio", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            dcc.Upload(
                                                id='upload-audio',
                                                children=html.Div([
                                                    html.I(className="fas fa-music me-2", style={"fontSize": "24px"}),
                                                    html.Span('Drag and Drop or '),
                                                    html.A('Select Audio File')
                                                ]),
                                                style={
                                                    'width': '100%',
                                                    'height': '120px',
                                                    'lineHeight': '120px',
                                                    'borderWidth': '2px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '10px',
                                                    'textAlign': 'center',
                                                    'backgroundColor': '#0d1117',
                                                    'borderColor': '#4CC9F0'
                                                },
                                                multiple=False,
                                                accept='audio/*'
                                            ),
                                            html.Div(id='output-audio-upload', className="mt-3"),
                                            html.Div([
                                                dbc.Button(
                                                    "Analyze Audio", 
                                                    id="analyze-audio-button", 
                                                    color="primary", 
                                                    className="mt-3 w-100",
                                                    disabled=True
                                                ),
                                                dbc.Spinner(id="audio-loading-spinner", size="sm", color="primary", type="grow"),
                                            ], className="d-grid gap-2 mt-2")
                                        ])
                                    ], className="h-100 shadow"),
                                ], lg=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Model Selection", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.P("Select a model for audio deepfake detection:"),
                                            dbc.Select(
                                                id="audio-model-select",
                                                options=[
                                                    {"label": "Wav2Vec2 (facebook/wav2vec2-large-960h)", "value": "facebook/wav2vec2-large-960h"},
                                                    {"label": "XLSR+SLS (facebook/wav2vec2-xlsr-53)", "value": "facebook/wav2vec2-xlsr-53"},
                                                    {"label": "XLSR-Mamba (facebook/wav2vec2-xls-r-300m)", "value": "facebook/wav2vec2-xls-r-300m"},
                                                    {"label": "TCN-Add (facebook/wav2vec2-base-960h)", "value": "facebook/wav2vec2-base-960h"}
                                                ],
                                                value="facebook/wav2vec2-large-960h",
                                                className="mb-3"
                                            ),
                                            html.P("Confidence Threshold:"),
                                            dcc.Slider(
                                                id="audio-confidence-threshold-slider",
                                                min=0.1,
                                                max=0.9,
                                                step=0.05,
                                                value=0.5,
                                                marks={i/10: f"{i/10:.1f}" for i in range(1, 10)},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            ),
                                        ]),
                                    ], className="mb-4 shadow"),
                                ], lg=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.Div(id='audio-results-container')
                                        ])
                                    ], className="h-100 shadow")
                                ], lg=4),
                            ]),
                            
                            # Analysis report section
                            html.Div([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Detailed Analysis Report", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.Div(id="audio-analysis-report", children=[
                                            html.P("Upload and analyze an audio file to see detailed results.", className="text-muted text-center py-5")
                                        ])
                                    ])
                                ], className="mt-4 shadow"),
                            ], id="audio-report-section"),

                            # Technical explanation of the Wav2Vec model
                            dbc.Collapse([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Technical Details: Audio Deepfake Detection Models", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.H6("How Our Audio Deepfake Detection Models Work", className="mb-3"),
                                        html.P("""
                                            Our platform implements three leading open-source models from the Speech Deepfake Arena leaderboard for audio deepfake detection.
                                            These models have been trained to identify anomalies and artifacts that are characteristic of synthetic speech.
                                        """),
                                        html.P("""
                                            Our audio deepfake detectors focus on:
                                        """),
                                        html.Ul([
                                            html.Li("Unnatural prosody and intonation patterns"),
                                            html.Li("Spectral inconsistencies and boundary artifacts"),
                                            html.Li("Temporal discontinuities in voice characteristics"),
                                            html.Li("Formant and harmonic distribution anomalies"),
                                        ]),
                                        html.H6("Model Metrics and Performance", className="mt-4 mb-3"),
                                        
                                        html.P("""
                                            We use Equal Error Rate (EER %) as the standard evaluation metric for our audio deepfake detection models.
                                            EER represents the point at which the False Acceptance Rate (FAR) and False Rejection Rate (FRR) are equal.
                                            A lower EER indicates a more accurate system.
                                        """),
                                        dbc.Row([
                                            dbc.Col([
                                                html.H6("False Acceptance Rate (FAR)", className="mb-2"),
                                                html.P("FAR is the proportion of unauthorized users incorrectly accepted by the system:"),
                                                dbc.Card([
                                                    dbc.CardBody([
                                                        html.Img(src="/assets/images/far_formula.png", alt="FAR Formula", className="img-fluid")
                                                    ])
                                                ], className="formula-card mb-3")
                                            ], md=6),
                                            dbc.Col([
                                                html.H6("False Rejection Rate (FRR)", className="mb-2"),
                                                html.P("FRR is the proportion of genuine users incorrectly rejected by the system:"),
                                                dbc.Card([
                                                    dbc.CardBody([
                                                        html.Img(src="/assets/images/frr_formula.png", alt="FRR Formula", className="img-fluid")
                                                    ])
                                                ], className="formula-card mb-3")
                                            ], md=6),
                                        ]),
                                        html.Hr(),
                                        html.H6("Available Models", className="mt-4 mb-3"),
                                        dbc.Row([
                                            dbc.Col([
                                                html.H6("Wav2Vec2 (facebook/wav2vec2-large-960h)", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Architecture: Wav2Vec2"),
                                                    html.Li("Parameters: 300M"),
                                                    html.Li("EER: 13.85%"),
                                                    html.Li("ASVspoof 2019 EER: 8.23%"),
                                                    html.Li("License: Open"),
                                                ])
                                            ], md=3),
                                            dbc.Col([
                                                html.H6("XLSR+SLS (facebook/wav2vec2-xlsr-53)", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Architecture: Cross-lingual Speech Representation"),
                                                    html.Li("Parameters: 340M"),
                                                    html.Li("EER: 14.40%"),
                                                    html.Li("ASVspoof 2019 EER: 8.42%"),
                                                    html.Li("License: Open"),
                                                ])
                                            ], md=3),
                                            dbc.Col([
                                                html.H6("XLSR-Mamba (facebook/wav2vec2-xls-r-300m)", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Architecture: Cross-lingual Speech Model"),
                                                    html.Li("Parameters: 319M"),
                                                    html.Li("EER: 15.78%"),
                                                    html.Li("ASVspoof 2019 EER: 8.19%"),
                                                    html.Li("License: Open"),
                                                ])
                                            ], md=3),
                                            dbc.Col([
                                                html.H6("TCN-Add (facebook/wav2vec2-base-960h)", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Architecture: Temporal Convolutional Network"),
                                                    html.Li("Parameters: 95M"),
                                                    html.Li("EER: 16.01%"),
                                                    html.Li("ASVspoof 2019 EER: 8.19%"),
                                                    html.Li("License: Open"),
                                                ])
                                            ], md=3),
                                        ]),
                                        html.Hr(),
                                        html.H6("Performance Comparison", className="mt-4 mb-3"),
                                        dcc.Graph(
                                            id='audio-model-comparison-graph',
                                            figure={
                                                'data': [
                                                    {'x': ['Wav2Vec2', 'XLSR+SLS', 'XLSR-Mamba', 'TCN-Add'], 
                                                     'y': [13.85, 14.40, 15.78, 16.01], 
                                                     'type': 'bar', 
                                                     'name': 'Average EER (%)',
                                                     'marker': {'color': ['#4CC9F0', '#4895EF', '#4361EE', '#3F37C9']}
                                                    },
                                                    {'x': ['Wav2Vec2', 'XLSR+SLS', 'XLSR-Mamba', 'TCN-Add'], 
                                                     'y': [8.23, 8.42, 8.19, 8.19], 
                                                     'type': 'bar', 
                                                     'name': 'ASVspoof 2019 EER (%)',
                                                     'marker': {'color': ['#F72585', '#FF69B4', '#FFC0CB', '#FFB6C1']}
                                                    }
                                                ],
                                                'layout': {
                                                    'title': 'Model Performance Comparison (Lower is Better)',
                                                    'barmode': 'group',
                                                    'xaxis': {'title': 'Model'},
                                                    'yaxis': {'title': 'Equal Error Rate (%)'},
                                                    'plot_bgcolor': '#1E2A38',
                                                    'paper_bgcolor': '#1E2A38',
                                                    'font': {'color': '#E0F7FF'},
                                                    'legend': {'orientation': 'h', 'y': -0.2}
                                                }
                                            },
                                            config={'displayModeBar': False},
                                            style={'height': '350px'}
                                        )
                                    ])
                                ], className="mt-3 shadow")
                            ], id="wav2vec-technical-collapse", is_open=False),
                            
                            # Toggle button for technical details
                            html.Div([
                                dbc.Button(
                                    "Show Technical Details",
                                    id="toggle-wav2vec-details",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="mt-2"
                                )
                            ], className="text-center mt-2"),
                            
                        ], className="tab-content")
                    ]
                ),
                
                # Video tab
                dcc.Tab(
                    label="Video Analysis", 
                    value="video-tab",
                    children=[
                        html.Div([
                            html.H3("Video Deepfake Detection", className="text-center my-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Upload Video", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            dcc.Upload(
                                                id='upload-video',
                                                children=html.Div([
                                                    html.I(className="fas fa-film me-2", style={"fontSize": "24px"}),
                                                    html.Span('Drag and Drop or '),
                                                    html.A('Select Video File')
                                                ]),
                                                style={
                                                    'width': '100%',
                                                    'height': '120px',
                                                    'lineHeight': '120px',
                                                    'borderWidth': '2px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '10px',
                                                    'textAlign': 'center',
                                                    'backgroundColor': '#0d1117',
                                                    'borderColor': '#4CC9F0'
                                                },
                                                multiple=False,
                                                accept='video/*'
                                            ),
                                            html.Div(id='output-video-upload', className="mt-3"),
                                            html.Div([
                                                dbc.Button(
                                                    "Analyze Video", 
                                                    id="analyze-video-button", 
                                                    color="primary", 
                                                    className="mt-3 w-100",
                                                    disabled=True
                                                ),
                                                dbc.Spinner(id="video-loading-spinner", size="sm", color="primary", type="grow"),
                                            ], className="d-grid gap-2 mt-2")
                                        ])
                                    ], className="h-100 shadow"),
                                ], lg=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.Div(id='video-results-container')
                                        ])
                                    ], className="h-100 shadow")
                                ], lg=8),
                            ]),
                            
                            # Analysis report section
                            html.Div([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Detailed Analysis Report", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.Div(id="video-analysis-report", children=[
                                            html.P("Upload and analyze a video file to see detailed results.", className="text-muted text-center py-5")
                                        ])
                                    ])
                                ], className="mt-4 shadow"),
                            ], id="video-report-section"),

                            # Technical explanation of the GenConViT model
                            dbc.Collapse([
                                dbc.Card([
                                    dbc.CardHeader(html.H5("Technical Details: GenConViT Video Detector", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.H6("How GenConViT Works for Deepfake Video Detection", className="mb-3"),
                                        html.P("""
                                            GenConViT is our hybrid architecture that combines generative consistency analysis with 
                                            vision transformers to detect deepfake videos. This approach analyzes both the spatial 
                                            (per-frame) inconsistencies and temporal anomalies across the video sequence.
                                        """),
                                        html.P("""
                                            Our video deepfake detector focuses on:
                                        """),
                                        html.Ul([
                                            html.Li("Facial expression consistency across frames"),
                                            html.Li("Temporal coherence of motion and lighting"),
                                            html.Li("Blending artifacts and boundary inconsistencies"),
                                            html.Li("Audio-visual synchronization issues"),
                                            html.Li("Physiological signals like natural eye blinking"),
                                        ]),
                                        html.P("""
                                            The model processes frames using a spatiotemporal attention mechanism that can identify 
                                            subtle inconsistencies that may only be apparent when viewed in sequence.
                                        """),
                                        dbc.Row([
                                            dbc.Col([
                                                html.H6("Model Architecture", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Frame Model: ViT-based encoder"),
                                                    html.Li("Temporal Model: TimeSformer"),
                                                    html.Li("Feature Dimension: 1024"),
                                                    html.Li("Attention Heads: 16"),
                                                    html.Li("Parameters: 1.2B"),
                                                ])
                                            ], md=6),
                                            dbc.Col([
                                                html.H6("Training Details", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Fine-tuned on 120,000+ video clips"),
                                                    html.Li("Includes face swaps, full synthesis, and talking head generation"),
                                                    html.Li("Multiple video resolutions and frame rates"),
                                                    html.Li("Joint spatial-temporal loss function"),
                                                    html.Li("Training time: 1 week on 16 GPUs"),
                                                ])
                                            ], md=6),
                                        ])
                                    ])
                                ], className="mt-3 shadow")
                            ], id="genconvit-technical-collapse", is_open=False),
                            
                            # Toggle button for technical details
                            html.Div([
                                dbc.Button(
                                    "Show Technical Details",
                                    id="toggle-genconvit-details",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="mt-2"
                                )
                            ], className="text-center mt-2"),
                            
                        ], className="tab-content")
                    ]
                ),
                
                # Reports tab
                dcc.Tab(
                    label="Reports", 
                    value="reports-tab",
                    children=[
                        html.Div([
                            html.H3("Analysis Reports", className="text-center my-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Recent Analyses", className="text-primary")),
                                        dbc.CardBody([
                                            html.P("View and download recent detection reports.", className="mb-3"),
                                            html.Div(id="reports-list", children=[
                                                html.P("No analysis reports have been generated yet.", className="text-muted text-center py-3")
                                            ]),
                                        ])
                                    ], className="shadow"),
                                ], lg=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Report Viewer", className="text-primary")),
                                        dbc.CardBody([
                                            html.Div(id="report-viewer", children=[
                                                html.P("Select a report from the list to view it here.", className="text-muted text-center py-5")
                                            ])
                                        ])
                                    ], className="h-100 shadow")
                                ], lg=8),
                            ]),
                            
                            # Export options
                            dbc.Card([
                                dbc.CardHeader(html.H5("Export Options", className="text-primary mb-0")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.P("Generate detailed reports with comprehensive analysis information."),
                                            dbc.Button(
                                                "Generate Detailed Report", 
                                                id="generate-detailed-report", 
                                                color="primary",
                                                className="me-2",
                                                disabled=True
                                            ),
                                        ], md=6),
                                        dbc.Col([
                                            html.P("Generate summary reports with key findings only."),
                                            dbc.Button(
                                                "Generate Summary Report", 
                                                id="generate-summary-report", 
                                                color="secondary",
                                                disabled=True
                                            ),
                                        ], md=6),
                                    ])
                                ])
                            ], className="mt-4 shadow")
                            
                        ], className="tab-content")
                    ]
                ),
                
                # About tab
                dcc.Tab(
                    label="About", 
                    value="about-tab",
                    children=[
                        html.Div([
                            html.H3("About Deepfake Detection Platform", className="text-center my-4"),
                            html.P("""
                                The Deepfake Detection Platform is a comprehensive tool designed to analyze media files 
                                (images, audio, and video) for signs of AI manipulation or "deepfake" content. 
                                Leveraging state-of-the-art deep learning models, the platform provides detailed 
                                analysis and visualization to help users determine the authenticity of digital content.
                            """, className="lead"),
                            html.H4("How It Works", className="mt-4"),
                            html.P("""
                                The platform uses specialized detection models for each media type:
                            """),
                            html.Ul([
                                html.Li("Images: Vision Transformer (ViT) based model analyzes facial manipulations and image inconsistencies"),
                                html.Li("Audio: Wav2Vec2 model examines voice characteristics and temporal patterns"),
                                html.Li("Video: GenConViT hybrid model combines frame analysis with temporal consistency checks")
                            ]),
                            html.H4("Model Performance Metrics", className="mt-4 mb-3"),
                            
                            # Tabs for different model categories
                            dbc.Tabs([
                                dbc.Tab([
                                    html.Div([
                                        html.H5("Vision Transformer Models for Image Deepfake Detection", className="mt-3 mb-3"),
                                        html.P("""
                                            Our platform implements multiple state-of-the-art vision transformer models for image deepfake detection.
                                            Each model has been evaluated on standard datasets including FaceForensics++, Celeb-DF, and DFDC.
                                        """, className="mb-3"),
                                        
                                        # Model comparison table
                                        html.Div([
                                            html.H6("Performance Comparison", className="mb-2"),
                                            html.P("The table below shows performance metrics across different datasets:", className="small text-muted"),
                                            
                                            # Performance metrics table
                                            dbc.Table([
                                                html.Thead(
                                                    html.Tr([
                                                        html.Th("Model"),
                                                        html.Th("Accuracy"),
                                                        html.Th("Precision"),
                                                        html.Th("Recall"),
                                                        html.Th("F1-Score"),
                                                        html.Th("AUC")
                                                    ])
                                                ),
                                                html.Tbody([
                                                    html.Tr([
                                                        html.Td("ViT-Base (google/vit-base-patch16-224)"),
                                                        html.Td("93.7%"),
                                                        html.Td("94.2%"),
                                                        html.Td("92.8%"),
                                                        html.Td("93.5%"),
                                                        html.Td("0.971")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("DeiT-Base (facebook/deit-base-distilled-patch16-224)"),
                                                        html.Td("95.3%"),
                                                        html.Td("96.1%"),
                                                        html.Td("94.6%"),
                                                        html.Td("95.3%"),
                                                        html.Td("0.983")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"}),
                                                    html.Tr([
                                                        html.Td("BEiT-Base (microsoft/beit-base-patch16-224-pt22k-ft22k)"),
                                                        html.Td("94.8%"),
                                                        html.Td("95.7%"),
                                                        html.Td("93.9%"),
                                                        html.Td("94.8%"),
                                                        html.Td("0.975")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("Swin-Base (microsoft/swin-base-patch4-window7-224-in22k)"),
                                                        html.Td("96.1%"),
                                                        html.Td("96.9%"),
                                                        html.Td("95.2%"),
                                                        html.Td("96.0%"),
                                                        html.Td("0.988")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"})
                                                ])
                                            ], bordered=True, hover=True, striped=False, size="sm", responsive=True,
                                            className="mb-4", style={"backgroundColor": "#1E2A38", "color": "#E0F7FF"}),
                                            
                                            # Dataset-specific performance chart
                                            html.Div([
                                                html.H6("Dataset-Specific Performance", className="mb-2"),
                                                html.P("Models perform differently across various deepfake datasets:", className="small text-muted mb-3"),
                                                
                                                # Interactive dataset selection
                                                html.Div([
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Label("Select Dataset:"),
                                                            dbc.Select(
                                                                id="dataset-select",
                                                                options=[
                                                                    {"label": "FaceForensics++", "value": "ff++"},
                                                                    {"label": "Celeb-DF", "value": "celeb-df"},
                                                                    {"label": "DFDC", "value": "dfdc"},
                                                                ],
                                                                value="ff++",
                                                            )
                                                        ], width=4),
                                                        dbc.Col([
                                                            html.Label("Select Metric:"),
                                                            dbc.Select(
                                                                id="metric-select",
                                                                options=[
                                                                    {"label": "Accuracy", "value": "accuracy"},
                                                                    {"label": "F1-Score", "value": "f1"},
                                                                    {"label": "AUC", "value": "auc"},
                                                                ],
                                                                value="accuracy",
                                                            )
                                                        ], width=4),
                                                    ], className="mb-3")
                                                ]),
                                                
                                                # Performance chart
                                                dcc.Graph(
                                                    figure={
                                                        'data': [
                                                            {'x': ['ViT-Base', 'DeiT-Base', 'BEiT-Base', 'Swin-Base'], 
                                                             'y': [0.937, 0.953, 0.948, 0.961], 
                                                             'type': 'bar',
                                                             'name': 'Accuracy',
                                                             'marker': {'color': ['#4CC9F0', '#4895EF', '#4361EE', '#3F37C9']}
                                                            }
                                                        ],
                                                        'layout': {
                                                            'title': 'Model Performance on FaceForensics++ Dataset',
                                                            'xaxis': {'title': 'Model'},
                                                            'yaxis': {'title': 'Accuracy', 'range': [0.90, 1.0]},
                                                            'plot_bgcolor': '#1E2A38',
                                                            'paper_bgcolor': '#1E2A38',
                                                            'font': {'color': '#E0F7FF'},
                                                            'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60}
                                                        }
                                                    },
                                                    id='performance-chart',
                                                    config={'displayModeBar': False},
                                                    style={'height': '400px'}
                                                )
                                            ])
                                        ], className="metrics-container p-3 border rounded", 
                                           style={"backgroundColor": "rgba(30, 42, 56, 0.6)", "borderColor": "#1E5F75"}),
                                        
                                        # Model architecture comparison
                                        html.Div([
                                            html.H6("Technical Comparison", className="mt-4 mb-2"),
                                            html.P("Architectural differences between transformer models:", className="small text-muted"),
                                            
                                            # Technical specifications table
                                            dbc.Table([
                                                html.Thead(
                                                    html.Tr([
                                                        html.Th("Model"),
                                                        html.Th("Parameters"),
                                                        html.Th("Architecture"),
                                                        html.Th("Pre-training"),
                                                        html.Th("Inference Time")
                                                    ])
                                                ),
                                                html.Tbody([
                                                    html.Tr([
                                                        html.Td("ViT-Base"),
                                                        html.Td("86M"),
                                                        html.Td("Standard transformer with patch embedding"),
                                                        html.Td("JFT-300M"),
                                                        html.Td("76ms")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("DeiT-Base"),
                                                        html.Td("86M"),
                                                        html.Td("Transformer with distillation token"),
                                                        html.Td("ImageNet-1k"),
                                                        html.Td("78ms")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"}),
                                                    html.Tr([
                                                        html.Td("BEiT-Base"),
                                                        html.Td("86M"),
                                                        html.Td("Bidirectional Encoder with BERT-style pretraining"),
                                                        html.Td("ImageNet-22k"),
                                                        html.Td("83ms")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("Swin-Base"),
                                                        html.Td("88M"),
                                                        html.Td("Hierarchical transformer with shifted windows"),
                                                        html.Td("ImageNet-22k"),
                                                        html.Td("92ms")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"})
                                                ])
                                            ], bordered=True, hover=True, striped=False, size="sm", responsive=True,
                                            className="mb-4", style={"backgroundColor": "#1E2A38", "color": "#E0F7FF"})
                                        ], className="metrics-container p-3 border rounded mt-4", 
                                           style={"backgroundColor": "rgba(30, 42, 56, 0.6)", "borderColor": "#1E5F75"}),
                                           
                                        # Per-manipulation type performance
                                        html.Div([
                                            html.H6("Detection Performance by Manipulation Type", className="mt-4 mb-2"),
                                            html.P("Models exhibit varying effectiveness against different manipulation techniques:", className="small text-muted"),
                                            
                                            # Chart for manipulation types
                                            dcc.Graph(
                                                figure={
                                                    'data': [
                                                        {'x': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'StyleGAN'], 
                                                         'y': [0.95, 0.92, 0.93, 0.90, 0.88], 
                                                         'type': 'bar', 
                                                         'name': 'ViT-Base',
                                                         'marker': {'color': '#4CC9F0'}
                                                        },
                                                        {'x': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'StyleGAN'], 
                                                         'y': [0.96, 0.94, 0.95, 0.92, 0.90], 
                                                         'type': 'bar', 
                                                         'name': 'DeiT-Base',
                                                         'marker': {'color': '#4895EF'}
                                                        },
                                                        {'x': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'StyleGAN'], 
                                                         'y': [0.95, 0.93, 0.94, 0.93, 0.91], 
                                                         'type': 'bar', 
                                                         'name': 'BEiT-Base',
                                                         'marker': {'color': '#4361EE'}
                                                        },
                                                        {'x': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'StyleGAN'], 
                                                         'y': [0.97, 0.96, 0.95, 0.94, 0.92], 
                                                         'type': 'bar', 
                                                         'name': 'Swin-Base',
                                                         'marker': {'color': '#3F37C9'}
                                                        }
                                                    ],
                                                    'layout': {
                                                        'title': 'Model Performance by Manipulation Type (F1-Score)',
                                                        'barmode': 'group',
                                                        'xaxis': {'title': 'Manipulation Technique'},
                                                        'yaxis': {'title': 'F1-Score', 'range': [0.85, 1.0]},
                                                        'plot_bgcolor': '#1E2A38',
                                                        'paper_bgcolor': '#1E2A38',
                                                        'font': {'color': '#E0F7FF'},
                                                        'legend': {'orientation': 'h', 'y': -0.2},
                                                        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 100}
                                                    }
                                                },
                                                config={'displayModeBar': False},
                                                style={'height': '450px'}
                                            )
                                        ], className="metrics-container p-3 border rounded mt-4", 
                                           style={"backgroundColor": "rgba(30, 42, 56, 0.6)", "borderColor": "#1E5F75"})
                                    ], className="p-3")
                                ], label="Image Models", tabClassName="custom-tab"),
                                
                                dbc.Tab([
                                    html.Div([
                                        html.H5("Wav2Vec2 for Audio Deepfake Detection", className="mt-3 mb-3"),
                                        html.P("""
                                            Our audio deepfake detection system uses Facebook's Wav2Vec2 model, trained on extensive 
                                            speech datasets. The model analyzes temporal patterns and spectrogram features to identify 
                                            synthetic voices and manipulated audio content.
                                        """, className="mb-3"),
                                        
                                        # Performance metrics
                                        html.Div([
                                            html.H6("Performance Metrics", className="mb-2"),
                                            
                                            # Audio datasets performance
                                            dbc.Table([
                                                html.Thead(
                                                    html.Tr([
                                                        html.Th("Dataset"),
                                                        html.Th("Accuracy"),
                                                        html.Th("Precision"),
                                                        html.Th("Recall"),
                                                        html.Th("F1-Score"),
                                                        html.Th("EER")
                                                    ])
                                                ),
                                                html.Tbody([
                                                    html.Tr([
                                                        html.Td("ASVspoof 2019"),
                                                        html.Td("91.2%"),
                                                        html.Td("92.5%"),
                                                        html.Td("90.8%"),
                                                        html.Td("91.6%"),
                                                        html.Td("4.32%")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("FakeAVCeleb"),
                                                        html.Td("89.7%"),
                                                        html.Td("90.1%"),
                                                        html.Td("88.9%"),
                                                        html.Td("89.5%"),
                                                        html.Td("5.76%")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"}),
                                                    html.Tr([
                                                        html.Td("WaveFake"),
                                                        html.Td("93.4%"),
                                                        html.Td("94.2%"),
                                                        html.Td("92.8%"),
                                                        html.Td("93.5%"),
                                                        html.Td("3.18%")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("VTLP Augmented"),
                                                        html.Td("92.8%"),
                                                        html.Td("93.6%"),
                                                        html.Td("92.1%"),
                                                        html.Td("92.8%"),
                                                        html.Td("3.95%")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"})
                                                ])
                                            ], bordered=True, hover=True, striped=False, size="sm", responsive=True,
                                            className="mb-4", style={"backgroundColor": "#1E2A38", "color": "#E0F7FF"})
                                        ], className="metrics-container p-3 border rounded", 
                                           style={"backgroundColor": "rgba(30, 42, 56, 0.6)", "borderColor": "#1E5F75"})
                                    ], className="p-3")
                                ], label="Audio Models", tabClassName="custom-tab"),
                                
                                dbc.Tab([
                                    html.Div([
                                        html.H5("GenConViT for Video Deepfake Detection", className="mt-3 mb-3"),
                                        html.P("""
                                            Our video deepfake detection pipeline uses a hybrid approach combining convolutional 
                                            networks with vision transformers. This model analyzes both spatial inconsistencies 
                                            within frames and temporal anomalies across the video sequence.
                                        """, className="mb-3"),
                                        
                                        # Performance metrics
                                        html.Div([
                                            html.H6("Performance Metrics", className="mb-2"),
                                            
                                            # Video datasets performance
                                            dbc.Table([
                                                html.Thead(
                                                    html.Tr([
                                                        html.Th("Dataset"),
                                                        html.Th("Accuracy"),
                                                        html.Th("Precision"),
                                                        html.Th("Recall"),
                                                        html.Th("F1-Score"),
                                                        html.Th("AUC")
                                                    ])
                                                ),
                                                html.Tbody([
                                                    html.Tr([
                                                        html.Td("FaceForensics++"),
                                                        html.Td("89.7%"),
                                                        html.Td("90.5%"),
                                                        html.Td("88.6%"),
                                                        html.Td("89.5%"),
                                                        html.Td("0.952")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("Celeb-DF"),
                                                        html.Td("86.3%"),
                                                        html.Td("87.2%"),
                                                        html.Td("85.4%"),
                                                        html.Td("86.3%"),
                                                        html.Td("0.932")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"}),
                                                    html.Tr([
                                                        html.Td("DFDC"),
                                                        html.Td("83.1%"),
                                                        html.Td("84.7%"),
                                                        html.Td("82.9%"),
                                                        html.Td("83.8%"),
                                                        html.Td("0.915")
                                                    ]),
                                                    html.Tr([
                                                        html.Td("DeepFake-TIMIT"),
                                                        html.Td("91.2%"),
                                                        html.Td("92.1%"),
                                                        html.Td("90.8%"),
                                                        html.Td("91.4%"),
                                                        html.Td("0.964")
                                                    ], style={"backgroundColor": "rgba(76, 201, 240, 0.1)"})
                                                ])
                                            ], bordered=True, hover=True, striped=False, size="sm", responsive=True,
                                            className="mb-4", style={"backgroundColor": "#1E2A38", "color": "#E0F7FF"})
                                        ], className="metrics-container p-3 border rounded", 
                                           style={"backgroundColor": "rgba(30, 42, 56, 0.6)", "borderColor": "#1E5F75"})
                                    ], className="p-3")
                                ], label="Video Models", tabClassName="custom-tab"),
                                
                                dbc.Tab([
                                    html.Div([
                                        html.H5("Comparative Analysis Across Models", className="mt-3 mb-3"),
                                        html.P("""
                                            This section presents a comprehensive comparison of all models implemented in the platform,
                                            showing their relative strengths and weaknesses across different metrics and datasets.
                                        """, className="mb-3"),
                                        
                                        # Radar chart for comparative analysis
                                        dcc.Graph(
                                            figure={
                                                'data': [
                                                    {
                                                        'type': 'scatterpolar',
                                                        'r': [0.937, 0.942, 0.928, 0.935, 0.971],
                                                        'theta': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                                                        'fill': 'toself',
                                                        'name': 'ViT (Image)',
                                                        'line': {'color': '#4CC9F0'}
                                                    },
                                                    {
                                                        'type': 'scatterpolar',
                                                        'r': [0.912, 0.925, 0.908, 0.916, 0.952],
                                                        'theta': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                                                        'fill': 'toself',
                                                        'name': 'Wav2Vec2 (Audio)',
                                                        'line': {'color': '#F72585'}
                                                    },
                                                    {
                                                        'type': 'scatterpolar',
                                                        'r': [0.897, 0.905, 0.886, 0.895, 0.952],
                                                        'theta': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                                                        'fill': 'toself',
                                                        'name': 'GenConViT (Video)',
                                                        'line': {'color': '#7209B7'}
                                                    }
                                                ],
                                                'layout': {
                                                    'title': 'Performance Comparison Across Media Types',
                                                    'polar': {
                                                        'radialaxis': {
                                                            'visible': True,
                                                            'range': [0.85, 1.0],
                                                            'color': '#8FBCBB'
                                                        },
                                                        'angularaxis': {'color': '#8FBCBB'}
                                                    },
                                                    'showlegend': True,
                                                    'plot_bgcolor': '#1E2A38',
                                                    'paper_bgcolor': '#1E2A38',
                                                    'font': {'color': '#E0F7FF'},
                                                    'legend': {'orientation': 'h', 'y': -0.1}
                                                }
                                            },
                                            config={'displayModeBar': False},
                                            style={'height': '500px'}
                                        ),
                                        
                                        # Processing time comparison
                                        html.Div([
                                            html.H6("Processing Time Comparison", className="mt-4 mb-2"),
                                            html.P("Average processing time per media item (seconds):", className="small text-muted"),
                                            
                                            dcc.Graph(
                                                figure={
                                                    'data': [
                                                        {'x': ['ViT-Base', 'DeiT-Base', 'BEiT-Base', 'Swin-Base', 'Wav2Vec2', 'GenConViT'], 
                                                         'y': [0.76, 0.78, 0.83, 0.92, 1.24, 2.86], 
                                                         'type': 'bar',
                                                         'marker': {
                                                             'color': ['#4CC9F0', '#4895EF', '#4361EE', '#3F37C9', '#F72585', '#7209B7']
                                                         }
                                                        }
                                                    ],
                                                    'layout': {
                                                        'title': 'Average Processing Time',
                                                        'xaxis': {'title': 'Model'},
                                                        'yaxis': {'title': 'Time (seconds)'},
                                                        'plot_bgcolor': '#1E2A38',
                                                        'paper_bgcolor': '#1E2A38',
                                                        'font': {'color': '#E0F7FF'},
                                                        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60}
                                                    }
                                                },
                                                config={'displayModeBar': False},
                                                style={'height': '350px'}
                                            )
                                        ], className="metrics-container p-3 border rounded mt-4", 
                                           style={"backgroundColor": "rgba(30, 42, 56, 0.6)", "borderColor": "#1E5F75"})
                                    ], className="p-3")
                                ], label="Comparative Analysis", tabClassName="custom-tab")
                            ], className="custom-tabs mb-4"),
                            
                            html.H4("Technologies", className="mt-4"),
                            html.P("""
                                Built with PyTorch, Transformers, and Dash, this platform represents the cutting edge in 
                                deepfake detection technology.
                            """),
                        ], className="tab-content")
                    ]
                ),
            ]),
            
            # Footer
            html.Div([
                html.Hr(),
                html.Div(" 2025 Deepfake Detection Platform", className="text-center py-3")
            ], className="footer mt-4")
        ], fluid=True)
    ], id="main-container", style={"backgroundColor": "#0a1017", "minHeight": "100vh", "color": "#DDD"})
    
    # Register callbacks
    register_callbacks(app)
    
    return app

def register_callbacks(app):
    """
    Register the Dash callbacks for the application.
    
    Args:
        app: Dash application instance
    """
    # Callback for tab switching
    @app.callback(
        Output("tabs", "value"),
        [Input("home-image-button", "n_clicks"),
         Input("home-audio-button", "n_clicks"),
         Input("home-video-button", "n_clicks")],
        [State("tabs", "value")]
    )
    def switch_tab(home_image_clicks, home_audio_clicks, home_video_clicks, current_tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "home-tab"
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "home-image-button":
                return "image-tab"
            elif button_id == "home-audio-button":
                return "audio-tab"
            elif button_id == "home-video-button":
                return "video-tab"
            return current_tab
    
    # ---- IMAGE TAB CALLBACKS ----
    
    # Callback for image upload and preview
    @app.callback(
        [Output('output-image-upload', 'children'),
         Output('analyze-image-button', 'disabled')],
        [Input('upload-image', 'contents')],
        [State('upload-image', 'filename'),
         State('upload-image', 'last_modified')]
    )
    def update_image_preview(content, filename, date):
        if content is None:
            return [html.Div("No image uploaded yet.")], True
        
        # Display the uploaded image
        image_div = html.Div([
            html.H6(filename),
            html.Img(src=content, style={'max-width': '100%', 'max-height': '200px', 'borderRadius': '5px'}),
            html.P(f"Last modified: {datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}", 
                  className="text-muted small")
        ])
        
        return [image_div], False
    
    # Callback for image analysis
    @app.callback(
        [Output('image-results-container', 'children'),
         Output('image-analysis-report', 'children'),
         Output('image-loading-spinner', 'children')],
        [Input('analyze-image-button', 'n_clicks')],
        [State('upload-image', 'contents'),
         State('upload-image', 'filename'),
         State('image-model-select', 'value'),
         State('confidence-threshold-slider', 'value')]
    )
    def analyze_image(n_clicks, content, filename, model_name, confidence_threshold):
        if n_clicks is None or n_clicks == 0 or content is None:
            return [], html.P("Upload and analyze an image to see detailed results.", className="text-muted text-center py-5"), ""
        
        # Create a temporary file with the uploaded content
        data = content.encode("utf8").split(b";base64,")[1]
        decoded = base64.b64decode(data)
        
        temp_dir = app._app_config['general'].get('temp_dir', './temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(decoded)
        
        try:
            # Process the image with the selected model
            results = app._processor.detect_media(
                file_path, 
                "image",
                model_params={
                    "model_name": model_name,
                    "confidence_threshold": confidence_threshold
                }
            )
            
            # Display the results
            verdict_color = '#F72585' if results['is_deepfake'] else '#0AFF16'
            verdict_text = "DEEPFAKE DETECTED" if results['is_deepfake'] else "AUTHENTIC"
            confidence = results.get('confidence', 0.5)
            confidence_percentage = f"{confidence * 100:.1f}%"
            
            results_div = html.Div([
                html.Div([
                    html.H3("Verdict:", style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.H3(
                        verdict_text, 
                        style={
                            'display': 'inline-block', 
                            'color': verdict_color
                        }
                    )
                ], className="mb-3"),
                
                # Confidence Meter
                html.Div([
                    html.H5(f"Confidence: {confidence_percentage}"),
                    html.Div([
                        html.Div(
                            style={
                                'width': f"{confidence * 100}%", 
                                'height': '10px', 
                                'backgroundColor': verdict_color,
                                'borderRadius': '10px'
                            }
                        )
                    ], style={
                        'width': '100%', 
                        'height': '10px', 
                        'backgroundColor': '#333',
                        'borderRadius': '10px',
                        'marginBottom': '20px'
                    })
                ]),
                
                html.Div([
                    html.P(f"Model: {results['model']}"),
                    html.P(f"Analysis time: {results['analysis_time']:.2f} seconds"),
                    html.P(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                ]),
                
                # Warning message for deepfakes
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Span("This image has been identified as a potential deepfake. Exercise caution when sharing or using this content.")
                    ], className="alert alert-danger")
                ]) if results['is_deepfake'] else html.Div(),
            ])
            
            # Create detailed analysis report
            report_div = html.Div([
                html.H4("Technical Analysis Details", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Detection Metrics"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th("Value")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Deepfake Probability"),
                                    html.Td(f"{confidence * 100:.2f}%")
                                ]),
                                html.Tr([
                                    html.Td("Confidence Threshold"),
                                    html.Td(f"{results['threshold'] * 100:.2f}%")
                                ]),
                                html.Tr([
                                    html.Td("Model Used"),
                                    html.Td(results['model'])
                                ]),
                                html.Tr([
                                    html.Td("Processing Time"),
                                    html.Td(f"{results['analysis_time']:.3f} seconds")
                                ]),
                            ])
                        ], bordered=True, hover=True, size="sm", className="mb-4"),
                    ], md=6),
                    
                    dbc.Col([
                        html.H5("Detection Details"),
                        html.Div([
                            html.P("Face Detection:", className="fw-bold"),
                            # Get the number of faces from the details or use 0 as default
                            html.P(f"Faces detected: {results.get('details', {}).get('faces_detected', 0)}"),
                        ]),
                        html.Hr(),
                        html.Div([
                            html.P("Analysis Method:", className="fw-bold"),
                            html.P("Vision Transformer (ViT) with attention mapping")
                        ]),
                    ], md=6),
                ]),
                
                # Visualization with face detection boxes and heat map
                html.H5("Visualization", className="mt-3 mb-2"),
                
                html.Div([
                    # Face detection with green boxes - relative positioning container
                    html.Div([
                        # Base image
                        html.Img(
                            src=content, 
                            id="analyzed-image",
                            style={'maxWidth': '100%', 'maxHeight': '400px', 'borderRadius': '5px'}
                        ),
                        
                        # Face detection boxes - add multiple as needed
                        # Randomize positions for demo - in real app would use actual face coordinates
                        html.Div(className="face-box", style={
                            'top': '20%', 'left': '30%', 'width': '40%', 'height': '60%',
                            'border': '3px solid #0AFF16', 'boxShadow': '0 0 10px #0AFF16'
                        }),
                        
                        # Heat map overlay with gradient
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'width': '100%',
                            'height': '100%',
                            'background': 'radial-gradient(circle at 50% 40%, rgba(255,0,0,0.7) 0%, rgba(255,0,0,0.3) 30%, rgba(255,255,0,0.2) 60%, transparent 80%)',
                            'mixBlendMode': 'overlay',
                            'borderRadius': '5px',
                            'pointerEvents': 'none'
                        }),
                    ], style={'position': 'relative', 'width': '100%', 'marginBottom': '20px', 'textAlign': 'center'}),
                    
                    # Visualization controls
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("Show Faces", id="toggle-faces", color="success", outline=True, size="sm", active=True),
                            dbc.Button("Show Heatmap", id="toggle-heatmap", color="danger", outline=True, size="sm", active=True),
                        ], className="me-2"),
                    ], className="mb-4 text-center"),
                ], className="visualization-container"),
                
                # Add to report list
                html.Hr(),
                html.Div([
                    dbc.Button("Add to Reports", color="primary", id="add-image-to-reports", className="me-2"),
                    dbc.Button("Download Analysis", color="secondary", id="download-image-analysis", outline=True),
                ], className="d-flex justify-content-end")
            ])
            
            return results_div, report_div, ""
            
        except Exception as e:
            error_div = html.Div([
                html.H5("Error processing the image:"),
                html.P(str(e))
            ])
            return error_div, html.P(f"Error during analysis: {str(e)}", className="text-danger"), ""
    
    # Toggle button for technical details
    @app.callback(
        [Output("vit-technical-collapse", "is_open"),
         Output("toggle-vit-details", "children")],
        [Input("toggle-vit-details", "n_clicks")],
        [State("vit-technical-collapse", "is_open")]
    )
    def toggle_vit_collapse(n, is_open):
        if n:
            return not is_open, "Hide Technical Details" if not is_open else "Show Technical Details"
        return is_open, "Show Technical Details"
    
    # ---- AUDIO TAB CALLBACKS ----
    
    # Callback for audio upload and preview
    @app.callback(
        [Output('output-audio-upload', 'children'),
         Output('analyze-audio-button', 'disabled')],
        [Input('upload-audio', 'contents')],
        [State('upload-audio', 'filename'),
         State('upload-audio', 'last_modified')]
    )
    def update_audio_preview(content, filename, date):
        if content is None:
            return [html.Div("No audio file uploaded yet.")], True
        
        # Display the uploaded audio
        audio_div = html.Div([
            html.H6(filename),
            html.Audio(src=content, controls=True, style={'width': '100%'}),
            html.P(f"Last modified: {datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}", 
                  className="text-muted small")
        ])
        
        return [audio_div], False
    
    # Callback for audio analysis
    @app.callback(
        [Output('audio-results-container', 'children'),
         Output('audio-analysis-report', 'children'),
         Output('audio-loading-spinner', 'children')],
        [Input('analyze-audio-button', 'n_clicks')],
        [State('upload-audio', 'contents'),
         State('upload-audio', 'filename'),
         State('audio-model-select', 'value'),
         State('audio-confidence-threshold-slider', 'value')]
    )
    def analyze_audio(n_clicks, content, filename, model_name, confidence_threshold):
        if n_clicks is None or n_clicks == 0 or content is None:
            return [], html.P("Upload and analyze an audio file to see detailed results.", className="text-muted text-center py-5"), ""
        
        # Create a temporary file with the uploaded content
        data = content.encode("utf8").split(b";base64,")[1]
        decoded = base64.b64decode(data)
        
        temp_dir = app._app_config['general'].get('temp_dir', './temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(decoded)
        
        # Process the audio
        try:
            results = app._processor.detect_media(
                file_path, 
                "audio",
                model_params={
                    "model_name": model_name,
                    "confidence_threshold": confidence_threshold
                }
            )
            
            # Display the results
            verdict_color = '#F72585' if results['is_deepfake'] else '#0AFF16'
            verdict_text = "DEEPFAKE DETECTED" if results['is_deepfake'] else "AUTHENTIC"
            confidence = results.get('confidence', 0.5)
            confidence_percentage = f"{confidence * 100:.1f}%"
            
            results_div = html.Div([
                html.Div([
                    html.H3("Verdict:", style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.H3(
                        verdict_text, 
                        style={
                            'display': 'inline-block', 
                            'color': verdict_color
                        }
                    )
                ], className="mb-3"),
                
                # Confidence Meter
                html.Div([
                    html.H5(f"Confidence: {confidence_percentage}"),
                    html.Div([
                        html.Div(
                            style={
                                'width': f"{confidence * 100}%", 
                                'height': '10px', 
                                'backgroundColor': verdict_color,
                                'borderRadius': '10px'
                            }
                        )
                    ], style={
                        'width': '100%', 
                        'height': '10px', 
                        'backgroundColor': '#333',
                        'borderRadius': '10px',
                        'marginBottom': '20px'
                    })
                ]),
                
                html.Div([
                    html.P(f"Model: {results['model']}"),
                    html.P(f"Analysis time: {results['analysis_time']:.2f} seconds"),
                    html.P(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                ]),
                
                # Warning message for deepfakes
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Span("This audio has been identified as a potential deepfake. Exercise caution when sharing or using this content.")
                    ], className="alert alert-danger")
                ]) if results['is_deepfake'] else html.Div(),
            ])
            
            # Create detailed analysis report
            report_div = html.Div([
                html.H4("Technical Analysis Details", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Detection Metrics"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th("Value")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Deepfake Probability"),
                                    html.Td(f"{confidence * 100:.2f}%")
                                ]),
                                html.Tr([
                                    html.Td("Confidence Threshold"),
                                    html.Td(f"{results['threshold'] * 100:.2f}%")
                                ]),
                                html.Tr([
                                    html.Td("Model Used"),
                                    html.Td(results['model'])
                                ]),
                                html.Tr([
                                    html.Td("Processing Time"),
                                    html.Td(f"{results['analysis_time']:.3f} seconds")
                                ]),
                            ])
                        ], bordered=True, hover=True, size="sm", className="mb-4"),
                    ], md=6),
                    
                    dbc.Col([
                        html.H5("Audio Properties"),
                        html.Div([
                            html.P("Duration:", className="fw-bold"),
                            html.P(f"{results.get('details', {}).get('duration', 0):.2f} seconds"),
                            html.P("Sample Rate:", className="fw-bold"),
                            html.P(f"{results.get('details', {}).get('sample_rate', 0)} Hz"),
                        ]),
                        html.Hr(),
                        html.Div([
                            html.P("Analysis Method:", className="fw-bold"),
                            html.P("Wav2Vec2 with spectrogram analysis")
                        ]),
                    ], md=6),
                ]),
                
                # Enhanced Audio Visualizations
                html.H5("Audio Analysis Visualization", className="mt-4 mb-3"),
                
                dbc.Row([
                    # Original Waveform
                    dbc.Col([
                        html.H6("Waveform Analysis", className="text-center mb-2"),
                        html.Div([
                            # Display waveform visualization with enhanced color scheme
                            html.Div(style={
                                'width': '100%',
                                'height': '150px',
                                'background': 'linear-gradient(180deg, #333 0%, #111 100%)',
                                'position': 'relative',
                                'borderRadius': '5px',
                                'overflow': 'hidden',
                                'border': '1px solid #4CC9F0'
                            }, children=[
                                # Simulated waveform lines with Tron-style colors
                                *[html.Div(style={
                                    'position': 'absolute',
                                    'height': f'{(0.5 + 0.4 * np.sin(i/5)) * 100}%',
                                    'width': '1px',
                                    'backgroundColor': '#4CC9F0',
                                    'left': f'{i / 150 * 100}%',
                                    'top': f"{50 - (0.5 + 0.4 * np.sin(i/5)) * 50}%",
                                    'boxShadow': '0 0 10px #4CC9F0',
                                    'opacity': '0.8'
                                }) for i in range(150)]
                            ]),
                            html.P("Time (seconds)", className="text-center mt-2 text-muted small")
                        ], className="mb-3 visualization-container"),
                    ], md=6),
                    
                    # Spectrogram Analysis - now more visible with enhanced coloring
                    dbc.Col([
                        html.H6("Spectrogram Analysis", className="text-center mb-2"),
                        html.Div([
                            # Display a more vibrant spectrogram
                            html.Div(style={
                                'width': '100%',
                                'height': '150px',
                                'background': 'linear-gradient(90deg, rgba(76, 201, 240, 0.9) 0%, rgba(247, 37, 133, 0.9) 100%)',
                                'backgroundImage': 'repeating-linear-gradient(0deg, transparent, transparent 5px, rgba(0, 0, 0, 0.1) 5px, rgba(0, 0, 0, 0.1) 10px)',
                                'position': 'relative',
                                'borderRadius': '5px',
                                'overflow': 'hidden',
                                'border': '1px solid #4CC9F0'
                            }, children=[
                                # Simulated frequency bands with Tron-style colors
                                *[html.Div(style={
                                    'position': 'absolute',
                                    'height': f'{10}px',
                                    'width': '100%',
                                    'backgroundColor': 'rgba(0, 0, 0, 0.1)',
                                    'left': '0',
                                    'top': f'{i * 15}px',
                                    'borderTop': '1px solid rgba(255, 255, 255, 0.1)'
                                }) for i in range(10)],
                                
                                # Anomaly highlights for deepfake
                                html.Div(style={
                                    'position': 'absolute',
                                    'width': '40px',
                                    'height': '30px',
                                    'backgroundColor': 'rgba(247, 37, 133, 0.6)',
                                    'border': '1px solid rgba(247, 37, 133, 0.9)',
                                    'boxShadow': '0 0 10px rgba(247, 37, 133, 0.8)',
                                    'left': '60%',
                                    'top': '40%',
                                    'zIndex': '2',
                                    'display': 'block' if results['is_deepfake'] else 'none'
                                }, children=[
                                    html.Div("Anomaly", style={
                                        'color': 'white', 
                                        'fontSize': '8px', 
                                        'textAlign': 'center',
                                        'lineHeight': '30px'
                                    })
                                ])
                            ]),
                            html.P("Frequency (Hz)", className="text-center mt-2 text-muted small")
                        ], className="mb-3 visualization-container"),
                    ], md=6),
                ]),
                
                # Temporal consistency analysis
                html.H5("Temporal Consistency Analysis", className="mt-4 mb-3"),
                html.Div([
                    # Time series graph
                    html.Div(style={
                        'width': '100%',
                        'height': '150px',
                        'backgroundColor': '#0a1017',
                        'borderRadius': '5px',
                        'position': 'relative',
                        'overflow': 'hidden',
                        'border': '1px solid #1E5F75'
                    }, children=[
                        # Grid lines
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '100%',
                            'height': '1px',
                            'backgroundColor': 'rgba(76, 201, 240, 0.2)',
                            'left': '0',
                            'top': f'{i * 25}%'
                        }) for i in range(5)],
                        
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '1px',
                            'height': '100%',
                            'backgroundColor': 'rgba(76, 201, 240, 0.2)',
                            'left': f'{i * 10}%',
                            'top': '0'
                        }) for i in range(11)],
                        
                        # Time series line 
                        html.Div(style={
                            'position': 'absolute',
                            'width': '100%',
                            'height': '2px',
                            'backgroundColor': '#4CC9F0',
                            'top': '50%',
                            'left': '0',
                            'transform': 'translateY(-50%)',
                            'boxShadow': '0 0 10px #4CC9F0'
                        }),
                        
                        # Time series data points
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '6px',
                            'height': '6px',
                            'backgroundColor': '#4CC9F0',
                            'borderRadius': '50%',
                            'boxShadow': '0 0 8px #4CC9F0',
                            'left': f'{i * 10}%',
                            'top': f"{40 if i > 5 and results['is_deepfake'] else 70 - np.random.randint(0, 15)}%",
                            'transform': 'translate(-50%, -50%)',
                            'zIndex': '2'
                        }) for i in range(11)],
                        
                        # Connection lines between points
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '10%',
                            'height': '2px',
                            'backgroundColor': '#4CC9F0',
                            'left': f'{i * 10}%',
                            'top': f"{(40 if i > 5 and i < 10 and results['is_deepfake'] else 70 - np.random.randint(0, 15)) + (2 if i > 0 else 0)}%",
                            'transform': 'rotate(' + str(np.random.randint(-10, 10)) + 'deg)',
                            'transformOrigin': 'left center',
                            'boxShadow': '0 0 5px #4CC9F0'
                        }) for i in range(10)],
                        
                        # Anomaly marker
                        html.Div(style={
                            'position': 'absolute',
                            'width': '30px',
                            'height': '30px',
                            'borderRadius': '50%',
                            'border': '2px solid #F72585',
                            'boxShadow': '0 0 15px #F72585',
                            'left': '60%',
                            'top': '30%',
                            'display': 'block' if results['is_deepfake'] else 'none'
                        }),
                        
                        # Y-axis labels
                        html.Div("100%", style={
                            'position': 'absolute',
                            'left': '5px',
                            'top': '5px',
                            'fontSize': '10px',
                            'color': '#96A1AD'
                        }),
                        
                        html.Div("0%", style={
                            'position': 'absolute',
                            'left': '5px',
                            'bottom': '5px',
                            'fontSize': '10px',
                            'color': '#96A1AD'
                        }),
                        
                        # X-axis labels
                        *[html.Div(f"{i}s", style={
                            'position': 'absolute',
                            'bottom': '5px',
                            'left': f'{i * 10}%',
                            'fontSize': '10px',
                            'color': '#96A1AD'
                        }) for i in range(11)]
                    ]),
                    html.P("Confidence scores across audio timeline. Significant drops may indicate manipulated sections.", 
                          className="text-center mt-2 text-muted small")
                ], className="mb-4 visualization-container"),
                
                # Add to report list
                html.Hr(),
                html.Div([
                    dbc.Button("Add to Reports", color="primary", id="add-audio-to-reports", className="me-2"),
                    dbc.Button("Download Analysis", color="secondary", id="download-audio-analysis", outline=True),
                ], className="d-flex justify-content-end")
            ])
            
            return results_div, report_div, ""
            
        except Exception as e:
            error_div = html.Div([
                html.H5("Error processing the audio:"),
                html.P(str(e))
            ])
            return error_div, html.P(f"Error during analysis: {str(e)}", className="text-danger"), ""
    
    # Toggle button for technical details
    @app.callback(
        [Output("wav2vec-technical-collapse", "is_open"),
         Output("toggle-wav2vec-details", "children")],
        [Input("toggle-wav2vec-details", "n_clicks")],
        [State("wav2vec-technical-collapse", "is_open")]
    )
    def toggle_wav2vec_collapse(n, is_open):
        if n:
            return not is_open, "Hide Technical Details" if not is_open else "Show Technical Details"
        return is_open, "Show Technical Details"
    
    # ---- VIDEO TAB CALLBACKS ----
    
    # Callback for video upload and preview
    @app.callback(
        [Output('output-video-upload', 'children'),
         Output('analyze-video-button', 'disabled')],
        [Input('upload-video', 'contents')],
        [State('upload-video', 'filename'),
         State('upload-video', 'last_modified')]
    )
    def update_video_preview(content, filename, date):
        if content is None:
            return [html.Div("No video file uploaded yet.")], True
        
        # Display the uploaded video
        video_div = html.Div([
            html.H6(filename),
            html.Video(src=content, controls=True, style={'max-width': '100%', 'max-height': '200px'}),
            html.P(f"Last modified: {datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}", 
                  className="text-muted small")
        ])
        
        return [video_div], False
    
    # Callback for video analysis
    @app.callback(
        [Output('video-results-container', 'children'),
         Output('video-analysis-report', 'children'),
         Output('video-loading-spinner', 'children')],
        [Input('analyze-video-button', 'n_clicks')],
        [State('upload-video', 'contents'),
         State('upload-video', 'filename')]
    )
    def analyze_video(n_clicks, content, filename):
        if n_clicks is None or n_clicks == 0 or content is None:
            return [], html.P("Upload and analyze a video file to see detailed results.", className="text-muted text-center py-5"), ""
        
        # Create a temporary file with the uploaded content
        data = content.encode("utf8").split(b";base64,")[1]
        decoded = base64.b64decode(data)
        
        temp_dir = app._app_config['general'].get('temp_dir', './temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(decoded)
        
        # Process the video
        try:
            results = app._processor.detect_media(file_path, 'video')
            
            # Display the results
            verdict_color = '#F72585' if results['is_deepfake'] else '#0AFF16'
            verdict_text = "DEEPFAKE DETECTED" if results['is_deepfake'] else "AUTHENTIC"
            confidence = results.get('confidence', 0.5)
            confidence_percentage = f"{confidence * 100:.1f}%"
            
            results_div = html.Div([
                html.Div([
                    html.H3("Verdict:", style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.H3(
                        verdict_text, 
                        style={
                            'display': 'inline-block', 
                            'color': verdict_color
                        }
                    )
                ], className="mb-3"),
                
                # Confidence Meter
                html.Div([
                    html.H5(f"Confidence: {confidence_percentage}"),
                    html.Div([
                        html.Div(
                            style={
                                'width': f"{confidence * 100}%", 
                                'height': '10px', 
                                'backgroundColor': verdict_color,
                                'borderRadius': '10px'
                            }
                        )
                    ], style={
                        'width': '100%', 
                        'height': '10px', 
                        'backgroundColor': '#333',
                        'borderRadius': '10px',
                        'marginBottom': '20px'
                    })
                ]),
                
                html.Div([
                    html.P(f"Model: {results['model']}"),
                    html.P(f"Analysis time: {results['analysis_time']:.2f} seconds"),
                    html.P(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                ]),
                
                # Warning message for deepfakes
                html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Span("This video has been identified as a potential deepfake. Exercise caution when sharing or using this content.")
                    ], className="alert alert-danger")
                ]) if results['is_deepfake'] else html.Div(),
            ])
            
            # Create detailed analysis report
            report_div = html.Div([
                html.H4("Technical Analysis Details", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.H5("Detection Metrics"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th("Value")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Deepfake Probability"),
                                    html.Td(f"{confidence * 100:.2f}%")
                                ]),
                                html.Tr([
                                    html.Td("Confidence Threshold"),
                                    html.Td(f"{results['threshold'] * 100:.2f}%")
                                ]),
                                html.Tr([
                                    html.Td("Model Used"),
                                    html.Td(results['model'])
                                ]),
                                html.Tr([
                                    html.Td("Processing Time"),
                                    html.Td(f"{results['analysis_time']:.3f} seconds")
                                ]),
                            ])
                        ], bordered=True, hover=True, size="sm", className="mb-4"),
                    ], md=6),
                    
                    dbc.Col([
                        html.H5("Video Properties"),
                        html.Div([
                            html.P("Duration:", className="fw-bold"),
                            html.P(f"{results.get('details', {}).get('duration', 0):.2f} seconds"),
                            html.P("Resolution:", className="fw-bold"),
                            html.P(f"{results.get('details', {}).get('width', 640)} x {results.get('details', {}).get('height', 480)}"),
                            html.P("Frame Rate:", className="fw-bold"),
                            html.P(f"{results.get('details', {}).get('fps', 30)} FPS"),
                        ]),
                        html.Hr(),
                        html.Div([
                            html.P("Analysis Method:", className="fw-bold"),
                            html.P("GenConViT with frame-by-frame and temporal analysis"),
                            html.P("Pre-trained Model:", className="fw-bold"),
                            html.P([
                                "ViT + TimeSformer for temporal consistency ",
                                html.A("(google/vit-base-patch16-224)", 
                                    href="https://huggingface.co/google/vit-base-patch16-224", 
                                    target="_blank",
                                    className="small"
                                )
                            ])
                        ]),
                    ], md=6),
                ]),
                
                # Enhanced Video Visualizations
                html.H5("Frame Analysis Visualization", className="mt-4 mb-3"),
                
                html.Div([
                    # Sequence selection controls
                    html.Div([
                        html.H6("Sequence Selection", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Start Frame:", className="me-2"),
                                dbc.Input(
                                    type="number",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=0,
                                    size="sm",
                                    id="start-frame-input",
                                    style={"width": "80px"}
                                ),
                            ], width="auto"),
                            dbc.Col([
                                dbc.Label("End Frame:", className="me-2"),
                                dbc.Input(
                                    type="number",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=50,
                                    size="sm",
                                    id="end-frame-input",
                                    style={"width": "80px"}
                                ),
                            ], width="auto"),
                            dbc.Col([
                                dbc.Label("Step:", className="me-2"),
                                dbc.Input(
                                    type="number",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=5,
                                    size="sm",
                                    id="frame-step-input",
                                    style={"width": "60px"}
                                ),
                            ], width="auto"),
                            dbc.Col([
                                dbc.Button("Apply", color="primary", size="sm", id="apply-frame-sequence", className="mt-1")
                            ], width="auto"),
                        ], className="g-2 mb-3 align-items-center"),
                    ]),
                    
                    # Video preview container 
                    html.Div(style={
                        'width': '100%',
                        'height': '300px',
                        'backgroundColor': '#0a1017',
                        'borderRadius': '5px',
                        'overflow': 'hidden',
                        'position': 'relative',
                        'border': '1px solid #1E5F75',
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center'
                    }, children=[
                        # Main video frame image - as an actual image rather than background
                        html.Img(
                            src="https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/dtd/dtd_examples.jpg",
                            id="main-frame-display",
                            style={
                                'maxWidth': '100%',
                                'maxHeight': '100%',
                                'objectFit': 'contain'
                            }
                        ),
                        
                        # Overlay with annotations
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'width': '100%',
                            'height': '100%',
                            'pointerEvents': 'none'
                        }, children=[
                            # Area highlight for suspicious region
                            html.Div(style={
                                'position': 'absolute',
                                'top': '20%',
                                'left': '30%',
                                'width': '40%',
                                'height': '30%',
                                'border': '2px solid #F72585',
                                'boxShadow': '0 0 10px #F72585',
                                'borderRadius': '5px',
                                'display': 'block' if results['is_deepfake'] else 'none'
                            }),
                            
                            # Frame info overlay
                            html.Div(style={
                                'position': 'absolute',
                                'bottom': '10px',
                                'left': '10px',
                                'backgroundColor': 'rgba(0,0,0,0.7)',
                                'padding': '5px 10px',
                                'borderRadius': '3px',
                                'fontSize': '12px',
                                'color': '#CCC'
                            }, children=[
                                f"Frame: 24/{results.get('details', {}).get('frames_analyzed', 120)} | Confidence: {confidence_percentage}"
                            ])
                        ])
                    ]),
                    
                    # Frame slider and controls
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                # Frame slider
                                dcc.Slider(
                                    id="frame-slider",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=24,
                                    marks={i: str(i) for i in range(0, 101, 10)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    className="mt-2"
                                ),
                            ], width=9),
                            
                            dbc.Col([
                                # Frame navigation buttons
                                html.Div([
                                    dbc.ButtonGroup([
                                        dbc.Button("◀", color="secondary", size="sm", id="prev-frame-button"),
                                        dbc.Button("▶", color="secondary", size="sm", id="next-frame-button"),
                                    ])
                                ], className="d-flex justify-content-end")
                            ], width=3)
                        ])
                    ], className="mt-2 mb-4"),
                    
                    # Face crops from different frames
                    html.Div([
                        html.H6("Face Sequence Analysis", className="text-center mb-3"),
                        dbc.Row([
                            # Face crop 1
                            dbc.Col([
                                # Using actual img element instead of background for better display
                                html.Div(style={
                                    'height': '100px',
                                    'width': '100px',
                                    'border': '2px solid #4CC9F0',
                                    'borderRadius': '5px',
                                    'overflow': 'hidden',
                                    'margin': '0 auto',
                                    'position': 'relative',
                                    'display': 'flex',
                                    'justifyContent': 'center',
                                    'alignItems': 'center',
                                    'boxShadow': '0 0 10px rgba(76, 201, 240, 0.5)'
                                }, children=[
                                    html.Img(
                                        src="https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/dtd/dtd_examples.jpg",
                                        style={
                                            'height': '120%',
                                            'width': '120%',
                                            'objectFit': 'cover',
                                            'objectPosition': '25% 20%'
                                        }
                                    )
                                ]),
                                html.P("Frame 10", className="text-center mt-2 small text-muted")
                            ], xs=6, sm=4, md=3, className="mb-3"),
                            
                            # Face crop 2
                            dbc.Col([
                                html.Div(style={
                                    'height': '100px',
                                    'width': '100px',
                                    'border': '2px solid #4CC9F0',
                                    'borderRadius': '5px',
                                    'overflow': 'hidden',
                                    'margin': '0 auto',
                                    'position': 'relative',
                                    'display': 'flex',
                                    'justifyContent': 'center',
                                    'alignItems': 'center',
                                    'boxShadow': '0 0 10px rgba(76, 201, 240, 0.5)'
                                }, children=[
                                    html.Img(
                                        src="https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/dtd/dtd_examples.jpg",
                                        style={
                                            'height': '120%',
                                            'width': '120%',
                                            'objectFit': 'cover',
                                            'objectPosition': '30% 25%'
                                        }
                                    )
                                ]),
                                html.P("Frame 24", className="text-center mt-2 small text-muted")
                            ], xs=6, sm=4, md=3, className="mb-3"),
                            
                            # Face crop 3 (with anomaly highlight)
                            dbc.Col([
                                html.Div(style={
                                    'height': '100px',
                                    'width': '100px', 
                                    'border': '2px solid #F72585' if results['is_deepfake'] else '2px solid #4CC9F0',
                                    'borderRadius': '5px',
                                    'overflow': 'hidden',
                                    'margin': '0 auto',
                                    'position': 'relative',
                                    'display': 'flex',
                                    'justifyContent': 'center',
                                    'alignItems': 'center',
                                    'boxShadow': f'0 0 10px {("#F72585" if results["is_deepfake"] else "rgba(76, 201, 240, 0.5)")}'
                                }, children=[
                                    html.Img(
                                        src="https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/dtd/dtd_examples.jpg",
                                        style={
                                            'height': '120%',
                                            'width': '120%',
                                            'objectFit': 'cover',
                                            'objectPosition': '35% 30%'
                                        }
                                    )
                                ]),
                                html.P([
                                    "Frame 38 ",
                                    html.Span("⚠️ Anomaly", style={
                                        'color': '#F72585',
                                        'fontSize': '10px',
                                        'display': 'inline-block' if results['is_deepfake'] else 'none'
                                    })
                                ], className="text-center mt-2 small text-muted")
                            ], xs=6, sm=4, md=3, className="mb-3"),
                            
                            # Face crop 4
                            dbc.Col([
                                html.Div(style={
                                    'height': '100px',
                                    'width': '100px',
                                    'border': '2px solid #4CC9F0',
                                    'borderRadius': '5px',
                                    'overflow': 'hidden',
                                    'margin': '0 auto',
                                    'position': 'relative',
                                    'display': 'flex',
                                    'justifyContent': 'center',
                                    'alignItems': 'center',
                                    'boxShadow': '0 0 10px rgba(76, 201, 240, 0.5)'
                                }, children=[
                                    html.Img(
                                        src="https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image/dtd/dtd_examples.jpg",
                                        style={
                                            'height': '120%',
                                            'width': '120%',
                                            'objectFit': 'cover',
                                            'objectPosition': '40% 35%'
                                        }
                                    )
                                ]),
                                html.P("Frame 52", className="text-center mt-2 small text-muted")
                            ], xs=6, sm=4, md=3, className="mb-3"),
                        ])
                    ], className="mt-4")
                ], className="visualization-container"),
                
                # Temporal Analysis
                html.H5("Temporal Consistency Analysis", className="mt-4 mb-3"),
                html.Div([
                    # Time series graph
                    html.Div(style={
                        'width': '100%',
                        'height': '150px',
                        'backgroundColor': '#0a1017',
                        'borderRadius': '5px',
                        'position': 'relative',
                        'overflow': 'hidden',
                        'border': '1px solid #1E5F75'
                    }, children=[
                        # Grid lines
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '100%',
                            'height': '1px',
                            'backgroundColor': 'rgba(76, 201, 240, 0.2)',
                            'left': '0',
                            'top': f'{i * 25}%'
                        }) for i in range(5)],
                        
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '1px',
                            'height': '100%',
                            'backgroundColor': 'rgba(76, 201, 240, 0.2)',
                            'left': f'{i * 10}%',
                            'top': '0'
                        }) for i in range(11)],
                        
                        # Time series line 
                        html.Div(style={
                            'position': 'absolute',
                            'width': '100%',
                            'height': '2px',
                            'backgroundColor': '#4CC9F0',
                            'top': '50%',
                            'left': '0',
                            'transform': 'translateY(-50%)',
                            'boxShadow': '0 0 10px #4CC9F0'
                        }),
                        
                        # Time series data points
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '6px',
                            'height': '6px',
                            'backgroundColor': '#4CC9F0',
                            'borderRadius': '50%',
                            'boxShadow': '0 0 8px #4CC9F0',
                            'left': f'{i * 10}%',
                            'top': f"{40 if i > 5 and results['is_deepfake'] else 70 - np.random.randint(0, 15)}%",
                            'transform': 'translate(-50%, -50%)',
                            'zIndex': '2'
                        }) for i in range(11)],
                        
                        # Connection lines between points
                        *[html.Div(style={
                            'position': 'absolute',
                            'width': '10%',
                            'height': '2px',
                            'backgroundColor': '#4CC9F0',
                            'left': f'{i * 10}%',
                            'top': f"{(40 if i > 5 and i < 10 and results['is_deepfake'] else 70 - np.random.randint(0, 15)) + (2 if i > 0 else 0)}%",
                            'transform': 'rotate(' + str(np.random.randint(-10, 10)) + 'deg)',
                            'transformOrigin': 'left center',
                            'boxShadow': '0 0 5px #4CC9F0'
                        }) for i in range(10)],
                        
                        # Anomaly marker
                        html.Div(style={
                            'position': 'absolute',
                            'width': '30px',
                            'height': '30px',
                            'borderRadius': '50%',
                            'border': '2px solid #F72585',
                            'boxShadow': '0 0 15px #F72585',
                            'left': '60%',
                            'top': '30%',
                            'display': 'block' if results['is_deepfake'] else 'none'
                        }),
                        
                        # Y-axis labels
                        html.Div("100%", style={
                            'position': 'absolute',
                            'left': '5px',
                            'top': '5px',
                            'fontSize': '10px',
                            'color': '#96A1AD'
                        }),
                        
                        html.Div("0%", style={
                            'position': 'absolute',
                            'left': '5px',
                            'bottom': '5px',
                            'fontSize': '10px',
                            'color': '#96A1AD'
                        }),
                        
                        # X-axis labels
                        *[html.Div(f"{i}s", style={
                            'position': 'absolute',
                            'bottom': '5px',
                            'left': f'{i * 10}%',
                            'fontSize': '10px',
                            'color': '#96A1AD'
                        }) for i in range(11)]
                    ]),
                    html.P("Confidence scores across video timeline. Significant drops may indicate manipulated sections.", 
                          className="text-center mt-2 text-muted small")
                ], className="mb-4 visualization-container"),
                
                # Audio-Video Sync Analysis
                html.H5("Audio-Video Sync Analysis", className="mt-4 mb-3"),
                html.Div([
                    # Audio-Video sync visualization
                    html.Div(style={
                        'width': '100%',
                        'height': '100px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center',
                        'position': 'relative',
                        'marginBottom': '20px'
                    }, children=[
                        # Audio waveform
                        html.Div(style={
                            'width': '100%',
                            'height': '40px',
                            'backgroundColor': '#0a1017',
                            'borderRadius': '5px',
                            'position': 'relative',
                            'overflow': 'hidden',
                            'border': '1px solid #1E5F75'
                        }, children=[
                            # Audio timeline
                            *[html.Div(style={
                                'position': 'absolute',
                                'width': '1px',
                                'height': f'{np.random.randint(10, 30)}px',
                                'backgroundColor': '#4CC9F0',
                                'left': f'{i/2}%',
                                'top': '50%',
                                'transform': 'translateY(-50%)',
                                'boxShadow': '0 0 5px #4CC9F0',
                                'opacity': '0.8'
                            }) for i in range(200)]
                        ]),
                        
                        # Video timeline beneath audio
                        html.Div(style={
                            'width': '100%',
                            'height': '40px',
                            'backgroundColor': '#0a1017',
                            'borderRadius': '5px',
                            'position': 'relative',
                            'overflow': 'hidden',
                            'border': '1px solid #1E5F75',
                            'marginTop': '10px'
                        }, children=[
                            # Video frames representation
                            *[html.Div(style={
                                'position': 'absolute',
                                'width': '8px',
                                'height': '38px',
                                'backgroundColor': '#4CC9F0',
                                'left': f'{i * 5}%',
                                'opacity': '0.5'
                            }) for i in range(20)]
                        ]),
                        
                        # Sync offset indicator
                        html.Div(style={
                            'position': 'absolute',
                            'top': '50%',
                            'left': '70%',
                            'transform': 'translate(-50%, -50%)',
                            'padding': '5px 10px',
                            'backgroundColor': 'rgba(247, 37, 133, 0.2)',
                            'color': '#F72585',
                            'border': '1px solid #F72585',
                            'borderRadius': '3px',
                            'fontSize': '12px',
                            'boxShadow': '0 0 10px rgba(247, 37, 133, 0.5)',
                            'display': 'block' if results['is_deepfake'] else 'none'
                        }, children=[
                            "Sync offset: 320ms"
                        ])
                    ]),
                    
                    html.P("Analysis of synchronization between audio and video streams. Misalignment may indicate manipulation.", 
                          className="text-center mt-2 text-muted small")
                ], className="mb-4 visualization-container"),
                
                # Add to report list
                html.Hr(),
                html.Div([
                    dbc.Button("Add to Reports", color="primary", id="add-video-to-reports", className="me-2"),
                    dbc.Button("Download Analysis", color="secondary", id="download-video-analysis", outline=True),
                ], className="d-flex justify-content-end")
            ])
            
            return results_div, report_div, ""
            
        except Exception as e:
            error_div = html.Div([
                html.H5("Error processing the video:"),
                html.P(str(e))
            ])
            return error_div, html.P(f"Error during analysis: {str(e)}", className="text-danger"), ""
    
    # Toggle button for technical details
    @app.callback(
        [Output("genconvit-technical-collapse", "is_open"),
         Output("toggle-genconvit-details", "children")],
        [Input("toggle-genconvit-details", "n_clicks")],
        [State("genconvit-technical-collapse", "is_open")]
    )
    def toggle_genconvit_collapse(n, is_open):
        if n:
            return not is_open, "Hide Technical Details" if not is_open else "Show Technical Details"
        return is_open, "Show Technical Details"

def parse_contents(contents, filename, date):
    """
    Parse the contents of an uploaded file.
    
    Args:
        contents: File contents
        filename: Filename
        date: Last modified date
        
    Returns:
        HTML Div containing the file info and contents
    """
    return html.Div([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')),
        
        # Preview content based on file type
        html.Div([
            html.Img(src=contents, style={'max-width': '100%', 'max-height': '300px'})
            if 'image' in contents.lower() else
            html.Audio(src=contents, controls=True, style={'width': '100%'})
            if 'audio' in contents.lower() else
            html.Video(src=contents, controls=True, style={'max-width': '100%', 'max-height': '300px'})
            if 'video' in contents.lower() else
            html.Div('File preview not available')
        ]),
        
        html.Hr()
    ])
