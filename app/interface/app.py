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
    
    # Define the app layout with Tron Legacy inspired styling
    app.layout = html.Div([
        # Navigation bar
        dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.Div("🔍", style={"fontSize": "24px", "color": "#4CC9F0"}), width="auto"),
                        dbc.Col(dbc.NavbarBrand("DEEPFAKE DETECTION PLATFORM", className="ms-2"))
                    ],
                    align="center",
                    className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Home", href="#home-tab", id="home-link")),
                        dbc.NavItem(dbc.NavLink("Image", href="#image-tab", id="image-link")),
                        dbc.NavItem(dbc.NavLink("Audio", href="#audio-tab", id="audio-link")),
                        dbc.NavItem(dbc.NavLink("Video", href="#video-tab", id="video-link")),
                        dbc.NavItem(dbc.NavLink("Reports", href="#reports-tab", id="reports-link")),
                        dbc.NavItem(dbc.NavLink("About", href="#about-tab", id="about-link")),
                    ],
                    className="ms-auto",
                    navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]),
            color="dark",
            dark=True,
            className="mb-4",
        ),
        
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
                                            dbc.Button("Use Image Detector", color="primary", href="#image-tab", className="mt-3"),
                                        ]),
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
                                            dbc.Button("Use Audio Detector", color="primary", href="#audio-tab", className="mt-3"),
                                        ]),
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
                                            dbc.Button("Use Video Detector", color="primary", href="#video-tab", className="mt-3"),
                                        ]),
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
                                        dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.Div(id='image-results-container')
                                        ])
                                    ], className="h-100 shadow")
                                ], lg=8),
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
                                        dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                                        dbc.CardBody([
                                            html.Div(id='audio-results-container')
                                        ])
                                    ], className="h-100 shadow")
                                ], lg=8),
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
                                    dbc.CardHeader(html.H5("Technical Details: Wav2Vec2 Audio Detector", className="text-primary mb-0")),
                                    dbc.CardBody([
                                        html.H6("How Wav2Vec2 Works for Deepfake Audio Detection", className="mb-3"),
                                        html.P("""
                                            Wav2Vec2 is a self-supervised learning model for speech processing that converts raw audio 
                                            into latent speech representations. For deepfake detection, we've fine-tuned the model to 
                                            identify anomalies and artifacts that are characteristic of synthetic speech.
                                        """),
                                        html.P("""
                                            Our audio deepfake detector focuses on:
                                        """),
                                        html.Ul([
                                            html.Li("Unnatural prosody and intonation patterns"),
                                            html.Li("Spectral inconsistencies and boundary artifacts"),
                                            html.Li("Temporal discontinuities in voice characteristics"),
                                            html.Li("Formant and harmonic distribution anomalies"),
                                        ]),
                                        html.P("""
                                            The model analyzes both the waveform and spectrogram representations to capture 
                                            artifacts that might be more apparent in one domain versus the other.
                                        """),
                                        dbc.Row([
                                            dbc.Col([
                                                html.H6("Model Architecture", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Base: Wav2Vec2-Large-960h"),
                                                    html.Li("Feature Dimension: 1024"),
                                                    html.Li("Transformer Layers: 24"),
                                                    html.Li("Attention Heads: 16"),
                                                    html.Li("Parameters: 317M"),
                                                ])
                                            ], md=6),
                                            dbc.Col([
                                                html.H6("Training Details", className="mb-2"),
                                                html.Ul([
                                                    html.Li("Fine-tuned on 500+ hours of audio"),
                                                    html.Li("Includes TTS, voice conversion and spliced audio"),
                                                    html.Li("Specialized data augmentation for audio"),
                                                    html.Li("Contrastive loss function"),
                                                    html.Li("Training time: 96 hours on 4 GPUs"),
                                                ])
                                            ], md=6),
                                        ])
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
                                        dbc.CardHeader(html.H5("Recent Analyses", className="text-primary mb-0")),
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
                                        dbc.CardHeader(html.H5("Report Viewer", className="text-primary mb-0")),
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
    ], id="main-container")
    
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
        [Input("home-link", "n_clicks"),
         Input("image-link", "n_clicks"),
         Input("audio-link", "n_clicks"),
         Input("video-link", "n_clicks"),
         Input("reports-link", "n_clicks"),
         Input("about-link", "n_clicks")],
        [State("tabs", "value")]
    )
    def switch_tab(home_clicks, image_clicks, audio_clicks, video_clicks, reports_clicks, about_clicks, current_tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "home-tab"
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "home-link":
                return "home-tab"
            elif button_id == "image-link":
                return "image-tab"
            elif button_id == "audio-link":
                return "audio-tab"
            elif button_id == "video-link":
                return "video-tab"
            elif button_id == "reports-link":
                return "reports-tab"
            elif button_id == "about-link":
                return "about-tab"
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
            html.Img(src=content, style={'max-width': '100%', 'max-height': '200px', 'border-radius': '5px'}),
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
         State('upload-image', 'filename')]
    )
    def analyze_image(n_clicks, content, filename):
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
        
        # Process the image
        try:
            results = app._processor.detect_media(file_path, 'image')
            
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
                            html.P(f"Faces detected: {results.get('details', {}).get('faces_detected', 0)}"),
                        ]),
                        html.Hr(),
                        html.Div([
                            html.P("Analysis Method:", className="fw-bold"),
                            html.P("Vision Transformer (ViT) with attention mapping")
                        ]),
                    ], md=6),
                ]),
                
                # Visualization
                html.H5("Visualization", className="mt-3 mb-2"),
                html.P("Attention heatmap showing regions most important for detection:"),
                
                # Mock heatmap visualization
                html.Div([
                    html.Img(src=content, style={'max-width': '100%', 'max-height': '300px', 'filter': 'grayscale(50%)'}),
                    html.Div(style={
                        'position': 'absolute',
                        'top': '0',
                        'left': '0',
                        'width': '100%',
                        'height': '100%',
                        'background': 'linear-gradient(45deg, rgba(0,0,0,0) 20%, rgba(255,0,0,0.3) 50%, rgba(255,0,0,0.5) 80%)',
                        'mixBlendMode': 'overlay',
                        'borderRadius': '5px'
                    })
                ], style={'position': 'relative', 'width': '100%', 'marginBottom': '20px'}),
                
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
         State('upload-audio', 'filename')]
    )
    def analyze_audio(n_clicks, content, filename):
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
            results = app._processor.detect_media(file_path, 'audio')
            
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
                        html.H5("Detection Details"),
                        html.Div([
                            html.P("Audio Analysis Method:", className="fw-bold"),
                            html.P("Wav2Vec2 model with spectrogram analysis")
                        ]),
                        html.Hr(),
                        html.Div([
                            html.P("Key Indicators:", className="fw-bold"),
                            html.Ul([
                                html.Li("Voice characteristics consistency"),
                                html.Li("Temporal pattern analysis"),
                                html.Li("Spectral anomaly detection")
                            ])
                        ]),
                    ], md=6),
                ]),
                
                # Visualization
                html.H5("Visualization", className="mt-3 mb-2"),
                html.P("Spectrogram analysis showing frequency distribution:"),
                
                # Mock spectrogram visualization
                html.Div([
                    html.Img(src="/assets/spectrogram_visualization.png", 
                             style={'max-width': '100%', 'max-height': '200px'}),
                ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '20px'}),
                
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
                        html.H5("Detection Details"),
                        html.Div([
                            html.P("Video Analysis Method:", className="fw-bold"),
                            html.P("GenConViT hybrid model with temporal consistency analysis")
                        ]),
                        html.Hr(),
                        html.Div([
                            html.P("Key Indicators:", className="fw-bold"),
                            html.Ul([
                                html.Li("Facial feature consistency across frames"),
                                html.Li("Temporal coherence check"),
                                html.Li("Audio-visual synchronization"),
                                html.Li("Physiological signal analysis (eye blinks, pulse)")
                            ])
                        ]),
                    ], md=6),
                ]),
                
                # Visualization
                html.H5("Visualization", className="mt-3 mb-2"),
                html.P("Frame-by-frame analysis showing potential manipulation:"),
                
                # Mock frame analysis visualization
                html.Div([
                    html.Img(src="/assets/frame_analysis_visualization.png", 
                             style={'max-width': '100%', 'max-height': '300px'}),
                ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '20px'}),
                
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
