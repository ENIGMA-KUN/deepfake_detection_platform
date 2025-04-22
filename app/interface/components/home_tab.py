"""
Home tab component for the Deepfake Detection Platform UI.
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_home_tab(app: dash.Dash) -> html.Div:
    """
    Create the home tab component.
    
    Args:
        app: Dash application instance
        
    Returns:
        Dash HTML Division containing the home tab layout
    """
    # Create home tab layout
    home_tab = html.Div([
        # Welcome heading
        html.H2("Welcome to the Deepfake Detection Platform", 
                className="text-center my-4"),
        
        # Introduction text
        html.P(
            "This platform uses advanced deep learning models to analyze media content (images, audio, video) and detect potential deepfake manipulations.",
            className="text-center mb-5"
        ),
        
        # Analysis Options - Modern UI
        html.H3("Media Analysis", className="section-title-modern"),
        html.Div([
            # Image Analysis Card
            html.Div([
                html.Div([
                    html.H4("Image Analysis", className="mb-2"),
                    # Singularity Mode Indicator
                    html.Div([
                        html.Span("Visual Sentinel", className="singularity-mode-title"),
                        html.Div("ACTIVE", className="singularity-active-indicator")
                    ], className="singularity-mode"),
                ], className="analysis-header"),
                
                html.Div([
                    html.P(
                        "Analyze static images for potential manipulation using Vision Transformer models."
                    ),
                    html.Div([
                        html.Div([
                            html.Span("Accuracy: ", className="fw-bold"),
                            html.Span("97.9%", className="model-spec-value")
                        ], className="model-spec-item"),
                    ], className="model-specs")
                ], className="analysis-content"),
                
                html.Div([
                    html.Button(
                        "ANALYZE IMAGES",
                        id="home-image-button",
                        className="analyze-button-modern"
                    )
                ], className="analysis-footer")
            ], className="analysis-card-modern"),
            
            # Audio Analysis Card
            html.Div([
                html.Div([
                    html.H4("Audio Analysis", className="mb-2"),
                    # Singularity Mode Indicator
                    html.Div([
                        html.Span("Acoustic Guardian", className="singularity-mode-title"),
                        html.Div("ACTIVE", className="singularity-active-indicator")
                    ], className="singularity-mode"),
                ], className="analysis-header"),
                
                html.Div([
                    html.P(
                        "Detect synthetic or manipulated audio using advanced speech models."
                    ),
                    html.Div([
                        html.Div([
                            html.Span("Accuracy: ", className="fw-bold"),
                            html.Span("96.2%", className="model-spec-value")
                        ], className="model-spec-item"),
                    ], className="model-specs")
                ], className="analysis-content"),
                
                html.Div([
                    html.Button(
                        "ANALYZE AUDIO",
                        id="home-audio-button",
                        className="analyze-button-modern"
                    )
                ], className="analysis-footer")
            ], className="analysis-card-modern"),
            
            # Video Analysis Card
            html.Div([
                html.Div([
                    html.H4("Video Analysis", className="mb-2"),
                    # Singularity Mode Indicator
                    html.Div([
                        html.Span("Temporal Oracle", className="singularity-mode-title"),
                        html.Div("ACTIVE", className="singularity-active-indicator")
                    ], className="singularity-mode"),
                ], className="analysis-header"),
                
                html.Div([
                    html.P(
                        "Identify deepfake videos through frame and temporal consistency analysis."
                    ),
                    html.Div([
                        html.Div([
                            html.Span("Accuracy: ", className="fw-bold"),
                            html.Span("97.3%", className="model-spec-value")
                        ], className="model-spec-item"),
                    ], className="model-specs")
                ], className="analysis-content"),
                
                html.Div([
                    html.Button(
                        "ANALYZE VIDEOS",
                        id="home-video-button",
                        className="analyze-button-modern"
                    )
                ], className="analysis-footer")
            ], className="analysis-card-modern"),
        ], className="analysis-section-modern"),
        
        # Detection Models Section
        html.H3("Our Detection Models", className="section-title-modern"),
        
        # ViT Model Card - Modern
        html.Div([
            html.Div([
                html.Div([
                    html.Img(src="/assets/vit_icon.png", className="model-icon"),
                    html.H4("Visual Sentinel (ViT)", className="model-title")
                ], className="model-header-left"),
                
                html.Div([
                    html.Span("97.9%", className="accuracy-badge")
                ])
            ], className="model-header-modern"),
            
            html.Div([
                html.P(
                    "Dynamic weighted ensemble of ViT, DeiT, BEIT, and Swin Transformer models, specializing in detecting GAN-generated images, face manipulations, and inconsistencies."
                ),
                
                html.Div([
                    html.Div([
                        html.Span("Detect face manipulations and swaps")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Identify GAN-generated synthetic images")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Analyze lighting and texture inconsistencies")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Generate attribution maps highlighting manipulated regions")
                    ], className="capability-item"),
                ], className="model-capabilities")
            ], className="model-content-modern")
        ], className="model-card-modern"),
        
        # Wav2Vec Model Card - Modern
        html.Div([
            html.Div([
                html.Div([
                    html.Img(src="/assets/audio_icon.png", className="model-icon"),
                    html.H4("Acoustic Guardian (Wav2Vec)", className="model-title")
                ], className="model-header-left"),
                
                html.Div([
                    html.Span("96.2%", className="accuracy-badge")
                ])
            ], className="model-header-modern"),
            
            html.Div([
                html.P(
                    "Adaptive ensemble of Wav2Vec2, XLSR-Mamba, XLSR+SLS, and TCN-Add models for superior detection of synthetic speech, voice cloning, and audio splicing artifacts."
                ),
                
                html.Div([
                    html.Div([
                        html.Span("Detect synthetic voice generation")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Identify audio splicing and editing")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Analyze spectrogram inconsistencies")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Evaluate natural speech patterns and prosody")
                    ], className="capability-item"),
                ], className="model-capabilities")
            ], className="model-content-modern")
        ], className="model-card-modern"),
        
        # GenConViT Model Card - Modern
        html.Div([
            html.Div([
                html.Div([
                    html.Img(src="/assets/video_icon.png", className="model-icon"),
                    html.H4("Temporal Oracle (GenConViT)", className="model-title")
                ], className="model-header-left"),
                
                html.Div([
                    html.Span("97.3%", className="accuracy-badge")
                ])
            ], className="model-header-modern"),
            
            html.Div([
                html.P(
                    "Multi-modal fusion of video models with image and audio analysis for comprehensive analysis of spatial, temporal, and audio-visual inconsistencies."
                ),
                
                html.Div([
                    html.Div([
                        html.Span("Analyze frame-level and temporal consistency")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Detect facial expression inconsistencies")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Identify audio-visual synchronization issues")
                    ], className="capability-item"),
                    html.Div([
                        html.Span("Evaluate motion coherence and natural physics")
                    ], className="capability-item"),
                ], className="model-capabilities")
            ], className="model-content-modern")
        ], className="model-card-modern"),
        
        # Platform Advantages Section
        html.H3("Platform Advantages", className="section-title-modern mt-5"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Adaptive Ensemble Architecture", className="text-center mb-3"),
                        html.P(
                            "Our platform's primary innovation is its weighted ensemble approach that dynamically calibrates model contributions based on content characteristics and historical performance.",
                            className="text-center"
                        )
                    ])
                ], className="h-100 shadow-sm mb-4"),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Cross-Modal Analysis", className="text-center mb-3"),
                        html.P(
                            "By analyzing relationships between audio and visual elements, our system detects inconsistencies that single-modality approaches miss.",
                            className="text-center"
                        )
                    ])
                ], className="h-100 shadow-sm mb-4"),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Explainable Results", className="text-center mb-3"),
                        html.P(
                            "Comprehensive visualization tools provide transparent insights into detection decisions, building user trust.",
                            className="text-center"
                        )
                    ])
                ], className="h-100 shadow-sm mb-4"),
            ], md=4),
        ]),
    ], className="tab-content")
    
    return home_tab
