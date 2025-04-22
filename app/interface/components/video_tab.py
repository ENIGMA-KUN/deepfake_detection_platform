"""
Video tab component for the Deepfake Detection Platform UI.
"""
import os
import base64
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
from typing import Dict, Any, List

from app.utils.visualization import VisualizationManager

def create_video_tab(app: dash.Dash) -> html.Div:
    """
    Create the video tab component.
    
    Args:
        app: Dash application instance
        
    Returns:
        Dash HTML Division containing the video tab layout
    """
    # Create video tab layout
    video_tab = html.Div([
        html.H3("Video Deepfake Detection", className="text-glow mb-4 text-center"),
        
        # About section - collapsible information panel
        dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    "About Video Detection",
                    id="toggle-video-about",
                    color="info",
                    outline=True,
                )
            ),
            dbc.Collapse(
                dbc.CardBody([
                    html.H5("Video Deepfake Detection", className="card-title"),
                    html.P([
                        "Our video detection module integrates frame-level and temporal analysis to identify deepfakes in video content. ",
                        "The system can detect face swaps, lip-sync manipulations, and full synthetic generation."
                    ]),
                    html.H6("Key Features:"),
                    html.Ul([
                        html.Li("Frame-by-frame deepfake analysis"),
                        html.Li("Temporal inconsistency detection"),
                        html.Li("Facial movement analysis"),
                        html.Li("Audio-video sync verification")
                    ]),
                    html.H6("Performance:"),
                    html.P([
                        "Our video detection achieves over 89% accuracy on benchmark datasets including ",
                        html.Span("DeepFake Detection Challenge", style={"font-style": "italic"}),
                        " and ",
                        html.Span("FaceForensics++", style={"font-style": "italic"}),
                        "."
                    ])
                ]),
                id="video-about-collapse",
                is_open=False,
            ),
        ], className="mb-4 shadow"),
        
        dbc.Row([
            # Upload and preview column
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
                            multiple=False
                        ),
                        html.Div(id='output-video-upload', className="mt-3"),
                        html.Div([
                            dbc.Button(
                                "Analyze Video", 
                                id="analyze-video-button", 
                                color="primary", 
                                className="w-100 mt-3",
                                disabled=True
                            )
                        ], className="text-center")
                    ])
                ], className="h-100 shadow")
            ], md=6),
            
            # Settings column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Detection Settings", className="text-primary mb-0")),
                    dbc.CardBody([
                        html.Div([
                            html.Label("Detection Model:", className="fw-bold"),
                            dbc.Select(
                                id='video-model-select',
                                options=[
                                    {"label": "TimeSformer (Temporal)", "value": "timesformer"},
                                    {"label": "SlowFast (Motion Analysis)", "value": "slowfast"},
                                    {"label": "Video Swin (Hierarchical)", "value": "videoswin"},
                                    {"label": "X3D (Efficient 3D Conv)", "value": "x3d"},
                                    {"label": "Ensemble (All Models)", "value": "ensemble"}
                                ],
                                value='ensemble',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Analysis Mode:", className="fw-bold"),
                            dbc.RadioItems(
                                id='video-analysis-mode',
                                options=[
                                    {"label": "Standard Analysis", "value": "standard"},
                                    {"label": "Deep Analysis (slower)", "value": "deep"},
                                    {"label": "Temporal Oracle™ Mode", "value": "singularity"}
                                ],
                                value='standard',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Confidence Threshold:", className="fw-bold"),
                            html.Div([
                                dcc.Slider(
                                    id='video-confidence-threshold-slider',
                                    min=0.5,
                                    max=0.95,
                                    step=0.05,
                                    value=0.7,
                                    marks={
                                        0.5: {'label': '0.5', 'style': {'color': '#77B5FE'}},
                                        0.7: {'label': '0.7', 'style': {'color': '#77B5FE'}},
                                        0.9: {'label': '0.9', 'style': {'color': '#77B5FE'}}
                                    },
                                ),
                                html.Div([
                                    html.Span("Lower: More sensitive, may have false positives", className="small text-muted")
                                ], style={"textAlign": "left", "marginTop": "8px"}),
                                html.Div([
                                    html.Span("Higher: Less sensitive, may miss subtle fakes", className="small text-muted")
                                ], style={"textAlign": "right", "marginTop": "8px"})
                            ])
                        ])
                    ])
                ], className="h-100 shadow")
            ], md=6)
        ], className="mb-4"),
        
        # Results section (initially hidden, shown after analysis)
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H5("Analysis Results", className="text-primary mb-0")),
                dbc.CardBody([
                    html.Div(id="video-results-container")
                ])
            ], className="shadow")
        ], id="video-results-section")
    ])
    
    return video_tab
