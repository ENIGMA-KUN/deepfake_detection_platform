"""
Audio tab component for the Deepfake Detection Platform UI.
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

def create_audio_tab(app: dash.Dash) -> html.Div:
    """
    Create the audio tab component.
    
    Args:
        app: Dash application instance
        
    Returns:
        Dash HTML Division containing the audio tab layout
    """
    # Create audio tab layout
    audio_tab = html.Div([
        html.H3("Audio Deepfake Detection", className="text-glow mb-4 text-center"),
        
        # About section - collapsible information panel
        dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    "About Audio Detection",
                    id="toggle-audio-about",
                    color="info",
                    outline=True,
                )
            ),
            dbc.Collapse(
                dbc.CardBody([
                    html.H5("Audio Deepfake Detection", className="card-title"),
                    html.P([
                        "Our audio detection module uses advanced Wav2Vec and XLSR models to identify AI-generated or manipulated audio. ",
                        "The system analyzes spectral patterns, linguistic inconsistencies, and audio artifacts that may not be audible to human ears."
                    ]),
                    html.H6("Key Features:"),
                    html.Ul([
                        html.Li("Voice cloning detection with high accuracy"),
                        html.Li("Text-to-speech synthesis identification"),
                        html.Li("Audio splicing and manipulation detection"),
                        html.Li("Multi-language support for global content analysis")
                    ]),
                    html.H6("Performance:"),
                    html.P([
                        "Our audio detection achieves over 91% accuracy on benchmark datasets including ",
                        html.Span("ASVspoof", style={"font-style": "italic"}),
                        " and ",
                        html.Span("FakeAudio", style={"font-style": "italic"}),
                        "."
                    ])
                ]),
                id="audio-about-collapse",
                is_open=False,
            ),
        ], className="mb-4 shadow"),
        
        dbc.Row([
            # Upload and preview column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Upload Audio", className="text-primary mb-0")),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-audio',
                            children=html.Div([
                                html.I(className="fas fa-microphone me-2", style={"fontSize": "24px"}),
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
                            multiple=False
                        ),
                        html.Div(id='output-audio-upload', className="mt-3"),
                        html.Div([
                            dbc.Button(
                                "Analyze Audio", 
                                id="analyze-audio-button", 
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
                                id='audio-model-select',
                                options=[
                                    {"label": "Wav2Vec2 (Self-supervised)", "value": "wav2vec"},
                                    {"label": "XLSR (Cross-lingual)", "value": "xlsr"},
                                    {"label": "Mamba-Audio (State Space Model)", "value": "mamba"},
                                    {"label": "TCN (Temporal CNN)", "value": "tcn"},
                                    {"label": "Ensemble (All Models)", "value": "ensemble"}
                                ],
                                value='ensemble',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Analysis Mode:", className="fw-bold"),
                            dbc.RadioItems(
                                id='audio-analysis-mode',
                                options=[
                                    {"label": "Standard Analysis", "value": "standard"},
                                    {"label": "Deep Analysis (slower)", "value": "deep"},
                                    {"label": "Frequency Sentinel™ Mode", "value": "singularity"}
                                ],
                                value='standard',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Confidence Threshold:", className="fw-bold"),
                            html.Div([
                                dcc.Slider(
                                    id='audio-confidence-threshold-slider',
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
                                    html.Span("Higher: Less sensitive, may miss subtle manipulation", className="small text-muted")
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
                    html.Div(id="audio-results-container")
                ])
            ], className="shadow")
        ], id="audio-results-section")
    ])
    
    return audio_tab
