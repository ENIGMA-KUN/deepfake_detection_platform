"""
Image tab component for the Deepfake Detection Platform UI.
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

def create_image_tab(app: dash.Dash) -> html.Div:
    """
    Create the image analysis tab component.
    
    Args:
        app: Dash application instance
        
    Returns:
        Dash HTML Division containing the image tab layout
    """
    # Create image tab layout
    image_tab = html.Div([
        html.H3("Image Deepfake Detection", className="text-glow mb-4 text-center"),
        
        # About section - collapsible information panel
        dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    "About Image Detection",
                    id="toggle-image-about",
                    color="info",
                    outline=True,
                )
            ),
            dbc.Collapse(
                dbc.CardBody([
                    html.H5("Image Deepfake Detection", className="card-title"),
                    html.P([
                        "Our image detection module uses state-of-the-art Vision Transformer (ViT) models to identify manipulated images. ",
                        "The system analyzes pixel patterns, facial anomalies, and inconsistencies that human eyes might miss."
                    ]),
                    html.H6("Key Features:"),
                    html.Ul([
                        html.Li("Advanced facial forensics for detecting face swaps and GAN-generated faces"),
                        html.Li("Pixel-level analysis for identifying photoshopped or AI-generated images"),
                        html.Li("Multiple model ensemble approach for higher accuracy"),
                        html.Li("Detailed visual heatmaps showing suspected manipulation areas")
                    ]),
                    html.H6("Performance:"),
                    html.P([
                        "Our image detection achieves over 95% accuracy on benchmark datasets including ",
                        html.Span("FaceForensics++", style={"font-style": "italic"}),
                        " and ",
                        html.Span("DeepfakeTIMIT", style={"font-style": "italic"}),
                        "."
                    ])
                ]),
                id="image-about-collapse",
                is_open=False,
            ),
        ], className="mb-4 shadow"),
        
        dbc.Row([
            # Upload and preview column
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
                                id='image-model-select',
                                options=[
                                    {"label": "ViT (Vision Transformer)", "value": "vit"},
                                    {"label": "BEiT (Bidirectional Encoder)", "value": "beit"},
                                    {"label": "DeiT (Distilled Transformer)", "value": "deit"},
                                    {"label": "Swin (Hierarchical Transformer)", "value": "swin"},
                                    {"label": "Ensemble (All Models)", "value": "ensemble"}
                                ],
                                value='ensemble',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Analysis Mode:", className="fw-bold"),
                            dbc.RadioItems(
                                id='image-analysis-mode',
                                options=[
                                    {"label": "Standard Analysis", "value": "standard"},
                                    {"label": "Deep Analysis (slower)", "value": "deep"},
                                    {"label": "Pixel Oracle Mode", "value": "singularity"}
                                ],
                                value='standard',
                                className="mb-3"
                            )
                        ]),
                        
                        html.Div([
                            html.Label("Confidence Threshold:", className="fw-bold"),
                            html.Div([
                                dcc.Slider(
                                    id='image-confidence-threshold-slider',
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
                    html.Div(id="image-results-container")
                ])
            ], className="shadow")
        ], id="image-results-section")
    ])
    
    return image_tab
