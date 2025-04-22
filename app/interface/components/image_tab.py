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
import plotly.graph_objects as go
from typing import Dict, Any, List
import json

from app.utils.visualization import VisualizationManager
from app.utils.premium_utils import get_premium_badge_html
from detectors.image_detector.face_detector import FaceDetector
from app.interface.components.detailed_analysis_panel import create_detailed_analysis_panel

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
                                    {"label": html.Span([
                                        "BEiT (Bidirectional Encoder)",
                                        get_premium_badge_html("beit", "image")
                                    ]), "value": "beit"},
                                    {"label": "DeiT (Distilled Transformer)", "value": "deit"},
                                    {"label": html.Span([
                                        "Swin (Hierarchical Transformer)",
                                        get_premium_badge_html("swin", "image")
                                    ]), "value": "swin"},
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
                    # Main results container
                    html.Div(id="image-results-container", className="mb-4"),
                    
                    # Heatmap visualization and controls
                    html.Div([
                        html.H5("Attention Heatmap Visualization", className="mb-3"),
                        
                        # Visualization container
                        dbc.Row([
                            # Original image column
                            dbc.Col([
                                html.H6("Original Image", className="text-center mb-2"),
                                html.Div(id="original-image-container", className="d-flex justify-content-center")
                            ], md=6, className="mb-3"),
                            
                            # Heatmap overlay column
                            dbc.Col([
                                html.H6("Attention Heatmap", className="text-center mb-2"),
                                html.Div(id="heatmap-image-container", className="d-flex justify-content-center")
                            ], md=6, className="mb-3")
                        ], className="mb-4"),
                        
                        # Heatmap controls
                        dbc.Row([
                            # Colormap selection
                            dbc.Col([
                                html.Label("Colormap:", className="fw-bold"),
                                dbc.Select(
                                    id='heatmap-colormap-select',
                                    options=[
                                        {"label": "Inferno (Red-Yellow)", "value": "inferno"},
                                        {"label": "Viridis (Blue-Green-Yellow)", "value": "viridis"},
                                        {"label": "Plasma (Purple-Orange)", "value": "plasma"},
                                        {"label": "Magma (Purple-Red)", "value": "magma"},
                                        {"label": "Turbo (Blue-Green-Red)", "value": "turbo"},
                                        {"label": "Jet (Blue-Green-Red-Yellow)", "value": "jet"}
                                    ],
                                    value='inferno',
                                    className="mb-3"
                                )
                            ], md=6),
                            
                            # Opacity slider
                            dbc.Col([
                                html.Label("Overlay Opacity:", className="fw-bold"),
                                dcc.Slider(
                                    id='heatmap-opacity-slider',
                                    min=0.1,
                                    max=0.9,
                                    step=0.05,
                                    value=0.6,
                                    marks={
                                        0.1: {'label': '10%', 'style': {'color': '#77B5FE'}},
                                        0.5: {'label': '50%', 'style': {'color': '#77B5FE'}},
                                        0.9: {'label': '90%', 'style': {'color': '#77B5FE'}}
                                    },
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        # Explanation text
                        html.Div([
                            html.P([
                                html.I(className="fas fa-info-circle me-2"),
                                "The heatmap shows regions that influenced the deepfake detection decision. ",
                                "Brighter areas indicate higher attention by the model, which may indicate manipulated regions."
                            ], className="small text-muted")
                        ])
                    ], id="heatmap-visualization-container", style={"display": "none"}),
                    
                    # Detailed Analysis Report Panel
                    html.Div(
                        id="image-detailed-analysis-container",
                        style={"display": "none"}
                    ),
                    
                    # Divider before detailed analysis
                    html.Hr(className="my-4")
                ])
            ], className="shadow")
        ], id="image-results-section")
    ])
    
    # Register callbacks for image analysis
    _register_image_callbacks(app)
    
    return image_tab

def _register_image_callbacks(app: dash.Dash):
    """
    Register callbacks for the image tab.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("heatmap-visualization-container", "style"),
        Input("image-results-container", "children"),
        prevent_initial_call=True
    )
    def show_heatmap_container(results_content):
        """Show heatmap container when results are available"""
        if results_content:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        [Output("original-image-container", "children"),
         Output("heatmap-image-container", "children")],
        [Input("image-results-container", "children"),
         Input("heatmap-colormap-select", "value"),
         Input("heatmap-opacity-slider", "value")],
        prevent_initial_call=True
    )
    def update_heatmap_visualization(results_content, colormap, opacity):
        """Update the heatmap visualization based on results and user settings"""
        if not results_content:
            return html.Div(), html.Div()
        
        try:
            # In a production implementation, we would extract the actual image and attention map 
            # from the detection results
            
            # Initialize visualization and face detector
            viz_manager = VisualizationManager()
            face_detector = FaceDetector()
            
            # In production, we would load the original image from storage
            # For this demo, we'll use a mock approach with placeholder images
            
            # Original image container
            original_img_base64 = "/assets/placeholder-original.jpg"
            original_img = html.Div([
                html.Img(
                    id="original-image-display",
                    src=original_img_base64,
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "300px",
                        "border": "1px solid #1E5F75",
                        "borderRadius": "4px"
                    }
                )
            ], className="text-center")
            
            # Draw face bounding boxes
            # In production, we would:
            # 1. Load the actual image that was analyzed
            # 2. Detect faces or retrieve face detections from the detection results
            # 3. Draw the face bounding boxes using the visualization manager
            face_detections = [
                {'bbox': (50, 30, 150, 150), 'confidence': 0.92},
                {'bbox': (250, 50, 120, 120), 'confidence': 0.87}
            ]
            
            # Create a base64 encoded image with face bounding boxes
            face_overlay_base64 = "/assets/placeholder-face-boxes.jpg"
            
            # Heatmap image container
            heatmap_img = html.Div([
                html.Div([
                    # Toggle between heatmap and face bounding boxes
                    dbc.ButtonGroup([
                        dbc.Button("Heatmap", id="show-heatmap-btn", color="primary", outline=True, size="sm", active=True),
                        dbc.Button("Face Boxes", id="show-face-boxes-btn", color="primary", outline=True, size="sm")
                    ], className="mb-2")
                ], className="text-center"),
                
                # Heatmap overlay image
                html.Img(
                    id="heatmap-overlay-image",
                    src="/assets/placeholder-heatmap.jpg",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "300px",
                        "border": "1px solid #1E5F75",
                        "borderRadius": "4px",
                        "opacity": opacity,  # Apply user-selected opacity
                        "display": "block"
                    }
                ),
                
                # Face boxes overlay image (initially hidden)
                html.Img(
                    id="face-boxes-image",
                    src=face_overlay_base64,
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "300px",
                        "border": "1px solid #1E5F75",
                        "borderRadius": "4px",
                        "display": "none"
                    }
                )
            ], className="text-center")
            
            return original_img, heatmap_img
            
        except Exception as e:
            print(f"Error creating heatmap visualization: {str(e)}")
            error_div = html.Div([
                html.P("Error loading visualization", className="text-danger"),
                html.Small(str(e), className="text-muted")
            ])
            return error_div, error_div
    
    @app.callback(
        [Output("heatmap-overlay-image", "style"),
         Output("face-boxes-image", "style")],
        [Input("show-heatmap-btn", "active"),
         Input("show-face-boxes-btn", "active"),
         Input("heatmap-opacity-slider", "value")],
        prevent_initial_call=True
    )
    def toggle_visualization_type(show_heatmap, show_boxes, opacity):
        """Toggle between heatmap and face boxes visualization"""
        heatmap_style = {
            "maxWidth": "100%",
            "maxHeight": "300px",
            "border": "1px solid #1E5F75",
            "borderRadius": "4px",
            "opacity": opacity,
            "display": "block" if show_heatmap else "none"
        }
        
        boxes_style = {
            "maxWidth": "100%",
            "maxHeight": "300px",
            "border": "1px solid #1E5F75",
            "borderRadius": "4px",
            "display": "block" if show_boxes else "none"
        }
        
        return heatmap_style, boxes_style
            
    @app.callback(
        [Output("image-detailed-analysis-container", "children"),
         Output("image-detailed-analysis-container", "style")],
        Input("image-results-container", "children"),
        prevent_initial_call=True
    )
    def update_detailed_analysis_panel(results_content):
        """Show detailed analysis panel when results are available"""
        if not results_content:
            return None, {"display": "none"}
        
        try:
            # In a production environment, we would retrieve the actual detection results
            # For this demo, we'll use placeholder data
            mock_detection_result = {
                "is_deepfake": True,
                "confidence": 0.87,
                "detection_time_ms": 346,
                "manipulation_type": "Face Swap",
                "analysis_mode": "Visual Sentinel™",
                "model_contributions": [
                    {"model": "ViT", "confidence": 0.82, "weight": 0.25},
                    {"model": "DeiT", "confidence": 0.75, "weight": 0.20},
                    {"model": "BEiT", "confidence": 0.90, "weight": 0.30},
                    {"model": "Swin", "confidence": 0.78, "weight": 0.25}
                ],
                "model_results": [
                    {"model": "ViT", "is_deepfake": True, "confidence": 0.82},
                    {"model": "DeiT", "is_deepfake": True, "confidence": 0.75},
                    {"model": "BEiT", "is_deepfake": True, "confidence": 0.90},
                    {"model": "Swin", "is_deepfake": False, "confidence": 0.22}
                ]
            }
            
            # Create the detailed analysis panel
            panel = create_detailed_analysis_panel("image", mock_detection_result)
            
            return panel, {"display": "block"}
            
        except Exception as e:
            print(f"Error creating detailed analysis panel: {str(e)}")
            return html.Div([
                html.P("Error loading detailed analysis", className="text-danger"),
                html.Small(str(e), className="text-muted")
            ]), {"display": "block"}
            
    # Add other callbacks for image tab functionality
