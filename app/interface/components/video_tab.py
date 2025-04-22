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
import plotly.graph_objects as go
from typing import Dict, Any, List
import json

from app.utils.visualization import VisualizationManager
from app.utils.premium_utils import get_premium_badge_html
from app.interface.components.detailed_analysis_panel import create_detailed_analysis_panel

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
                                    {"label": html.Span([
                                        "TimeSformer (Temporal)",
                                        get_premium_badge_html("timesformer", "video")
                                    ]), "value": "timesformer"},
                                    {"label": "SlowFast (Motion Analysis)", "value": "slowfast"},
                                    {"label": html.Span([
                                        "Video Swin (Hierarchical)",
                                        get_premium_badge_html("video_swin", "video")
                                    ]), "value": "videoswin"},
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
                                    {"label": "Temporal Oracle Mode", "value": "singularity"}
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
                    # Main results container
                    html.Div(id="video-results-container", className="mb-4"),
                    
                    # Timeline visualization container
                    html.Div([
                        html.H5("Temporal Analysis", className="mb-3"),
                        
                        # Timeline graph
                        dbc.Row([
                            dbc.Col([
                                html.H6("Confidence Timeline", className="text-center mb-2"),
                                dcc.Graph(
                                    id="video-timeline-graph",
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['toImage', 'select2d', 'lasso2d'],
                                        'displaylogo': False
                                    },
                                    style={"height": "250px", "backgroundColor": "#0d1117", "border": "1px solid #1E5F75"}
                                )
                            ], className="mb-3")
                        ]),
                        
                        # Suspicious frames preview
                        html.H5("Suspicious Frames", className="mt-4 mb-3"),
                        html.Div(id="suspicious-frames-container", className="mb-3")
                        
                    ], id="video-visualization-container", style={"display": "none"}),
                    
                    # Detailed Analysis Report Panel
                    html.Div(
                        id="video-detailed-analysis-container",
                        style={"display": "none"}
                    ),
                    
                    # Divider before detailed analysis
                    html.Hr(className="my-4")
                ])
            ], className="shadow")
        ], id="video-results-section")
    ])
    
    # Register callbacks for video analysis
    _register_video_callbacks(app)
    
    return video_tab

def _register_video_callbacks(app: dash.Dash):
    """
    Register callbacks for the video tab.
    
    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("video-visualization-container", "style"),
        Input("video-results-container", "children"),
        prevent_initial_call=True
    )
    def show_timeline(results_content):
        """Show timeline container when results are available"""
        if results_content:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("video-timeline-graph", "figure"),
        Input("video-results-container", "children"),
        State("video-confidence-threshold-slider", "value"),
        prevent_initial_call=True
    )
    def update_timeline_graph(results_content, threshold_value):
        """Update the timeline graph based on analysis results"""
        # Default empty figure
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor="#1E2A38",
            paper_bgcolor="#1E2A38",
            font=dict(color="#CDE7FB"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=250,
            xaxis=dict(
                title="Time (seconds)",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                title="Confidence Score",
                range=[0, 1],
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        if not results_content:
            return empty_fig
            
        # For now, use sample data - this will be replaced with actual data
        # from the detection results in the full implementation
        try:
            # Generate placeholder data for demo - in production this would
            # come from the video processing results
            timestamps = list(np.linspace(0, 30, 30))
            # Generate varying scores for demo purposes
            scores = [
                0.3, 0.32, 0.35, 0.4, 0.6, 0.8, 0.85, 0.82, 0.7, 0.6,
                0.5, 0.4, 0.3, 0.25, 0.3, 0.5, 0.7, 0.8, 0.9, 0.92,
                0.9, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.2, 0.3
            ]
            
            # Create the timeline figure
            fig = go.Figure()
            
            # Add confidence score line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode="lines",
                name="Confidence",
                line=dict(
                    color="#4CC9F0",
                    width=3
                ),
                hovertemplate="Time: %{x:.2f}s<br>Confidence: %{y:.2f}<extra></extra>"
            ))
            
            # Add threshold line
            fig.add_trace(go.Scatter(
                x=[min(timestamps), max(timestamps)],
                y=[threshold_value, threshold_value],
                mode="lines",
                name="Threshold",
                line=dict(
                    color="#F72585",
                    width=2,
                    dash="dash"
                ),
                hovertemplate="Threshold: %{y:.2f}<extra></extra>"
            ))
            
            # Highlight regions above threshold
            above_threshold_indices = [i for i, score in enumerate(scores) if score >= threshold_value]
            if above_threshold_indices:
                # Group consecutive indices
                groups = []
                group = [above_threshold_indices[0]]
                for i in range(1, len(above_threshold_indices)):
                    if above_threshold_indices[i] == above_threshold_indices[i-1] + 1:
                        group.append(above_threshold_indices[i])
                    else:
                        groups.append(group)
                        group = [above_threshold_indices[i]]
                if group:
                    groups.append(group)
                
                # Add shaded regions for each group
                for i, group in enumerate(groups):
                    start_idx = group[0]
                    end_idx = group[-1]
                    fig.add_vrect(
                        x0=timestamps[start_idx],
                        x1=timestamps[end_idx],
                        fillcolor="#F72585",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        annotation_text="Suspicious" if i == 0 else None,
                        annotation_position="top left",
                        annotation_font_color="#F72585",
                        annotation_font_size=12
                    )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor="#1E2A38",
                paper_bgcolor="#1E2A38",
                font=dict(color="#CDE7FB"),
                margin=dict(l=20, r=20, t=30, b=20),
                height=250,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0)"
                ),
                xaxis=dict(
                    title="Time (seconds)",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)"
                ),
                yaxis=dict(
                    title="Confidence Score",
                    range=[0, 1],
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)"
                ),
                hovermode="x unified"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating timeline: {str(e)}")
            return empty_fig
    
    @app.callback(
        Output("suspicious-frames-container", "children"),
        Input("video-timeline-graph", "figure"),
        Input("video-results-container", "children"),
        prevent_initial_call=True
    )
    def update_suspicious_frames(timeline_fig, results_content):
        """Show preview of frames with highest suspicion scores"""
        if not results_content:
            return html.Div("No suspicious frames detected")
            
        # In a real implementation, we would extract the actual frame images
        # from the video analysis results. For now, we'll create placeholder frames.
        try:
            # Create a row of dummy frame previews
            frames = []
            # Generate 5 sample frames
            for i in range(5):
                confidence = 0.6 + (i * 0.06)  # Varying confidence values
                timestamp = 5 + (i * 5)        # Frames at different timestamps
                
                # Frame card with score overlay
                frame_card = dbc.Card([
                    html.Div([
                        # Sample placeholder for frame image
                        html.Div(
                            className="placeholder-frame",
                            style={
                                "width": "120px",
                                "height": "80px",
                                "backgroundColor": f"rgba(247, 37, 133, {confidence/2})",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "borderRadius": "4px"
                            },
                            children=html.I(className="fas fa-film", style={"fontSize": "24px"})
                        ),
                        
                        # Frame metadata
                        html.Div([
                            html.Span(
                                f"{confidence:.2f}",
                                className="badge bg-danger position-absolute top-0 end-0 m-1"
                            ),
                            html.Small(
                                f"Time: {timestamp:.1f}s", 
                                className="text-muted d-block text-center mt-1"
                            )
                        ])
                    ]),
                ], 
                className="m-1 suspicious-frame", 
                style={"width": "120px", "cursor": "pointer"},
                id=f"frame-{i}")
                
                frames.append(frame_card)
            
            return frames
            
        except Exception as e:
            print(f"Error creating suspicious frames: {str(e)}")
            return html.Div("Error loading suspicious frames")
            
    # Add callback for detailed analysis panel
    @app.callback(
        [Output("video-detailed-analysis-container", "children"),
         Output("video-detailed-analysis-container", "style")],
        Input("video-results-container", "children"),
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
                "confidence": 0.79,
                "detection_time_ms": 1293,  # Longer processing time for video
                "manipulation_type": "Face Replacement",
                "analysis_mode": "Temporal Sentinel™",
                "model_contributions": [
                    {"model": "GenConViT", "confidence": 0.72, "weight": 0.20},
                    {"model": "TimeSformer", "confidence": 0.83, "weight": 0.25},
                    {"model": "SlowFast", "confidence": 0.77, "weight": 0.20},
                    {"model": "Video-Swin", "confidence": 0.85, "weight": 0.25},
                    {"model": "X3D", "confidence": 0.71, "weight": 0.10}
                ],
                "model_results": [
                    {"model": "GenConViT", "is_deepfake": True, "confidence": 0.72},
                    {"model": "TimeSformer", "is_deepfake": True, "confidence": 0.83},
                    {"model": "SlowFast", "is_deepfake": True, "confidence": 0.77},
                    {"model": "Video-Swin", "is_deepfake": True, "confidence": 0.85},
                    {"model": "X3D", "is_deepfake": False, "confidence": 0.29}
                ]
            }
            
            # Create the detailed analysis panel
            panel = create_detailed_analysis_panel("video", mock_detection_result)
            
            return panel, {"display": "block"}
            
        except Exception as e:
            print(f"Error creating detailed analysis panel: {str(e)}")
            return html.Div([
                html.P("Error loading detailed analysis", className="text-danger"),
                html.Small(str(e), className="text-muted")
            ]), {"display": "block"}
            
    # Add other callbacks for video tab functionality
