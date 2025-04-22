"""
Detailed analysis report panel component for the Deepfake Detection Platform UI.
Displays metrics, model contributions, and export options in a three-column grid.
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px

def create_detailed_analysis_panel(media_type: str, detection_result: Dict[str, Any] = None) -> html.Div:
    """
    Create a detailed analysis report panel with three-column grid layout.
    
    Args:
        media_type: Type of media being analyzed ('image', 'audio', or 'video')
        detection_result: Detection result dictionary (if available)
        
    Returns:
        Dash HTML Division containing the detailed analysis panel
    """
    # Default values if detection_result is None
    is_deepfake = False
    confidence = 0.0
    model_contributions = []
    detection_time_ms = 0
    manipulation_type = "Unknown"
    singularity_mode = "Standard Analysis"
    model_consensus = "0/0"
    
    # Extract information from detection_result if available
    if detection_result:
        is_deepfake = detection_result.get("is_deepfake", False)
        confidence = detection_result.get("confidence", 0.0)
        model_contributions = detection_result.get("model_contributions", [])
        detection_time_ms = detection_result.get("detection_time_ms", 0)
        manipulation_type = detection_result.get("manipulation_type", "Unknown")
        singularity_mode = detection_result.get("analysis_mode", "Standard Analysis")
        
        # Calculate model consensus
        if "model_results" in detection_result:
            models_total = len(detection_result["model_results"])
            models_detecting_deepfake = sum(1 for model in detection_result["model_results"] 
                                         if model.get("is_deepfake", False))
            model_consensus = f"{models_detecting_deepfake}/{models_total}"
    
    # Create the three-column grid layout
    detailed_panel = html.Div([
        dbc.Row([
            # Column 1: Detection Metrics
            dbc.Col([
                html.H5("Detection Metrics", className="text-primary mb-3"),
                
                # Create metrics cards
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Analysis Mode", className="card-subtitle text-muted mb-1"),
                        html.P(singularity_mode, className="mb-0 fs-5")
                    ])
                ], className="mb-2 shadow-sm"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Detection Time", className="card-subtitle text-muted mb-1"),
                        html.P([
                            f"{detection_time_ms:.0f}",
                            html.Span(" ms", className="text-muted small")
                        ], className="mb-0 fs-5")
                    ])
                ], className="mb-2 shadow-sm"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Manipulation Type", className="card-subtitle text-muted mb-1"),
                        html.P(manipulation_type, className="mb-0 fs-5")
                    ])
                ], className="mb-2 shadow-sm"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Model Consensus", className="card-subtitle text-muted mb-1"),
                        html.P(model_consensus, className="mb-0 fs-5")
                    ])
                ], className="mb-2 shadow-sm")
            ], md=4, className="mb-3"),
            
            # Column 2: Model Contributions
            dbc.Col([
                html.H5("Model Contributions", className="text-primary mb-3"),
                
                # Create model contributions chart
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id=f"{media_type}-model-contributions-chart",
                            figure=create_model_contributions_chart(model_contributions),
                            config={
                                'displayModeBar': False
                            },
                            style={"height": "260px"}
                        )
                    ])
                ], className="shadow-sm")
            ], md=4, className="mb-3"),
            
            # Column 3: Export Options
            dbc.Col([
                html.H5("Export Options", className="text-primary mb-3"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.P("Export your analysis results in various formats for documentation and sharing.", 
                               className="small text-muted mb-3"),
                        
                        # PDF Report button
                        dbc.Button([
                            html.I(className="fas fa-file-pdf me-2"),
                            "Export PDF Report"
                        ], 
                        id=f"{media_type}-export-pdf-btn",
                        color="primary",
                        className="w-100 mb-2"),
                        
                        # JSON Metadata button
                        dbc.Button([
                            html.I(className="fas fa-file-code me-2"),
                            "Download JSON Metadata"
                        ],
                        id=f"{media_type}-export-json-btn",
                        color="secondary",
                        outline=True,
                        className="w-100 mb-2"),
                        
                        # Embed Code button
                        dbc.Button([
                            html.I(className="fas fa-code me-2"),
                            "Get Embed Code"
                        ],
                        id=f"{media_type}-export-embed-btn",
                        color="secondary",
                        outline=True,
                        className="w-100 mb-2"),
                        
                        # Share Results button
                        dbc.Button([
                            html.I(className="fas fa-share-alt me-2"),
                            "Share Results"
                        ],
                        id=f"{media_type}-share-results-btn",
                        color="info",
                        outline=True,
                        className="w-100")
                    ])
                ], className="shadow-sm h-100")
            ], md=4, className="mb-3")
        ])
    ], className="mt-4 detailed-analysis-panel")
    
    return detailed_panel

def create_model_contributions_chart(model_contributions: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a bar chart showing the contributions of different models to the final result.
    
    Args:
        model_contributions: List of model contribution dictionaries
        
    Returns:
        Plotly figure object
    """
    # Use placeholder data if no real contributions provided
    if not model_contributions:
        model_contributions = [
            {"model": "ViT", "confidence": 0.82, "weight": 0.25},
            {"model": "DeiT", "confidence": 0.75, "weight": 0.20},
            {"model": "BEiT", "confidence": 0.90, "weight": 0.30},
            {"model": "Swin", "confidence": 0.78, "weight": 0.25}
        ]
    
    # Extract data for the chart
    models = [item["model"] for item in model_contributions]
    confidences = [item["confidence"] for item in model_contributions]
    weights = [item.get("weight", 1.0/len(model_contributions)) for item in model_contributions]
    
    # Calculate weighted contributions
    weighted_scores = [conf * weight for conf, weight in zip(confidences, weights)]
    
    # Create figure with two traces: raw confidence and weighted contribution
    fig = go.Figure()
    
    # Add raw confidence bars
    fig.add_trace(go.Bar(
        x=models,
        y=confidences,
        name="Raw Confidence",
        marker_color="#4CC9F0",
        opacity=0.7
    ))
    
    # Add weighted contribution bars
    fig.add_trace(go.Bar(
        x=models,
        y=weighted_scores,
        name="Weighted Contribution",
        marker_color="#4361EE"
    ))
    
    # Update the layout
    fig.update_layout(
        title="",
        barmode='group',
        xaxis=dict(
            title="",
            tickfont=dict(size=10),
            showgrid=False
        ),
        yaxis=dict(
            title="Score",
            range=[0, 1.05],
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(82, 113, 255, 0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=30, r=30, t=30, b=30),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#FFFFFF"),
        height=260
    )
    
    return fig
