"""
Settings tab component for the Deepfake Detection Platform.
Provides configuration options for API keys and other platform settings.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import os
import yaml
from typing import Dict, Any

from app.utils.premium_utils import load_config, get_config_path


def create_settings_tab(app: dash.Dash) -> html.Div:
    """
    Create the settings tab layout.
    
    Args:
        app: The Dash application instance
        
    Returns:
        html.Div: The settings tab layout
    """
    # Load current configuration
    config = load_config()
    api_keys = config.get('api_keys', {})
    
    # Register callbacks
    register_settings_callbacks(app)
    
    return html.Div([
        html.H3("Settings & Configuration", className="text-glow mb-4 text-center"),
        
        # API Keys Section
        dbc.Card([
            dbc.CardHeader(html.H4("API Keys for Premium Features", className="mb-0")),
            dbc.CardBody([
                html.P([
                    "Configure API keys to unlock premium model features. Premium models offer enhanced accuracy and ",
                    "specialized capabilities for advanced deepfake detection."
                ], className="mb-4"),
                
                # Hugging Face API Key
                dbc.Row([
                    dbc.Col([
                        html.Label("Hugging Face API Key", className="fw-bold"),
                        dbc.Input(
                            id="huggingface-api-key",
                            type="password",
                            placeholder="Enter Hugging Face API key",
                            value=api_keys.get("huggingface", ""),
                            className="mb-2"
                        ),
                        html.Small([
                            "Used for BEiT image models and other premium transformers. ",
                            html.A("Get API key", href="https://huggingface.co/settings/tokens", target="_blank")
                        ], className="text-muted d-block mb-3")
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-info-circle me-2"),
                            "Premium Models Enabled: ",
                            html.Span("BEiT", className="badge bg-info me-1"),
                            html.Span("DIT++", className="badge bg-info me-1"),
                        ], className="mb-2 mt-4"),
                        dbc.Button(
                            "Verify Key", 
                            id="verify-huggingface-btn", 
                            color="secondary", 
                            size="sm",
                            className="mt-1"
                        ),
                        html.Div(id="huggingface-verify-output", className="mt-2")
                    ], md=6)
                ], className="mb-4"),
                
                # Swin API Key
                dbc.Row([
                    dbc.Col([
                        html.Label("Swin Transformer API Key", className="fw-bold"),
                        dbc.Input(
                            id="swin-api-key",
                            type="password",
                            placeholder="Enter Swin API key",
                            value=api_keys.get("swin", ""),
                            className="mb-2"
                        ),
                        html.Small([
                            "Used for hierarchical Swin transformer models. ",
                            html.A("Get API key", href="https://ai.meta.com/resources/models-and-libraries/swin/", target="_blank")
                        ], className="text-muted d-block mb-3")
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-info-circle me-2"),
                            "Premium Models Enabled: ",
                            html.Span("Swin-B", className="badge bg-info me-1"),
                            html.Span("Swin-L", className="badge bg-info me-1"),
                        ], className="mb-2 mt-4"),
                        dbc.Button(
                            "Verify Key", 
                            id="verify-swin-btn", 
                            color="secondary", 
                            size="sm",
                            className="mt-1"
                        ),
                        html.Div(id="swin-verify-output", className="mt-2")
                    ], md=6)
                ], className="mb-4"),
                
                # XLSR/Mamba API Key
                dbc.Row([
                    dbc.Col([
                        html.Label("XLSR/Mamba API Key", className="fw-bold"),
                        dbc.Input(
                            id="xlsr-mamba-api-key",
                            type="password",
                            placeholder="Enter XLSR/Mamba API key",
                            value=api_keys.get("xlsr_mamba", ""),
                            className="mb-2"
                        ),
                        html.Small([
                            "Used for advanced audio models with cross-lingual capabilities. ",
                            html.A("Get API key", href="https://ai.meta.com/resources/models-and-libraries/xlsr/", target="_blank")
                        ], className="text-muted d-block mb-3")
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-info-circle me-2"),
                            "Premium Models Enabled: ",
                            html.Span("XLSR", className="badge bg-info me-1"),
                            html.Span("Mamba-Audio", className="badge bg-info me-1"),
                        ], className="mb-2 mt-4"),
                        dbc.Button(
                            "Verify Key", 
                            id="verify-xlsr-mamba-btn", 
                            color="secondary", 
                            size="sm",
                            className="mt-1"
                        ),
                        html.Div(id="xlsr-mamba-verify-output", className="mt-2")
                    ], md=6)
                ], className="mb-4"),
                
                # TimeSformer API Key
                dbc.Row([
                    dbc.Col([
                        html.Label("TimeSformer API Key", className="fw-bold"),
                        dbc.Input(
                            id="timesformer-api-key",
                            type="password",
                            placeholder="Enter TimeSformer API key",
                            value=api_keys.get("timesformer", ""),
                            className="mb-2"
                        ),
                        html.Small([
                            "Used for temporal video analysis with transformers. ",
                            html.A("Get API key", href="https://ai.meta.com/resources/models-and-libraries/timesformer/", target="_blank")
                        ], className="text-muted d-block mb-3")
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-info-circle me-2"),
                            "Premium Models Enabled: ",
                            html.Span("TimeSformer", className="badge bg-info me-1"),
                        ], className="mb-2 mt-4"),
                        dbc.Button(
                            "Verify Key", 
                            id="verify-timesformer-btn", 
                            color="secondary", 
                            size="sm",
                            className="mt-1"
                        ),
                        html.Div(id="timesformer-verify-output", className="mt-2")
                    ], md=6)
                ], className="mb-4"),
                
                # Video-Swin API Key
                dbc.Row([
                    dbc.Col([
                        html.Label("Video-Swin API Key", className="fw-bold"),
                        dbc.Input(
                            id="video-swin-api-key",
                            type="password",
                            placeholder="Enter Video-Swin API key",
                            value=api_keys.get("video_swin", ""),
                            className="mb-2"
                        ),
                        html.Small([
                            "Used for hierarchical video transformers. ",
                            html.A("Get API key", href="https://ai.meta.com/resources/models-and-libraries/video-swin/", target="_blank")
                        ], className="text-muted d-block mb-3")
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-info-circle me-2"),
                            "Premium Models Enabled: ",
                            html.Span("Video-Swin", className="badge bg-info me-1"),
                        ], className="mb-2 mt-4"),
                        dbc.Button(
                            "Verify Key", 
                            id="verify-video-swin-btn", 
                            color="secondary", 
                            size="sm",
                            className="mt-1"
                        ),
                        html.Div(id="video-swin-verify-output", className="mt-2")
                    ], md=6)
                ], className="mb-4"),
                
                # Save Button
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-save me-2"), "Save API Keys"], 
                            id="save-api-keys-btn", 
                            color="primary", 
                            className="w-100"
                        ),
                        html.Div(id="save-api-keys-output", className="mt-3")
                    ], md={"size": 6, "offset": 3})
                ])
            ])
        ], className="mb-4 shadow"),
        
        # Premium Features Documentation
        dbc.Card([
            dbc.CardHeader(html.H4("Premium Features Documentation", className="mb-0")),
            dbc.CardBody([
                html.P([
                    "Premium models provide enhanced detection capabilities and specialized features ",
                    "for more accurate deepfake analysis."
                ]),
                
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P("Premium image models offer advanced capabilities:"),
                        html.Ul([
                            html.Li("BEiT - Bidirectional Encoder representation from Image Transformers with SOTA performance"),
                            html.Li("Swin - Hierarchical Transformer using shifted windows for enhanced spatial detail"),
                            html.Li("Improved attention map visualization for more precise manipulation localization"),
                            html.Li("Higher resolution processing for detecting subtle manipulations")
                        ])
                    ], title="Premium Image Models"),
                    
                    dbc.AccordionItem([
                        html.P("Premium audio models offer advanced capabilities:"),
                        html.Ul([
                            html.Li("XLSR - Cross-lingual speech representation model with support for 25+ languages"),
                            html.Li("Mamba-Audio - State space model for efficient long sequence modeling"),
                            html.Li("Enhanced voice clone detection with specialized training"),
                            html.Li("Low-resource language support for global content analysis")
                        ])
                    ], title="Premium Audio Models"),
                    
                    dbc.AccordionItem([
                        html.P("Premium video models offer advanced capabilities:"),
                        html.Ul([
                            html.Li("TimeSformer - Time-Space Transformer for temporal consistency analysis"),
                            html.Li("Video-Swin - Hierarchical video transformer with window attention"),
                            html.Li("Temporal anomaly detection for frame-level manipulation identification"),
                            html.Li("High-resolution processing with reduced computational overhead")
                        ])
                    ], title="Premium Video Models"),
                    
                    dbc.AccordionItem([
                        html.P("To obtain API keys for premium models:"),
                        html.Ol([
                            html.Li([
                                "Visit the respective model provider websites (links above)"
                            ]),
                            html.Li([
                                "Sign up for an account and request API access"
                            ]),
                            html.Li([
                                "Enter your API keys in the settings panel"
                            ]),
                            html.Li([
                                "Verify your keys using the verify buttons"
                            ])
                        ]),
                        html.P([
                            "Note: Some models may require approval from the model provider. ",
                            "Please refer to the provider's documentation for more details."
                        ])
                    ], title="How to Obtain API Keys"),
                ], flush=True, start_collapsed=True)
            ])
        ], className="shadow")
    ], className="tab-content")


def register_settings_callbacks(app: dash.Dash) -> None:
    """
    Register callbacks for the settings tab.
    
    Args:
        app: The Dash application instance
    """
    @callback(
        Output("save-api-keys-output", "children"),
        Input("save-api-keys-btn", "n_clicks"),
        [
            State("huggingface-api-key", "value"),
            State("swin-api-key", "value"),
            State("xlsr-mamba-api-key", "value"),
            State("timesformer-api-key", "value"),
            State("video-swin-api-key", "value")
        ],
        prevent_initial_call=True
    )
    def save_api_keys(n_clicks, huggingface_key, swin_key, xlsr_mamba_key, timesformer_key, video_swin_key):
        if n_clicks is None:
            return ""
            
        try:
            # Load current config
            config_path = get_config_path()
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Update API keys
            if 'api_keys' not in config:
                config['api_keys'] = {}
                
            config['api_keys']['huggingface'] = huggingface_key or ""
            config['api_keys']['swin'] = swin_key or ""
            config['api_keys']['xlsr_mamba'] = xlsr_mamba_key or ""
            config['api_keys']['timesformer'] = timesformer_key or ""
            config['api_keys']['video_swin'] = video_swin_key or ""
            
            # Save updated config
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
                
            return html.Div(
                "API keys saved successfully! Premium models are now available.",
                className="text-success text-center"
            )
            
        except Exception as e:
            return html.Div(
                f"Error saving API keys: {str(e)}",
                className="text-danger text-center"
            )
            
    # Verify key callbacks - one for each provider
    @callback(
        Output("huggingface-verify-output", "children"),
        Input("verify-huggingface-btn", "n_clicks"),
        State("huggingface-api-key", "value"),
        prevent_initial_call=True
    )
    def verify_huggingface_key(n_clicks, api_key):
        if n_clicks is None or not api_key:
            return ""
            
        # In a real implementation, this would make an API call to validate the key
        # For this demo, we'll just check if the key looks valid (non-empty)
        if api_key and len(api_key) > 8:
            return html.Div("API key appears valid", className="text-success small")
        else:
            return html.Div("Invalid API key format", className="text-danger small")
    
    # Similar verify callbacks for other API keys would be implemented here
    @callback(
        Output("swin-verify-output", "children"),
        Input("verify-swin-btn", "n_clicks"),
        State("swin-api-key", "value"),
        prevent_initial_call=True
    )
    def verify_swin_key(n_clicks, api_key):
        if n_clicks is None or not api_key:
            return ""
        if api_key and len(api_key) > 8:
            return html.Div("API key appears valid", className="text-success small")
        else:
            return html.Div("Invalid API key format", className="text-danger small")
