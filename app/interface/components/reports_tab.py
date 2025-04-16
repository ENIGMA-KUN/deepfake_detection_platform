"""
Reports tab component for the Deepfake Detection Platform UI.
"""
import os
import json
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import datetime
from typing import Dict, Any, List

from app.utils.visualization import VisualizationManager

def create_reports_tab(app: dash.Dash, visualizer: VisualizationManager) -> html.Div:
    """
    Create the reports tab component.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
        
    Returns:
        Dash HTML Division containing the reports tab layout
    """
    # Create reports tab layout
    reports_tab = html.Div([
        html.H3("Analysis Reports", className="text-glow mb-4 text-center"),
        
        dbc.Row([
            # Report history and filters column
            dbc.Col([
                html.Div([
                    html.H5("Report History", className="mb-3"),
                    
                    # Search and filter
                    html.Div([
                        dbc.Input(
                            id="report-search",
                            placeholder="Search reports...",
                            type="text",
                            className="mb-3"
                        ),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Media Type", className="mb-2"),
                                dbc.Select(
                                    id="report-filter-media",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Image", "value": "image"},
                                        {"label": "Audio", "value": "audio"},
                                        {"label": "Video", "value": "video"}
                                    ],
                                    value="all"
                                )
                            ], md=6),
                            
                            dbc.Col([
                                html.Label("Result", className="mb-2"),
                                dbc.Select(
                                    id="report-filter-result",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Authentic", "value": "authentic"},
                                        {"label": "Deepfake", "value": "deepfake"}
                                    ],
                                    value="all"
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Sort By", className="mb-2"),
                                dbc.Select(
                                    id="report-sort",
                                    options=[
                                        {"label": "Date (Newest First)", "value": "date_desc"},
                                        {"label": "Date (Oldest First)", "value": "date_asc"},
                                        {"label": "Confidence (High to Low)", "value": "conf_desc"},
                                        {"label": "Confidence (Low to High)", "value": "conf_asc"}
                                    ],
                                    value="date_desc"
                                )
                            ], md=6),
                            
                            dbc.Col([
                                html.Label("Date Range", className="mb-2"),
                                dbc.Select(
                                    id="report-date-range",
                                    options=[
                                        {"label": "All Time", "value": "all"},
                                        {"label": "Today", "value": "today"},
                                        {"label": "This Week", "value": "week"},
                                        {"label": "This Month", "value": "month"}
                                    ],
                                    value="all"
                                )
                            ], md=6)
                        ])
                    ], className="mb-4 p-3 border border-glow rounded"),
                    
                    # Report list
                    html.Div([
                        html.H6("Recent Reports", className="mb-3"),
                        html.Div(id="report-list", className="report-list")
                    ])
                    
                ], className="card h-100")
            ], md=4),
            
            # Report details column
            dbc.Col([
                html.Div([
                    html.H5("Report Details", className="mb-3"),
                    
                    # Initial state - no report selected
                    html.Div([
                        html.P("Select a report from the list to view details.", className="text-center py-5"),
                    ], id='no-report-selected', style={'display': 'block'}),
                    
                    # Report details view
                    html.Div([
                        # Report header
                        html.Div([
                            html.H4(id="report-title", className="results-title"),
                            html.Div(id='report-verdict', className="results-verdict")
                        ], className="results-header"),
                        
                        # Report metadata
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.P([
                                        html.Strong("Date: "),
                                        html.Span(id="report-date")
                                    ])
                                ], md=6),
                                dbc.Col([
                                    html.P([
                                        html.Strong("Media Type: "),
                                        html.Span(id="report-media-type")
                                    ])
                                ], md=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.P([
                                        html.Strong("Confidence: "),
                                        html.Span(id="report-confidence")
                                    ])
                                ], md=6),
                                dbc.Col([
                                    html.P([
                                        html.Strong("Filename: "),
                                        html.Span(id="report-filename")
                                    ])
                                ], md=6)
                            ])
                        ], className="mb-4 p-3 border-bottom"),
                        
                        # Report visualization
                        html.Div([
                            html.H5("Visualization", className="mb-3"),
                            html.Div(id="report-visualization", className="text-center")
                        ], className="mb-4"),
                        
                        # Report details
                        html.Div([
                            html.H5("Analysis Details", className="mb-3"),
                            html.Div(id="report-details")
                        ], className="mb-4"),
                        
                        # Export options
                        html.Div([
                            dbc.Button("Export as PDF", id="export-report-pdf", 
                                      color="primary", className="me-2"),
                            dbc.Button("Download Visualization", id="download-report-vis", 
                                      color="secondary", className="me-2"),
                            dbc.Button("Delete Report", id="delete-report", 
                                      color="danger")
                        ], className="mt-4 pt-3 border-top")
                        
                    ], id='report-details-view', style={'display': 'none'})
                    
                ], className="card h-100")
            ], md=8)
        ])
    ], id="reports-tab-content")
    
    # Register callbacks for the reports tab
    register_reports_callbacks(app, visualizer)
    
    return reports_tab

def register_reports_callbacks(app: dash.Dash, visualizer: VisualizationManager):
    """
    Register all callbacks for the reports tab.
    
    Args:
        app: Dash application instance
        visualizer: Visualization manager instance
    """
    # Callback to update the report list based on filters
    @app.callback(
        Output('report-list', 'children'),
        [Input('report-search', 'value'),
         Input('report-filter-media', 'value'),
         Input('report-filter-result', 'value'),
         Input('report-sort', 'value'),
         Input('report-date-range', 'value')]
    )
    def update_report_list(search_term, media_filter, result_filter, sort_by, date_range):
        # Get reports from the output directory
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(app.server.root_path))), 
                                'reports', 'output')
        
        if not os.path.exists(reports_dir):
            return html.P("No reports found.")
        
        # Get all JSON report files
        reports = []
        for filename in os.listdir(reports_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(reports_dir, filename), 'r') as f:
                        report_data = json.load(f)
                        # Add filename to report data
                        report_data['filename'] = filename
                        reports.append(report_data)
                except:
                    continue
        
        if not reports:
            return html.P("No reports found.")
        
        # Apply filters
        filtered_reports = reports.copy()
        
        # Media type filter
        if media_filter != "all":
            filtered_reports = [r for r in filtered_reports if r.get('media_type', '') == media_filter]
        
        # Result filter
        if result_filter != "all":
            is_deepfake = result_filter == "deepfake"
            filtered_reports = [r for r in filtered_reports if r.get('is_deepfake', False) == is_deepfake]
        
        # Date range filter
        if date_range != "all":
            now = datetime.now()
            
            for r in filtered_reports[:]:
                if 'timestamp' not in r:
                    filtered_reports.remove(r)
                    continue
                    
                report_date = datetime.fromtimestamp(r['timestamp'])
                
                if date_range == "today":
                    if report_date.date() != now.date():
                        filtered_reports.remove(r)
                elif date_range == "week":
                    # Simple check for same week
                    if now.toordinal() - report_date.toordinal() > 7:
                        filtered_reports.remove(r)
                elif date_range == "month":
                    if (now.year != report_date.year or 
                        now.month != report_date.month):
                        filtered_reports.remove(r)
        
        # Search term
        if search_term:
            search_term = search_term.lower()
            for r in filtered_reports[:]:
                # Search in filename, media_type, and other fields
                searchable_text = (
                    r.get('filename', '').lower() + 
                    r.get('media_type', '').lower() +
                    ('deepfake' if r.get('is_deepfake', False) else 'authentic')
                )
                if search_term not in searchable_text:
                    filtered_reports.remove(r)
        
        # Sort reports
        if sort_by == "date_desc":
            filtered_reports.sort(key=lambda r: r.get('timestamp', 0), reverse=True)
        elif sort_by == "date_asc":
            filtered_reports.sort(key=lambda r: r.get('timestamp', 0))
        elif sort_by == "conf_desc":
            filtered_reports.sort(key=lambda r: r.get('confidence', 0), reverse=True)
        elif sort_by == "conf_asc":
            filtered_reports.sort(key=lambda r: r.get('confidence', 0))
        
        # Generate report list items
        report_items = []
        for report in filtered_reports:
            # Create report card
            is_deepfake = report.get('is_deepfake', False)
            confidence = report.get('confidence', 0)
            media_type = report.get('media_type', 'unknown')
            timestamp = report.get('timestamp', 0)
            
            # Format date
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
            
            # Create a visual indicator for the report type
            if media_type == "image":
                icon = html.I(className="fas fa-image")
            elif media_type == "audio":
                icon = html.I(className="fas fa-volume-up")
            elif media_type == "video":
                icon = html.I(className="fas fa-film")
            else:
                icon = html.I(className="fas fa-file")
            
            # Get filename without .json extension
            display_name = os.path.splitext(report.get('filename', 'Report'))[0]
            
            # Create report card
            report_card = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(icon, className="report-icon")
                        ], width=2),
                        dbc.Col([
                            html.H6(display_name, className="mb-1"),
                            html.Small(f"Date: {date_str}", className="text-muted d-block"),
                            html.Small(f"Type: {media_type.capitalize()}", className="text-muted d-block")
                        ], width=7),
                        dbc.Col([
                            html.Div(
                                "DEEPFAKE" if is_deepfake else "AUTHENTIC",
                                className=f"{'deepfake-detected' if is_deepfake else 'authentic-media'} text-center"
                            ),
                            html.Small(f"Confidence: {confidence:.2f}", className="d-block text-center")
                        ], width=3)
                    ])
                ]),
                dbc.CardFooter([
                    dbc.Button("View", id={"type": "view-report", "index": report.get('filename', '')}, 
                              size="sm", color="primary", className="w-100")
                ])
            ], className="mb-3 report-card")
            
            report_items.append(report_card)
        
        if not report_items:
            return html.P("No reports match your filters.")
            
        return report_items
    
    # Callback to show report details when a report is selected
    @app.callback(
        [Output('no-report-selected', 'style'),
         Output('report-details-view', 'style'),
         Output('report-title', 'children'),
         Output('report-verdict', 'children'),
         Output('report-verdict', 'className'),
         Output('report-date', 'children'),
         Output('report-media-type', 'children'),
         Output('report-confidence', 'children'),
         Output('report-filename', 'children'),
         Output('report-visualization', 'children'),
         Output('report-details', 'children'),
         Output('download-report-vis', 'href')],
        [Input({'type': 'view-report', 'index': dash.dependencies.ALL}, 'n_clicks')],
        [State({'type': 'view-report', 'index': dash.dependencies.ALL}, 'id')]
    )
    def show_report_details(n_clicks, ids):
        # Check if any button was clicked
        if not any(n_clicks) or not ids:
            return {'display': 'block'}, {'display': 'none'}, "", "", "", "", "", "", "", None, None, ""
        
        # Find which button was clicked
        clicked_idx = next((i for i, clicks in enumerate(n_clicks) if clicks), None)
        if clicked_idx is None:
            return {'display': 'block'}, {'display': 'none'}, "", "", "", "", "", "", "", None, None, ""
        
        # Get the report filename
        report_filename = ids[clicked_idx]['index']
        
        # Load the report data
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(app.server.root_path))), 
                                'reports', 'output')
        report_path = os.path.join(reports_dir, report_filename)
        
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except:
            return {'display': 'block'}, {'display': 'none'}, "", "", "", "", "", "", "", None, None, ""
        
        # Extract report data
        is_deepfake = report.get('is_deepfake', False)
        confidence = report.get('confidence', 0)
        media_type = report.get('media_type', 'unknown')
        timestamp = report.get('timestamp', 0)
        details = report.get('details', {})
        
        # Format date
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
        
        # Configure verdict
        verdict_text = "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC"
        verdict_class = "results-verdict verdict-deepfake" if is_deepfake else "results-verdict verdict-authentic"
        
        # Get report title
        title = os.path.splitext(report_filename)[0]
        
        # Create visualization based on media type
        visualization_div = None
        visualization_src = ""
        
        if "visualization" in report:
            # If the report contains a pre-rendered visualization
            visualization_src = report["visualization"]
            visualization_div = html.Img(
                src=visualization_src,
                style={'max-width': '100%', 'max-height': '400px'}
            )
        else:
            # Create a placeholder for visualization
            visualization_div = html.P("Visualization not available for this report.")
        
        # Create details view based on media type
        details_div = html.Div([
            html.H6("Analysis Summary"),
            html.P(f"This {media_type} was analyzed and determined to be {'a deepfake' if is_deepfake else 'authentic'} with {confidence:.2f} confidence.")
        ])
        
        # Add more details based on media type
        if media_type == "image":
            if "faces_detected" in details:
                details_div.children.append(
                    html.P(f"Number of faces detected: {details['faces_detected']}")
                )
                
        elif media_type == "audio":
            if "duration" in details:
                details_div.children.append(
                    html.P(f"Audio duration: {details['duration']:.2f} seconds")
                )
                
            if "temporal_analysis" in details and "inconsistency_index" in details["temporal_analysis"]:
                details_div.children.append(
                    html.P(f"Temporal inconsistency index: {details['temporal_analysis']['inconsistency_index']:.3f}")
                )
                
        elif media_type == "video":
            if "video_info" in details:
                video_info = details["video_info"]
                details_div.children.append(
                    html.P(f"Video duration: {video_info.get('duration', 0):.2f} seconds, Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
                )
                
            if "frames_analyzed" in details:
                details_div.children.append(
                    html.P(f"Frames analyzed: {details['frames_analyzed']}")
                )
                
            if "temporal_inconsistency" in details:
                details_div.children.append(
                    html.P(f"Temporal inconsistency: {details['temporal_inconsistency']:.3f}")
                )
                
            if "av_sync_score" in details:
                details_div.children.append(
                    html.P(f"Audio-video sync score: {details['av_sync_score']:.3f}")
                )
        
        # Show report details
        return (
            {'display': 'none'}, 
            {'display': 'block'}, 
            title, 
            verdict_text, 
            verdict_class,
            date_str,
            media_type.capitalize(),
            f"{confidence:.3f}",
            report_filename,
            visualization_div,
            details_div,
            visualization_src
        )
    
    # Callback for delete report button
    @app.callback(
        Output('delete-report', 'disabled'),
        [Input('delete-report', 'n_clicks')],
        [State('report-filename', 'children')]
    )
    def delete_report(n_clicks, filename):
        if not n_clicks or not filename:
            return False
            
        # Delete the report file
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(app.server.root_path))), 
                                'reports', 'output')
        report_path = os.path.join(reports_dir, filename)
        
        try:
            if os.path.exists(report_path):
                os.remove(report_path)
            return True
        except:
            return False
