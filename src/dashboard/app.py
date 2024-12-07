"""
Web Dashboard for Disaster Alert System
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import sys
from pathlib import Path
import logging

# Add the src directory to the Python path
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from alerts.alert_manager import AlertManager, AlertType, AlertSeverity
from data.collectors import SeismicDataCollector, WeatherDataCollector
from models.predictor import DisasterPredictor

# Initialize components
alert_manager = AlertManager()
seismic_collector = SeismicDataCollector()

# Try to initialize weather collector, but continue without it if API key is missing
try:
    weather_collector = WeatherDataCollector()
    has_weather_data = True
except ValueError as e:
    logging.warning("Weather data collection disabled: %s", e)
    weather_collector = None
    has_weather_data = False

predictor = DisasterPredictor()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Disaster Alert System",
    update_title=None
)

# Styling
CARD_STYLE = {
    "height": "100%",
    "margin-bottom": "20px"
}

MAP_STYLE = {
    "height": "60vh"
}

# Layout components
header = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Disaster Alert System", className="text-white"), width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Refresh", color="light", id="refresh-button", className="me-2"),
                    dbc.Button("Settings", color="light", id="settings-button")
                ])
            ], width=4, className="d-flex justify-content-end")
        ], align="center")
    ]),
    dark=True,
    color="dark",
    className="mb-4"
)

# Alert Panel
alert_card = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Col(html.H4("Active Alerts", className="mb-0"), width=8),
            dbc.Col(
                dbc.Badge(id="alert-count", color="danger", className="ms-2"),
                width=4,
                className="text-end"
            )
        ])
    ]),
    dbc.CardBody([
        html.Div(id="alert-content", style={"max-height": "300px", "overflow-y": "auto"})
    ])
], className="mb-4")

# Map Panel
map_card = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Col(html.H4("Disaster Map", className="mb-0"), width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Earthquakes", id="show-earthquakes", color="primary", size="sm"),
                    dbc.Button("Weather", id="show-weather", color="primary", size="sm"),
                    dbc.Button("Predictions", id="show-predictions", color="primary", size="sm")
                ])
            ], width=4, className="text-end")
        ])
    ]),
    dbc.CardBody([
        dcc.Graph(id="disaster-map", style=MAP_STYLE)
    ])
], className="mb-4")

# Statistics Cards
stats_row = dbc.Row([
    # Earthquake Stats
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H4("Earthquake Statistics", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Recent Events", className="text-center"),
                        html.H2(id="earthquake-count", className="text-center text-primary")
                    ], width=6),
                    dbc.Col([
                        html.H5("Average Magnitude", className="text-center"),
                        html.H2(id="avg-magnitude", className="text-center text-warning")
                    ], width=6)
                ]),
                dcc.Graph(id="earthquake-trend")
            ])
        ], style=CARD_STYLE)
    ], width=6),
    
    # Weather Stats
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(html.H4("Weather Alerts", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Active Warnings", className="text-center"),
                        html.H2(id="weather-alert-count", className="text-center text-danger")
                    ], width=6),
                    dbc.Col([
                        html.H5("Monitored Locations", className="text-center"),
                        html.H2(id="monitored-locations", className="text-center text-info")
                    ], width=6)
                ]),
                dcc.Graph(id="weather-trend")
            ])
        ], style=CARD_STYLE)
    ], width=6)
])

# Prediction Panel
prediction_card = dbc.Card([
    dbc.CardHeader(html.H4("Disaster Predictions", className="mb-0")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="prediction-chart")
            ], width=8),
            dbc.Col([
                html.H5("Risk Assessment"),
                html.Div(id="risk-assessment"),
                html.Hr(),
                html.H5("Recommended Actions"),
                html.Div(id="recommended-actions")
            ], width=4)
        ])
    ])
], className="mb-4")

# Settings Modal
settings_modal = dbc.Modal([
    dbc.ModalHeader("Settings"),
    dbc.ModalBody([
        html.H5("Alert Preferences"),
        dbc.Checklist(
            id="alert-types",
            options=[
                {"label": "Earthquakes", "value": "earthquake"},
                {"label": "Severe Weather", "value": "weather"},
                {"label": "Predictions", "value": "prediction"}
            ],
            value=["earthquake", "weather", "prediction"],
            switch=True
        ),
        html.Hr(),
        html.H5("Notification Settings"),
        dbc.Input(
            id="email-input",
            type="email",
            placeholder="Enter email for notifications"
        ),
        html.Hr(),
        html.H5("Data Update Interval"),
        dcc.Slider(
            id="update-interval",
            min=10,
            max=300,
            step=10,
            value=30,
            marks={i: f"{i}s" for i in range(0, 301, 60)}
        )
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="close-settings", className="ms-auto")
    )
], id="settings-modal")

# Main layout
app.layout = dbc.Container([
    header,
    dbc.Row([
        dbc.Col(alert_card, width=12)
    ]),
    dbc.Row([
        dbc.Col(map_card, width=12)
    ]),
    stats_row,
    dbc.Row([
        dbc.Col(prediction_card, width=12)
    ]),
    settings_modal,
    dcc.Interval(
        id="interval-component",
        interval=30*1000,  # 30 seconds
        n_intervals=0
    ),
    dcc.Store(id="earthquake-data"),
    dcc.Store(id="weather-data"),
    dcc.Store(id="prediction-data")
], fluid=True)

# Callbacks
@app.callback(
    [
        Output("alert-content", "children"),
        Output("alert-count", "children")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_alerts(n):
    """Update active alerts"""
    alerts = alert_manager.get_active_alerts()
    alert_items = []
    
    for alert in alerts:
        color = {
            AlertSeverity.HIGH: "danger",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "info"
        }.get(alert.severity, "secondary")
        
        alert_items.append(
            dbc.ListGroupItem(
                [
                    html.H5(alert.title),
                    html.P(f"Severity: {alert.severity.name}"),
                    html.P(f"Location: {alert.location}"),
                    html.P(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"),
                    html.P(alert.description)
                ],
                color=color
            )
        )
    
    return dbc.ListGroup(alert_items), len(alerts)

@app.callback(
    Output("disaster-map", "figure"),
    [
        Input("interval-component", "n_intervals"),
        Input("show-earthquakes", "n_clicks"),
        Input("show-weather", "n_clicks"),
        Input("show-predictions", "n_clicks")
    ]
)
def update_map(n, show_eq, show_weather, show_pred):
    """Update disaster map"""
    # Create base map
    fig = go.Figure(go.Scattermapbox())
    
    # Add earthquake data if available
    if show_eq:
        earthquake_data = seismic_collector.get_recent_events()
        if earthquake_data:
            df = pd.DataFrame(earthquake_data)
            fig.add_trace(go.Scattermapbox(
                lat=df['latitude'],
                lon=df['longitude'],
                mode='markers',
                marker=dict(
                    size=df['magnitude'] * 5,
                    color='red',
                    opacity=0.7
                ),
                text=df.apply(lambda x: f"Magnitude {x['magnitude']} at {x['time']}", axis=1),
                name='Earthquakes'
            ))
    
    # Add weather data if available
    if show_weather and has_weather_data:
        weather_data = weather_collector.get_current_conditions()
        if weather_data:
            df = pd.DataFrame(weather_data)
            fig.add_trace(go.Scattermapbox(
                lat=df['latitude'],
                lon=df['longitude'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='blue',
                    opacity=0.7
                ),
                text=df['description'],
                name='Weather Alerts'
            ))
    
    # Add prediction data if available
    if show_pred:
        predictions = predictor.get_risk_areas()
        if predictions:
            df = pd.DataFrame(predictions)
            fig.add_trace(go.Scattermapbox(
                lat=df['latitude'],
                lon=df['longitude'],
                mode='markers',
                marker=dict(
                    size=df['risk_level'] * 10,
                    color='yellow',
                    opacity=0.5
                ),
                text=df['description'],
                name='Predicted Risk Areas'
            ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=3,
            center=dict(lat=37.0902, lon=-95.7129)
        ),
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    
    return fig

@app.callback(
    [
        Output("earthquake-count", "children"),
        Output("avg-magnitude", "children"),
        Output("earthquake-trend", "figure")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_earthquake_stats(n):
    """Update earthquake statistics"""
    events = seismic_collector.get_recent_events()
    if not events:
        return "0", "0.0", go.Figure()
    
    df = pd.DataFrame(events)
    count = len(df)
    avg_mag = df['magnitude'].mean()
    
    # Create trend chart
    fig = px.scatter(
        df,
        x='time',
        y='magnitude',
        size='magnitude',
        color='magnitude',
        title='Recent Earthquake Magnitudes'
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return str(count), f"{avg_mag:.1f}", fig

@app.callback(
    [
        Output("weather-alert-count", "children"),
        Output("monitored-locations", "children"),
        Output("weather-trend", "figure")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_weather_stats(n):
    """Update weather statistics"""
    if not has_weather_data:
        return "N/A", "N/A", go.Figure()
    
    weather_data = weather_collector.get_current_conditions()
    if not weather_data:
        return "0", "0", go.Figure()
    
    df = pd.DataFrame(weather_data)
    alert_count = len(df[df['severity'] > 0])
    location_count = len(df)
    
    # Create weather trend chart
    fig = px.scatter(
        df,
        x='time',
        y='temperature',
        size='severity',
        color='severity',
        title='Weather Conditions by Location'
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return str(alert_count), str(location_count), fig

@app.callback(
    [
        Output("prediction-chart", "figure"),
        Output("risk-assessment", "children"),
        Output("recommended-actions", "children")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_predictions(n):
    """Update disaster predictions"""
    predictions = predictor.get_predictions()
    if not predictions:
        return go.Figure(), "No data available", "No recommendations available"
    
    df = pd.DataFrame(predictions)
    
    # Create prediction chart
    fig = px.line(
        df,
        x='time',
        y='risk_level',
        color='disaster_type',
        title='Predicted Risk Levels'
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Generate risk assessment
    risk_items = [
        dbc.ListGroupItem(
            [
                html.H6(f"{row['disaster_type']} Risk"),
                html.P(f"Level: {row['risk_level']}/10"),
                html.P(row['description'])
            ],
            color="danger" if row['risk_level'] > 7 else "warning" if row['risk_level'] > 4 else "info"
        )
        for _, row in df.iterrows()
    ]
    
    # Generate recommendations
    action_items = [
        dbc.ListGroupItem(
            action['description'],
            color=action['priority']
        )
        for action in predictor.get_recommended_actions()
    ]
    
    return fig, dbc.ListGroup(risk_items), dbc.ListGroup(action_items)

# Settings callbacks
@app.callback(
    Output("settings-modal", "is_open"),
    [
        Input("settings-button", "n_clicks"),
        Input("close-settings", "n_clicks")
    ],
    [State("settings-modal", "is_open")]
)
def toggle_settings(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("interval-component", "interval"),
    [Input("update-interval", "value")]
)
def update_interval(value):
    return value * 1000  # Convert to milliseconds

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
