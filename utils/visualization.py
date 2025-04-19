import plotly.graph_objects as go
import pandas as pd

def plot_historical_forecast(data, metric):
    """Create historical + forecast visualization"""
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['y'],
        name='Historical Data',
        line=dict(color='#1f77b4')
    ))
    
    # Forecast Data
    forecast_dates = pd.date_range(start=data['date'].iloc[-1], periods=30)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=[data['y'].iloc[-1] * (1.02 ** i) for i in range(30)],  # Placeholder
        name='Forecast',
        line=dict(color='#ff7f0e', dash='dot')
    ))
    
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Forecast",
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    return fig

def plot_forecast_only(data, metric):
    """Create focused forecast visualization"""
    fig = go.Figure()
    
    forecast_dates = pd.date_range(start=data['date'].iloc[-1], periods=30)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=[data['y'].iloc[-1] * (1.02 ** i) for i in range(30)],  # Placeholder
        name='Forecast',
        line=dict(color='#2ca02c')
    ))
    
    fig.update_layout(
        title=f"30-Day {metric.replace('_', ' ').title()} Projection",
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    return fig