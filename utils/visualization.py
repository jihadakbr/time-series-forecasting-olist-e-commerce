import pandas as pd
import streamlit as st
from calendar import month_name
import plotly.graph_objects as go

def create_full_forecast_plot(
    merged_data,
    merged_forecast,
    merged_future_dates_forecast,
    merged_metrics,
    burn_in,
    split_date,
    title="30-Day Order Volume Forecast",
    legend_name="Order",
    units=''
):
    """
    Generate an interactive forecast visualization with confidence intervals and metrics.
    
    Creates a Plotly figure showing historical data, actuals, and multiple forecast types
    (historical, baseline, rolling) with their confidence intervals and performance metrics.
    
    Args:
        merged_data (dict): Contains historical and actual data DataFrames
        merged_forecast (dict): Forecast results DataFrames
        merged_future_dates_forecast (dict): Future predictions DataFrames
        merged_metrics (dict): Performance metrics for each forecast type
        burn_in (int): Initial period to exclude from metrics
        split_date (str/datetime): Date separating historical and forecast data
        title (str): Chart title (default: '30-Day Order Volume Forecast')
        legend_name (str): Name for data series in legend (default: 'Order')
        units (str): Unit label for hover text (default: '')
        
    Returns:
        None: Displays Plotly chart directly via Streamlit
    """

    # Unpacking the variables
    historical_data = merged_data['historical_data']
    actual_data = merged_data['actual_data']

    historical_df = merged_forecast['historical_df']

    baseline_forecast = merged_future_dates_forecast['baseline_forecast']
    rolling_forecast = merged_future_dates_forecast['rolling_forecast']

    historical_metrics = merged_metrics['historical_metrics']
    actual_data_baseline_metrics = merged_metrics['actual_data_baseline_metrics']
    actual_data_rolling_metrics = merged_metrics['actual_data_rolling_metrics']

    # Initialize the Plotly figure
    fig = go.Figure()

    # Historical - Baseline - Confidence Interval
    fig.add_trace(go.Scatter(
        x=historical_data['ds'].iloc[burn_in:].tolist() + historical_data['ds'].iloc[burn_in:][::-1].tolist(),
        y=historical_df['historical_ci_upper'].tolist() + historical_df['historical_ci_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f"95% CI (PIC: {historical_metrics['PIC']}%, PIW: {units}{historical_metrics['PIW']})",
        legendrank=5
    ))

    # Baseline - Confidence Interval
    fig.add_trace(go.Scatter(
        x=baseline_forecast['baseline_dates'].tolist() + baseline_forecast['baseline_dates'][::-1].tolist(),
        y=baseline_forecast['baseline_ci_upper'].tolist() + baseline_forecast['baseline_ci_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(211, 211, 211, 0.7)',  # lightgray
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f"70% CI (PIC: {actual_data_baseline_metrics['PIC']}%, PIW: {units}{actual_data_baseline_metrics['PIW']})",
        legendrank=6
    ))

    # Rolling - Confidence Interval
    fig.add_trace(go.Scatter(
        x=rolling_forecast['rolling_dates'].tolist() + rolling_forecast['rolling_dates'][::-1].tolist(),
        y=rolling_forecast['rolling_ci_upper'].tolist() + rolling_forecast['rolling_ci_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',  # green
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f"95% CI (PIC: {actual_data_rolling_metrics['PIC']}%, PIW: {units}{actual_data_rolling_metrics['PIW']})",
        legendrank=7
    ))

    # Vertical Line - End of Historical Data
    split_date_ts = pd.to_datetime(split_date).value // 10**6
    fig.add_vline(
        x=split_date_ts,
        line=dict(color='brown', dash='dash', width=1),
        opacity=0.5,
        annotation_text="End of Historical Data",
        annotation_position="top left"
    )

    # Vertical Line - Black Friday
    historical_data['ds'] = pd.to_datetime(historical_data['ds'])
    bf_date = pd.to_datetime("2017-11-24")
    
    if (historical_data['ds'].dt.date == bf_date.date()).any():
        bf_date_ts = bf_date.value // 10**6
        fig.add_vline(
            x=bf_date_ts,
            line=dict(color='magenta', dash='dash', width=1),
            opacity=0.5,
            annotation_text="Black Friday",
            annotation_position="top left"
        )

    # Vertical Line - Start of Forecast
    forecast_start_x = pd.to_datetime(actual_data['ds'].iloc[-1]).timestamp() * 1000  # Convert to milliseconds

    fig.add_vline(
        x=forecast_start_x,
        line=dict(color='brown', dash='dash', width=1),
        opacity=0.5
    )

    # Add annotation separately for better control
    fig.add_annotation(
        x=forecast_start_x,
        y=0.5,
        xref="x",
        yref="paper",
        text="Start of Forecast",
        showarrow=False,
        yanchor="bottom",
        xanchor="right"
    )

    # Historical Data
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['y_original'],
        name='Historical Data',
        line=dict(color='black', dash='dot'),
        mode='lines',
        legendrank=0
    ))

    # Actual Data
    fig.add_trace(go.Scatter(
        x=actual_data['ds'],
        y=actual_data['y_original'],
        name=f'Actual {legend_name}',
        line=dict(color='red', dash='dot'),
        mode='lines',
        legendrank=1
    ))

    # Historical - Baseline - Forecast
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y_pred'],
        name=f"Historical Forecast (MDA: {historical_metrics['MDA']}%)",
        line=dict(color='blue'),
        mode='lines',
        legendrank=2
    ))

   # Baseline - Forecast
    fig.add_trace(go.Scatter(
        x=baseline_forecast['baseline_dates'],
        y=baseline_forecast['baseline_pred'],
        name=f"Baseline Forecast (MDA: {actual_data_baseline_metrics['MDA']}%)",
        line=dict(color='dimgray', dash='dot'),
        mode='lines',
        legendrank=3
    ))

    # Rolling - Forecast
    fig.add_trace(go.Scatter(
        x=rolling_forecast['rolling_dates'],
        y=rolling_forecast['rolling_pred'],
        name=f"Rolling Forecast (MDA: {actual_data_baseline_metrics['MDA']}%)",
        line=dict(color='green'),
        mode='lines',
        legendrank=4
    ))

    # Next day's most accurate forecast
    fig.add_trace(go.Scatter(
        x=[rolling_forecast['rolling_dates'].iloc[-1]],
        y=[rolling_forecast['rolling_pred'].iloc[-1]],
        mode='markers',
        marker=dict(
            size=10,
            color='red'
        ),
        name="Next day's most accurate forecast"
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        margin=dict(t=125),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        hovermode="x unified",
        width=1200,
        height=700
    )

    # Apply formatting to all traces
    fig.update_traces(
        hovertemplate=f"{units}"+"%{y:,.0f}"
    )

    # Display the charts on streamlit directly
    st.plotly_chart(fig, use_container_width=True)

def create_forecast_comparison_plot(
    merged_data,
    merged_metrics,
    merged_forecast,
    title="30-Day Order Volume Forecast Comparison",
    legend_name="Order",
    units=''
):  
    """
    Generate a comparison plot between baseline and rolling forecasts with metrics.
    
    Creates an interactive Plotly visualization comparing forecast methods, showing:
    - Actual vs predicted values
    - Confidence intervals for each method
    - Key performance metrics (PIC, PIW, MDA)
    - Forecast starting point marker
    
    Args:
        merged_data (dict): Contains actual data DataFrame
        merged_metrics (dict): Performance metrics for baseline and rolling forecasts
        merged_forecast (dict): Forecast DataFrames for both methods
        title (str): Chart title (default: '30-Day Order Volume Forecast Comparison')
        legend_name (str): Name for data series in legend (default: 'Order')
        units (str): Unit label for hover text (default: '')
        
    Returns:
        None: Displays Plotly chart directly via Streamlit
    """

    # Unpack the variables
    actual_data = merged_data['actual_data']
    actual_data_baseline_metrics = merged_metrics['actual_data_baseline_metrics']
    actual_data_rolling_metrics = merged_metrics['actual_data_rolling_metrics']
    baseline_forecast = merged_forecast['baseline_forecast']
    rolling_forecast = merged_forecast['rolling_forecast']

    # Initialize the Plotly figure.
    fig = go.Figure()

    # Baseline - Confidence Interval
    fig.add_trace(go.Scatter(
        x=baseline_forecast['baseline_dates'].tolist() + baseline_forecast['baseline_dates'][::-1].tolist(),
        y=baseline_forecast['baseline_ci_upper'].tolist() + baseline_forecast['baseline_ci_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(211, 211, 211, 0.7)',  # lightgray
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f"70% CI (PIC: {actual_data_baseline_metrics['PIC']}%, PIW: {units}{actual_data_baseline_metrics['PIW']})",
        legendrank=4
    ))

    # Rolling - Confidence Interval
    fig.add_trace(go.Scatter(
        x=rolling_forecast['rolling_dates'].tolist() + rolling_forecast['rolling_dates'][::-1].tolist(),
        y=rolling_forecast['rolling_ci_upper'].tolist() + rolling_forecast['rolling_ci_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',  # green
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f"95% CI (PIC: {actual_data_rolling_metrics['PIC']}%, PIW: {units}{actual_data_rolling_metrics['PIW']})",
        legendrank=5
    ))

    # Vertical Line - Start of Forecast
    fig.add_vline(
        x=pd.to_datetime(actual_data['ds'].iloc[-1]).timestamp() * 1000,  # Convert to milliseconds
        line=dict(color='brown', dash='dash', width=1),
        opacity=0.5,
        annotation_text="Start of Forecast",
        annotation_position="top left"
    )

    # Actual Data
    fig.add_trace(go.Scatter(
        x=actual_data['ds'],
        y=actual_data['y_original'],
        name=f'Actual {legend_name}',
        line=dict(color='red', dash='dot'),
        mode='lines', 
        legendrank=1
    ))

   # Baseline - Forecast
    fig.add_trace(go.Scatter(
        x=baseline_forecast['baseline_dates'],
        y=baseline_forecast['baseline_pred'],
        name=f"Baseline Forecast (MDA: {actual_data_baseline_metrics['MDA']}%)",
        line=dict(color='dimgray', dash='dash'),
        mode='lines',
        legendrank=2
    ))

    # Rolling - Forecast
    fig.add_trace(go.Scatter(
        x=rolling_forecast['rolling_dates'],
        y=rolling_forecast['rolling_pred'],
        name=f"Rolling Forecast (MDA: {actual_data_baseline_metrics['MDA']}%)",
        line=dict(color='green'),
        mode='lines',
        legendrank=3
    ))

    # Next day's most accurate forecast
    fig.add_trace(go.Scatter(
        x=[rolling_forecast['rolling_dates'].iloc[-1]],
        y=[rolling_forecast['rolling_pred'].iloc[-1]],
        mode='markers',
        marker=dict(
            size=15,
            color='red'
        ),
        name="Next day's most accurate forecast"
    ))

    # Adjusting the layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        margin=dict(t=125),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        hovermode="x unified",
        width=1200,
        height=700
    )

    # Rotate x-axis ticks and add grid
    fig.update_xaxes(tickangle=0, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')

    # Apply formatting to all traces
    fig.update_traces(
        hovertemplate=f"{units}"+"%{y:,.0f}"
    )

    # # Display the charts on streamlit directly
    st.plotly_chart(fig, use_container_width=True)

def create_forecast_performance_plot(
    merged_data,
    merged_forecast,
    year=None, 
    month=None,
    show_trend_line=True,
    show_all_daily_ticks=True,
    units='',
    name='orders',
    plot_title='Orders',
    unique_key_slider='orders',
    ):
    """
    Generate an interactive forecast performance visualization with deviation analysis.
    
    Creates a Plotly figure showing:
    - Daily forecast vs actual deviations
    - Average deviation line
    - Optional trend line
    - Black Friday marker (when applicable)
    - Month/year selection slider
    
    Args:
        merged_data (dict): Contains historical and actual DataFrames with 'ds' and 'y_original' columns
        merged_forecast (dict): Contains forecast DataFrames with prediction columns
        year (int, optional): Specific year to display. Defaults to None (use slider)
        month (int, optional): Specific month to display. Defaults to None (use slider)
        show_trend_line (bool): Whether to show deviation trend line (default: True)
        show_all_daily_ticks (bool): Whether to show all day ticks (default: True)
        units (str): Unit label for values (default: '')
        name (str): Metric name for labels (default: 'orders')
        plot_title (str): Main plot title prefix (default: 'Orders')
        unique_key_slider (str): Unique identifier for Streamlit slider (default: 'orders')
        
    Returns:
        None: Displays Plotly chart directly via Streamlit with interactive month selection
    """

    # Merge Historical and Actual Data
    historical_data = {
        'date': merged_data['historical_data']['ds'],
        'y_original': merged_data['historical_data']['y_original']
    }

    actual_data = {
        'date': merged_data['actual_data']['ds'],
        'y_original': merged_data['actual_data']['y_original']
    }

    historical_data = pd.DataFrame(historical_data)
    actual_data = pd.DataFrame(actual_data)
    
    all_data = pd.concat([historical_data, actual_data], ignore_index=True)

    # Merge Forecast Results for all data
    historical_df = {
        'date': merged_forecast['historical_df']['ds'],
        'pred': merged_forecast['historical_df']['y_pred']
    }

    rolling_pred = {
        'date': merged_forecast['rolling_df']['rolling_dates'],
        'pred': merged_forecast['rolling_df']['rolling_pred']
    }

    historical_df = pd.DataFrame(historical_df)
    rolling_pred = pd.DataFrame(rolling_pred)

    forecast_all_data = pd.concat([historical_df, rolling_pred], ignore_index=True)

    # Merge all_data and forecast_all_data
    merged = pd.merge(
        all_data,
        forecast_all_data,
        on='date',
        how='left' # left join
    )
    
    # Convert date column to datetime
    merged['date'] = pd.to_datetime(merged['date'])
    
    # Extract data
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month
    merged['day'] = merged['date'].dt.day
    
    unique_years = sorted(merged['year'].unique())
    unique_months = sorted(merged['month'].unique())
    
    # Create month-year pairs for the slider
    month_year_pairs = []
    for year in unique_years:
        for month in unique_months:
            if not merged[(merged['year'] == year) & (merged['month'] == month)].empty:
                month_year_pairs.append((year, month))
    
    # Create dropdown for month selection
    if month_year_pairs:
        # Format options for display
        options = [f"{month_name[month]} {year}" for year, month in month_year_pairs]
        
        # Create select box
        selected_month_str = st.selectbox(
            "Select Month:",
            options=options,
            index=len(options)-1,  # Default to most recent month
            key=unique_key_slider
        )
        
        # Extract year and month from selection
        selected_month_name, selected_year = selected_month_str.rsplit(' ', 1)
        selected_year = int(selected_year)
        
        # Convert month name to month number
        month_name_to_num = {name.lower(): num for num, name in enumerate(month_name) if num}
        selected_month = month_name_to_num[selected_month_name.lower()]
        
        # Filter data based on selection
        merged = merged[(merged['year'] == selected_year) & 
                    (merged['month'] == selected_month)].copy()
        
        if merged.empty:
            fig = go.Figure()
            fig.add_annotation(text=f"No data for {selected_month_name} {selected_year}",
                            x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
            return fig
        
    # Calculate deviation
    deviation = merged['pred'] - merged['y_original']
    avg_dev = deviation.mean()
    x_dates = merged['date']
    x_min, x_max = x_dates.min(), x_dates.max()

    # Initialize the Plotly figure
    fig = go.Figure()

    # Average Deviation
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[avg_dev, avg_dev],
        mode='lines',
        line=dict(color='magenta', dash='dash'),
        name=f'Avg. Deviation: {units}{avg_dev:,.0f} {name}/day',
        legendrank=4
    ))

    # Perfect Forecast Line
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[0, 0],
        mode='lines',
        line=dict(color='orange'),
        name=f'Actual {plot_title} (normalized to zero)',
        legendrank=1
    ))

    # Vertical Line - Black Friday
    filtered_dates = pd.to_datetime(merged[['year', 'month', 'day']])
    bf_date = pd.to_datetime(f"{selected_year}-11-24")  # Dynamic year based on selection

    if (filtered_dates == bf_date).any():
        bf_date_ts = bf_date.value // 10**6  # Convert to milliseconds for Plotly
        fig.add_vline(
            x=bf_date_ts,
            line=dict(color='black', dash='dash', width=1),
            opacity=0.5,
            annotation_text="Black Friday",
            annotation_position="top left"
        )

    # Deviation Trend
    if show_trend_line:
        fig.add_trace(go.Scatter(
            x=merged['date'],
            y=deviation,
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash'),
            name='Deviation Trend',
            legendrank=5
        ))

    # Scatter points
    over_mask = deviation > 0
    under_mask = deviation < 0

    # Forecast too high
    fig.add_trace(go.Scatter(
        x=merged.loc[over_mask, 'date'],
        y=deviation[over_mask],
        mode='markers',
        marker=dict(color='green', size=8),
        name='Forecast too high',
        legendrank=2
    ))

    # Forecast too low
    fig.add_trace(go.Scatter(
        x=merged.loc[under_mask, 'date'],
        y=deviation[under_mask],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Forecast too low',
        legendrank=3
    ))

    # Formatting
    title = f"Daily {plot_title}: Forecast vs Actual"
    if month_year_pairs:
        title += f" ({selected_month_name} {selected_year})"
        if show_all_daily_ticks:
            days_in_month = pd.date_range(
                start=f"{selected_year}-{selected_month}-01",
                end=f"{selected_year}-{selected_month}-{pd.Timestamp(year=selected_year, month=selected_month, day=1).days_in_month}",
                freq='D'
            )
            fig.update_xaxes(
                tickvals=days_in_month,
                tickformat="%d",
                tickmode="array"
            )
    else:
        title += " (All Historical Data)"

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        margin=dict(t=125),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        hovermode="x unified",
        width=1200,
        height=500
    )

    # Rotate x-axis ticks and add grid
    fig.update_xaxes(tickangle=0, showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')

    # Apply formatting to all traces
    fig.update_traces(
        hovertemplate=f"{units}"+"%{y:,.0f}"
    )

    # Display the charts on streamlit directly
    st.plotly_chart(fig, use_container_width=True)