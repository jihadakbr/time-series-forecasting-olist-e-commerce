import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import os
import joblib

@st.cache_resource
def load_model(metric):
    """Load pre-trained model for specified metric"""
    # Map metrics to file names
    metric_to_filename = {
        'order_volume': 'order_volume_sarima_v20250418_2116.joblib',
        'revenue_trend': 'revenue_trend_sarima_v20250418_2117.joblib'
    }
    
    # Get the corresponding filename for the metric
    filename = metric_to_filename.get(metric)
    if filename is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    model_path = f'saved_models/{filename}'
    if os.path.exists(model_path):
        try:
            artifact = joblib.load(model_path)
            return artifact['model'], artifact['transformer']
        except Exception as e:
            raise ValueError(f"Error loading {metric} model: {e}")
    else:
        raise FileNotFoundError(f"No model found for {metric} at {model_path}")
    
def predict(model, data):
    """Make predictions using loaded model"""
    # SARIMA models use the forecast() or get_forecast() methods
    # Adjust based on your actual model type
    forecast = model.get_forecast(steps=len(data))
    return forecast.predicted_mean

def generate_forecast(model, transformer, history, periods=30):
    """Generate forecast with confidence intervals"""
    # Prepare future dates
    future = pd.date_range(
        start=history['date'].max() + timedelta(days=1),periods=periods
    ).to_frame(name='date')
    
    future = create_features(future)
    regressor_features = ['is_black_friday','is_black_friday_peak']
    exog = future[regressor_features]
    
    # Generate forecast
    forecast = model.get_forecast(steps=periods, exog=exog)

    # Adjust Confidence Interval
    ci = forecast.conf_int(alpha=0.30)  # 100% - 70% CI → alpha=0.30

    # Inverse Transform
    preds = transformer.inverse_transform(
        forecast.predicted_mean.values.reshape(-1, 1)
    ).flatten().clip(min=0)

    sarima_beyond_ci_lower = transformer.inverse_transform(ci.iloc[:, 0].values.reshape(-1, 1)).flatten()
    sarima_beyond_ci_upper = transformer.inverse_transform(ci.iloc[:, 1].values.reshape(-1, 1)).flatten()

    return pd.DataFrame({
        'date': future['date'],
        'yhat': preds,
        'yhat_lower': sarima_beyond_ci_lower,
        'yhat_upper': sarima_beyond_ci_upper
    })

def create_features(df):
    """Feature engineering for time series"""
    df = df.copy()
    # Ensure 'ds' column is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    # Now you can use .dt accessor
    df['year'] = df['ds'].dt.year

    df['black_friday'] = df.apply(
        lambda row: pd.date_range(start=f'{row.year}-11-01', end=f'{row.year}-11-30', freq='WOM-4FRI')[0], axis=1
        )
    df['is_black_friday'] = (df['ds'].isin(['2017-11-24','2017-11-25'])).astype(int)
    df['is_black_friday_peak'] = (df['ds'] == df['black_friday']).astype(int)
    df.drop(columns=['black_friday', 'year'], inplace=True)
    return df

@st.cache_data(ttl=86400) # Refresh every 24 hours
def fetch_google_sheets_data(metric):
    """Fetch data from Google Sheets for specified metric"""
    # Base URL without gid and single parameters
    base_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT4mRpLwHCmzA4-QiAOzFRk7GzeFC6Q6xu1cs4bL21KtzhIGjYofff2t8n2tOs6XYSAc3jdCOJcgpB7/pub"
    
    # Determine gid based on metric
    if metric == 'order_volume':
        gid = '0'
    elif metric == 'revenue_trend':
        gid = '227377561'
    else:
        st.error(f"Unknown metric: {metric}")
        return pd.DataFrame()
    
    # Construct the final URL
    url = f"{base_url}?gid={gid}&single=true&output=csv"
    
    try:
        df = pd.read_csv(url)
        df = df[['date', metric]].rename(columns={'date': 'ds', metric: 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Handle missing dates by creating complete date range
        if not df.empty:
            min_date = df['ds'].min()
            max_date = df['ds'].max()
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            df = df.set_index('ds').reindex(full_date_range).rename_axis('ds').reset_index()
            
            # Fill missing values
            df['y'] = df['y'].fillna(0)  # Fill NaN values with 0
        return df
    
    except Exception as e:
        st.error(f"Error fetching Google Sheets data: {e}")
        return pd.DataFrame()

def get_combined_data(metric):
    """Combine local historical data with latest Google Sheets data"""
    try:
        # Load local data
        local_df = pd.read_csv(f'saved_csv/{metric}.csv')
        local_df['date'] = pd.to_datetime(local_df['date'])
        # Rename the column to match Google Sheets data
        local_df.rename(columns={f'{metric}': 'y'}, inplace=True)
        # Get latest data
        sheets_df = fetch_google_sheets_data(metric)
        # Merge and deduplicate
        combined_df = pd.concat([local_df, sheets_df]).drop_duplicates('date', keep='last')
        return combined_df.sort_values('date').reset_index(drop=True)
    except FileNotFoundError:
        return sheets_df  # Fallback to Google Sheets data

def generate_forecast_table(forecast_df):
    """Generate formatted forecast table with confidence intervals"""
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Forecast', 
                 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'}
    )

def generate_condition_table(metric, forecast_df, selected_date=None):
    """Generate condition/action table based on forecast for selected date"""
    # First validate the input DataFrame
    if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
        return pd.DataFrame({'Status': ['No forecast data available']})
    
    # Determine the correct column names (handling different forecast formats)
    date_col = None
    for col in ['ds', 'date', 'Date']:
        if col in forecast_df.columns:
            date_col = col
            break
    
    if date_col is None:
        return pd.DataFrame({'Status': ['Forecast data missing date column']})
    
    # Determine value column - UPDATED TO INCLUDE 'y'
    value_col = None
    for col in ['yhat', 'forecast', 'Forecast', 'y']:  # Added 'y'
        if col in forecast_df.columns:
            value_col = col
            break
    
    if value_col is None:
        return pd.DataFrame({'Status': ['Forecast data missing value column']})
    
    # Convert dates to datetime if needed
    try:
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
    except:
        return pd.DataFrame({'Status': ['Could not parse dates in forecast data']})
    
    # Set default to latest date if none selected
    if selected_date is None:
        selected_date = forecast_df[date_col].max()
    else:
        try:
            selected_date = pd.to_datetime(selected_date)
        except:
            selected_date = forecast_df[date_col].max()
    
    # Find closest date
    date_diff = (forecast_df[date_col] - selected_date).abs()
    closest_idx = date_diff.idxmin()
    selected_row = forecast_df.iloc[closest_idx]
    
    # Get forecast value with multiple fallbacks
    forecast_value = selected_row.get(
        value_col,
        selected_row.get('Forecast Value', 0)  # Additional fallback
    )
    
    # Generate conditions based on metric
    conditions = []
    if metric == 'order_volume':
        if forecast_value > 200:
            conditions = ["Hire 20% more staff", "Negotiate shipping contracts"]
        elif 100 <= forecast_value <= 200:
            conditions = ["Maintain staffing", "Optimize delivery routes"]
        else:
            conditions = ["Freeze hiring", "Audit retention metrics"]
    elif metric == 'revenue_trend':
        if forecast_value > 50000:
            conditions = ["Open new markets", "Expand sales team"]
        else:
            conditions = ["Optimize pricing", "Boost marketing"]
    else:
        conditions = ["No specific recommendations available"]
    
    # Format the output
    result_date = selected_row[date_col]
    try:
        date_str = result_date.strftime('%Y-%m-%d')
    except:
        date_str = str(result_date)
    
    return pd.DataFrame({
        'Date': [date_str],
        'Forecast Value': [forecast_value],
        'Condition': [f"{metric.replace('_', ' ').title()} Level"],
        'Recommended Actions': ["; ".join(conditions)]
    })
