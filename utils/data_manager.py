import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error

@st.cache_resource
def load_model(metric, metric_to_filename):
    """
    Load a cached forecasting model and its transformer for the specified metric.
    
    Args:
        metric (str): The metric to load ('order_volume' or 'revenue_trend')
        
    Returns:
        tuple: (model, transformer) pair loaded from the saved artifact
        
    Raises:
        ValueError: If the metric is unknown or the model fails to load
        FileNotFoundError: If no model file exists for the specified metric
    """
    
    # Get the corresponding filename for the metric
    filename = metric_to_filename.get(metric)
    if filename is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get the model path and load the model and the transformer
    model_path = f'saved_models/{filename}'
    if os.path.exists(model_path):
        try:
            artifact = joblib.load(model_path)
            return artifact['model'], artifact['transformer']
        except Exception as e:
            raise ValueError(f"Error loading {metric} model: {e}")
    else:
        raise FileNotFoundError(f"No model found for {metric} at {model_path}")

@st.cache_data(ttl=86400) # Refresh every 24 hours
def fetch_google_sheets_data(metric):
    """
    Fetch and preprocess time series data from Google Sheets with daily caching.
    
    Retrieves specified metric data from predefined Google Sheets, handles missing dates,
    and performs basic data cleaning. Results are cached for 24 hours.

    Args:
        metric (str): Metric to fetch ('order_volume' or 'revenue_trend')

    Returns:
        pd.DataFrame: Processed DataFrame with columns:
            - ds: Date (datetime.date)
            - y: Metric value (float, with NaN filled as 0)
        Returns empty DataFrame on error.

    Raises:
        Displays Streamlit error messages for:
            - Unknown metrics
            - Failed data fetching
            - Processing errors
    """

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
            
            df['ds'] = df['ds'].dt.date
            # Fill missing values
            df['y'] = df['y'].fillna(0)  # Fill NaN values with 0
        return df
    
    except Exception as e:
        st.error(f"Error fetching Google Sheets data: {e}")
        return pd.DataFrame()

def fetch_historical_data(metric: str) -> pd.DataFrame:
    """
    Load and preprocess historical time series data from local CSV files.
    
    Args:
        metric: The metric to load ('order_volume' or 'revenue_trend').
                Must correspond to an existing CSV file in saved_csv/ directory.

    Returns:
        pd.DataFrame: Processed DataFrame with columns:
            - ds: Date (datetime.date)
            - y: Metric value (float)

    Note:
        The CSV file is expected to have columns named 'date' and '{metric}'.
        Dates are converted to datetime.date objects.
    """
    
    # Read the saved csv files
    historical_data = pd.read_csv(f'saved_csv/{metric}.csv')

    # Rename it for consistency
    historical_data = historical_data[['date', f'{metric}']].rename(columns={'date': 'ds', f'{metric}': 'y'})

    # Convert it to datetyime type
    historical_data['ds'] = pd.to_datetime(historical_data['ds']).dt.date

    return historical_data

def combine_datasets(historical_data, actual_data, forecast_days):
    """
    Combines historical data with actual data, handling cases where actual_data length 
    doesn't match forecast_days.
    
    Parameters:
    - historical_data: DataFrame containing historical data
    - actual_data: DataFrame containing actual data
    - forecast_days: Number of forecast days expected in actual_data
    
    Returns:
    - historical data, actual data, and full data
    """

    # Check if actual_data length doesn't match forecast_days
    if len(actual_data) != forecast_days:
        # Get the last 'forecast_days' from actual_data
        recent_actual = actual_data.tail(forecast_days)
        
        # The remaining data (before the last forecast_days) gets added to historical_data
        remaining_actual = actual_data.iloc[:-forecast_days]
        historical_data = pd.concat([historical_data, remaining_actual], axis=0).reset_index(drop=True)
        
        # Use only the recent actual data for the final concatenation
        actual_data = recent_actual.reset_index(drop=True)
    
    # Combine historical and (processed) actual data
    full_data = pd.concat([historical_data, actual_data], axis=0).reset_index(drop=True)
    
    return historical_data, actual_data, full_data

def create_features(df):
    """
    Create time-based features for forecasting from a datetime DataFrame.
    
    Args:
        df: Input DataFrame containing at least a 'ds' column with datetime values
        
    Returns:
        pd.DataFrame: A copy of the input DataFrame with added features:
            - is_black_friday: Binary indicator for Black Friday dates (2017-11-24/25)
            - is_black_friday_peak: Binary indicator for the 4th Friday of November each year
            
    Processing Steps:
        1. Converts 'ds' to datetime if not already
        2. Creates temporary year column
        3. Identifies Black Friday dates (both specific 2017 dates and annual 4th Fridays)
        4. Drops temporary columns
        5. Converts 'ds' back to date objects
    """

    # Copy the df dataframe
    df = df.copy()

    # Ensure 'ds' column is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Extract the year
    df['year'] = df['ds'].dt.year

    # Creating a feature around black friday events
    df['black_friday'] = df.apply(
        lambda row: pd.date_range(start=f'{row.year}-11-01', end=f'{row.year}-11-30', freq='WOM-4FRI')[0], axis=1
        )
    df['is_black_friday'] = (df['ds'].isin(['2017-11-24','2017-11-25'])).astype(int)
    df['is_black_friday_peak'] = (df['ds'] == df['black_friday']).astype(int)
    
    # Drop the initial black_friday 
    df.drop(columns=['black_friday', 'year'], inplace=True)

    # Only extract the date without the time    
    df['ds'] = df['ds'].dt.date

    return df

def apply_transformer(historical_data, actual_data, full_data_future_dates, transformer, full_transformer, target_col='y'):
    """
    Apply a transformer to target columns across multiple DataFrames while preserving original values.
    
    Args:
        historical_data: DataFrame containing historical time series data
        actual_data: DataFrame containing recent/actual time series data
        full_data_future_dates: DataFrame containing combined time series data
        transformer: Pre-fitted transformer object with transform() method
        full_transformer: Pre-fitted transformer object on full dataset with transform() method
        target_col: Name of the target column to transform (default: 'y')
        
    Returns:
        tuple: Three transformed DataFrames (historical, actual, full) each with:
            - y_transformed: Transformed target values
            - y_original: Original target values (preserved)
            
    Note:
        Modifies input DataFrames in place by adding new columns.
        The transformer should already be fitted before calling this function.
    """
    
    # Transforming the data using the loaded transformer
    historical_data['y_transformed'] = transformer.transform(historical_data[[target_col]]).flatten()
    actual_data['y_transformed'] = transformer.transform(actual_data[[target_col]]).flatten()

    # Transforming the full data using the loaded transformer
    full_data_future_dates['y_transformed'] = full_transformer.transform(full_data_future_dates[[target_col]]).flatten()
    
    # Preserve original values
    for df in [historical_data, actual_data, full_data_future_dates]:
        df['y_original'] = df[target_col]
    
    return historical_data, actual_data, full_data_future_dates

def generate_baseline_forecast(model, actual_data, regressor_features, transformer, forecast_days=30, ci_alpha=0.30):
    """
    Generate baseline forecast with confidence intervals using a trained model.
    
    Args:
        model: Pre-trained forecasting model with get_forecast() method
        actual_data: DataFrame containing recent data with regressor features
        regressor_features: List of column names to use as exogenous variables
        transformer: Fitted transformer for inverse transforming predictions
        forecast_days: Number of days to forecast (default: 30)
        ci_alpha: Significance level for confidence intervals (default: 0.30)
        
    Returns:
        dict: Dictionary containing:
            - baseline_dates: Array of forecast dates
            - baseline_pred: Inverse-transformed point predictions
            - baseline_ci_lower: Inverse-transformed lower confidence bounds
            - baseline_ci_upper: Inverse-transformed upper confidence bounds
            
    Note:
        - The model should support exogenous variables if regressor_features are provided
        - Confidence intervals are generated at 1-ci_alpha level
        - All predictions are inverse-transformed using the provided transformer
    """

    # Generate Forecast
    exog_test = actual_data[regressor_features]

    # Generate forecast with exogenous variables
    forecast_baseline = model.get_forecast(
        steps=forecast_days,
        exog=exog_test
    )
    
    # Get predictions and inverse transform
    baseline_pred = transformer.inverse_transform(
        forecast_baseline.predicted_mean.values.reshape(-1, 1)
    ).flatten()
    
    # Get confidence intervals
    baseline_ci = forecast_baseline.conf_int(alpha=ci_alpha)
    
    # Inverse transform lower and upper bounds
    baseline_ci_lower = transformer.inverse_transform(
        baseline_ci.iloc[:, 0].values.reshape(-1, 1)
    ).flatten()
    
    baseline_ci_upper = transformer.inverse_transform(
        baseline_ci.iloc[:, 1].values.reshape(-1, 1)
    ).flatten()
    
    return {
            'baseline_dates': actual_data['ds'],
            'baseline_pred': baseline_pred, 
            'baseline_ci_lower': baseline_ci_lower, 
            'baseline_ci_upper': baseline_ci_upper
            }

def generate_rolling_forecast(model, actual_data, regressor_features, transformer):
    """
    Generate a rolling one-step-ahead forecast with confidence intervals.
    
    Performs walk-forward validation by:
    1. Making a one-step forecast
    2. Updating the model with the actual observation
    3. Repeating for each time step in the input data

    Args:
        model: Pre-trained forecasting model supporting:
               - get_forecast(steps, exog) for predictions
               - append([values], exog) for model updating
        actual_data: DataFrame containing:
                     - 'ds': datetime column
                     - 'y_transformed': transformed target values
                     - regressor feature columns
        regressor_features: List of column names to use as exogenous variables
        transformer: Fitted transformer with inverse_transform() method

    Returns:
        dict: {
            'rolling_dates': pd.Series of forecast dates,
            'rolling_pred': array of inverse-transformed predictions,
            'rolling_ci_lower': array of inverse-transformed lower CI bounds,
            'rolling_ci_upper': array of inverse-transformed upper CI bounds
        }

    Note:
        - Uses 70% confidence intervals by default (model-dependent)
        - Maintains temporal order during walk-forward validation
        - Preserves original dates while returning transformed predictions
        - Updates model state incrementally with each observation
    """

    # Generate exogeneous features from actual data
    exog_test = actual_data[regressor_features] 
 
    # Get the current model
    current_model = model

    # Initialize an empty list for forecasting, along with its lower and upper bounds
    preds_transformed = []
    ci_lower_transformed, ci_upper_transformed = [], []

    # Loop to forecast the next day using the updated the endogenous feature from today and exogenous feature for tomorrow 
    for i in range(len(actual_data)):
        exog_next = exog_test.iloc[i:i+1]
        forecast = current_model.get_forecast(steps=1, exog=exog_next)
        forecast_pred_mean = forecast.predicted_mean.iloc[0]
        preds_transformed.append(forecast_pred_mean)
        ci = forecast.conf_int().iloc[0]
        ci_lower_transformed.append(ci[0])
        ci_upper_transformed.append(ci[1])
        
        new_endog = actual_data['y_transformed'].iloc[i]
        current_model = current_model.append(
            [new_endog], 
            exog=exog_next.values
        )

    # Inverse the results to their original scale
    preds = transformer.inverse_transform(np.array(preds_transformed).reshape(-1, 1)).flatten()
    ci_lower = transformer.inverse_transform(np.array(ci_lower_transformed).reshape(-1, 1)).flatten()
    ci_upper = transformer.inverse_transform(np.array(ci_upper_transformed).reshape(-1, 1)).flatten()
    
    return {
            'rolling_dates': actual_data['ds'],
            'rolling_pred': preds, 
            'rolling_ci_lower': ci_lower,
            'rolling_ci_upper': ci_upper
            }

def generate_historical_forecast(model, historical_data, transfomer, burn_in):
    """
    Generate historical forecast with confidence intervals after specified burn-in period.
    
    Args:
        model: Pre-trained forecasting model with get_prediction() method
        historical_data: DataFrame containing:
                        - 'ds': datetime column
                        - 'y_original': original target values
        transformer: Fitted transformer with inverse_transform() method
        burn_in: Number of initial observations to exclude from forecast
        
    Returns:
        dict: {
            'ds': pd.Series of dates (after burn-in),
            'y_true': pd.Series of actual values,
            'y_pred': array of predicted values (inverse-transformed and clipped at 0),
            'historical_ci_lower': array of lower confidence bounds,
            'historical_ci_upper': array of upper confidence bounds
        }

    Note:
        - All predictions and confidence intervals are inverse-transformed and clipped at 0
        - Output lengths are aligned to the shortest available series
        - Uses model's default confidence interval level
        - Burn-in period helps avoid model initialization effects
    """

    # Get historical_data predictions after burn-in
    pred = model.get_prediction(start=burn_in)
    pred_transformed = pred.predicted_mean
    
    # Capture historical_data confidence intervals
    historical_ci_transformed = pred.conf_int()
    
    # Inverse transform predictions and confidence intervals
    def inverse_transform_clip(values):
        return transfomer.inverse_transform(
            values.reshape(-1, 1)
        ).flatten().clip(min=0)
    
    # Get the inverse transformation using the function
    historical_pred = inverse_transform_clip(pred_transformed.values)
    historical_ci_lower = inverse_transform_clip(historical_ci_transformed.iloc[:, 0].values)
    historical_ci_upper = inverse_transform_clip(historical_ci_transformed.iloc[:, 1].values)
    
    # Get aligned dates and actuals
    historical_dates = historical_data['ds'].iloc[burn_in:burn_in + len(historical_pred)]
    historical_actuals = historical_data['y_original'].iloc[burn_in:burn_in + len(historical_pred)]
    
    # Ensure equal lengths
    min_length = min(len(historical_actuals), len(historical_pred))
    historical_actuals = historical_actuals.iloc[:min_length]
    historical_pred = historical_pred[:min_length]
    historical_ci_lower = historical_ci_lower[:min_length]
    historical_ci_upper = historical_ci_upper[:min_length]
    historical_dates = historical_dates.iloc[:min_length]
    
    return {
            'ds': historical_dates, 
            'y_true': historical_actuals, 
            'y_pred': historical_pred, 
            'historical_ci_lower': historical_ci_lower, 
            'historical_ci_upper': historical_ci_upper
            }

def generate_baseline_future_dates_forecast(full_model, full_data_future_dates, full_transformer, regressor_features, steps=30, ci_alpha=0.30):
    """
    Generate future forecasts with confidence intervals starting from the last available date.
    
    Args:
        full_model: Pre-trained forecasting model with get_forecast() method
        full_data_future_dates: DataFrame containing full data dates in 'ds' column
        full_transformer: Fitted full_transformer for inverse transforming predictions
        regressor_features: List of exogenous feature columns required by model
        steps: Number of future periods to forecast (default: 30)
        ci_alpha: Significance level for confidence intervals (default: 0.30)
        
    Returns:
        dict: {
            'baseline_future_dates': array of future dates (datetime.date),
            'baseline_future_pred': array of predicted values (clipped at 0),
            'baseline_future_ci_lower': array of lower confidence bounds,
            'baseline_future_ci_upper': array of upper confidence bounds
        }

    Note:
        - Forecast starts from the day after the last date in full_data_future_dates
        - Predictions are inverse-transformed and clipped at minimum 0
        - Automatically generates required features for future dates
        - Confidence intervals are generated at (1-ci_alpha) level
    """

    # Identify the latest full data point to begin forecasting the next day
    last_date = full_data_future_dates['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    # Create DataFrame with future dates
    future_df = pd.DataFrame({'ds': future_dates.date})
    future_df = create_features(future_df)

    # Create the exogeneous features
    exog_future = future_df[regressor_features]

    # Forecast using the next exogenous feature
    forecast = full_model.get_forecast(steps=steps, exog=exog_future)
    forecast_pred_mean = forecast.predicted_mean
    ci = forecast.conf_int(alpha=ci_alpha)

    # Inverse the results to their original scale
    preds = full_transformer.inverse_transform(forecast_pred_mean.values.reshape(-1, 1)).flatten().clip(min=0)
    ci_lower = full_transformer.inverse_transform(ci.iloc[:, 0].values.reshape(-1, 1)).flatten()
    ci_upper = full_transformer.inverse_transform(ci.iloc[:, 1].values.reshape(-1, 1)).flatten()

    return {
            'baseline_future_dates': future_dates.date,
            'baseline_future_pred': preds,
            'baseline_future_ci_lower': ci_lower,
            'baseline_future_ci_upper': ci_upper
            }

def generate_rolling_future_dates_forecast(full_model, full_data_future_dates, full_transformer, regressor_features, steps=1):
    """
    Generate rolling one-step future forecast with confidence intervals.

    Creates a single-day forecast starting from the day after the last available date,
    updating the model with the most recent observation.

    Args:
        full_model: Pre-trained forecasting model with:
                    - get_forecast() for predictions
                    - append() for model updating
        full_data_future_dates: DataFrame containing:
                     - 'ds': datetime column
                     - 'y_transformed': transformed target values
        full_transformer: Fitted full_transformer with inverse_transform() method
        regressor_features: List of exogenous feature columns
        steps: Number of steps to forecast (default: 1)

    Returns:
        dict: {
            'rolling_future_dates': array of future dates (datetime.date),
            'rolling_future_pred': array of inverse-transformed predictions,
            'rolling_future_ci_lower': array of lower confidence bounds,
            'rolling_future_ci_upper': array of upper confidence bounds
        }

    Note:
        - Forecast starts from day after last date in full_data_future_dates
        - Model is updated with the most recent observation
        - Uses model's default confidence interval level
        - Automatically generates required features for future dates
        - Primarily designed for single-step forecasting (steps=1)
    """

    # Identify the latest actual data point to begin forecasting the next day
    last_date = full_data_future_dates['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1)

    # Create DataFrame with future dates
    future_df = pd.DataFrame({'ds': future_dates.date})
    future_df = create_features(future_df)

    # Create the exogeneous features
    exog_future = future_df[regressor_features]

    # Get the current model
    current_model = full_model

    # Initialize empty lists for forecasting, as well as its lower and upper bounds.
    preds_transformed = []
    ci_lower_transformed, ci_upper_transformed = [], []

    # Get the latest data
    latest_data = full_data_future_dates.iloc[-1:]

    # Loop to forecast the next day using the updated the endogenous feature from today and exogenous feature for tomorrow 
    for i in range (len(exog_future)):
        exog_next = exog_future.iloc[i:i+1]
        forecast = full_model.get_forecast(steps=steps, exog=exog_next)
        forecast_pred_mean = forecast.predicted_mean
        preds_transformed.append(forecast_pred_mean)

        ci = forecast.conf_int().iloc[0]
        ci_lower_transformed.append(ci[0])
        ci_upper_transformed.append(ci[1])

        new_endog = latest_data['y_transformed'].iloc[i]
        current_model = current_model.append(
            [new_endog], 
            exog=exog_next.values
        )

    # Inverse the results to their original scale
    preds = full_transformer.inverse_transform(np.array(preds_transformed).reshape(-1, 1)).flatten()
    ci_lower = full_transformer.inverse_transform(np.array(ci_lower_transformed).reshape(-1, 1)).flatten()
    ci_upper = full_transformer.inverse_transform(np.array(ci_upper_transformed).reshape(-1, 1)).flatten()

    return {
            'rolling_future_dates': future_dates.date,
            'rolling_future_pred': preds,
            'rolling_future_ci_lower': ci_lower,
            'rolling_future_ci_upper': ci_upper
            }

def calculate_metrics(true, pred, ci_lower, ci_upper, model_desc, burn_in, is_historical=False):
    """
    Calculate forecast evaluation metrics including accuracy and directional performance.
    
    Args:
        true: Actual observed values
        pred: Predicted values
        ci_lower: Lower confidence interval bounds
        ci_upper: Upper confidence interval bounds
        model_desc: Description of the forecasting model
        burn_in: Initial period to exclude from MDA calculation (historical only)
        is_historical: Whether evaluating historical forecasts (affects MDA baseline)
        
    Returns:
        dict: Dictionary of calculated metrics with formatted strings:
            - Model Description: Model identifier
            - RMSE: Root Mean Squared Error
            - PIW: Prediction Interval Width (average)
            - PIC: Prediction Interval Coverage (% within CI)
            - MDA: Mean Directional Accuracy (% correct direction)
            - Bias: Average forecast error (true - predicted)
            
    Note:
        - RMSE, PIW and Bias are formatted with thousands separators
        - PIC and MDA are percentages without decimal places
        - MDA uses different baseline logic for historical vs actual forecasts
        - All metrics are calculated on the full provided series
    """

    # Initialize an empty dictionary for metrics.
    metrics = {}
    
    # Create model description column
    metrics['Model Description'] = model_desc

    # Calculate RMSE
    metrics['RMSE'] = f"{np.sqrt(mean_squared_error(true, pred)):,.0f}"

    # Calculate PIW PIC 
    within_ci = (true >= ci_lower) & (true <= ci_upper)
    metrics['PIW'] = f"{np.mean(ci_upper - ci_lower):,.0f}"
    metrics['PIC'] = f"{np.mean(within_ci) * 100:.0f}"

    # Calculate MDA
    if is_historical == True:
        prev_actual = true.iloc[burn_in-1]
    else:
        prev_actual = true.iloc[-1]
    
    mda_correct = 0
    for i in range(len(true)):
        current_actual = true.iloc[i]
        current_pred = pred[i]
        actual_dir = current_actual > prev_actual
        forecast_dir = current_pred > prev_actual
        if actual_dir == forecast_dir:
            mda_correct += 1
        prev_actual = current_actual
    metrics['MDA'] = f"{(mda_correct / len(true)) * 100:.0f}"

    # Calculate Forecast Bias / Deviation
    metrics['Bias'] = f"{np.mean(true - pred):,.2f}"

    return metrics

def merge_data(historical_data, actual_data):
    """Combine historical and actual data into a structured dictionary."""

    return {
            'historical_data': historical_data,
            'actual_data': actual_data
            }

def merge_forecast(historical_df, baseline_df, rolling_df):
    """Combine multiple forecast DataFrames into a structured dictionary."""
 
    return {
            'historical_df': historical_df,
            'baseline_df': baseline_df,
            'rolling_df': rolling_df
            }

def merge_future_dates_forecast(baseline_df, rolling_df, baseline_future_dates, rolling_future_dates):
    """Combine evaluation and future forecast data for both baseline and rolling methods."""
    
    # Merge Baseline Evaluation and Baseline Future Dates
    baseline_df = pd.DataFrame(baseline_df)
    baseline_future_dates = pd.DataFrame(baseline_future_dates)
    baseline_future_dates.rename(columns={'baseline_future_dates': 'baseline_dates',
                                          'baseline_future_pred': 'baseline_pred', 
                                          'baseline_future_ci_lower': 'baseline_ci_lower', 
                                          'baseline_future_ci_upper': 'baseline_ci_upper'
                                          }, inplace=True)
    baseline_forecast = pd.concat([baseline_df, baseline_future_dates], ignore_index=True)
    # baseline_forecast consists of: baseline_dates, baseline_pred, baseline_ci_lower, baseline_ci_upper

    # Merge Rolling Evaluation and Rolling Future Dates
    rolling_df = pd.DataFrame(rolling_df)
    rolling_future_dates = pd.DataFrame(rolling_future_dates)
    rolling_future_dates.rename(columns={'rolling_future_dates': 'rolling_dates',
                                         'rolling_future_pred': 'rolling_pred', 
                                         'rolling_future_ci_lower': 'rolling_ci_lower', 
                                         'rolling_future_ci_upper': 'rolling_ci_upper'
                                         }, inplace=True)
    rolling_forecast = pd.concat([rolling_df, rolling_future_dates], ignore_index=True)
    # rolling_forecast consists of: rolling_dates, rolling_pred, rolling_ci_lower, rolling_ci_upper

    return {
            'baseline_forecast': baseline_forecast,
            'rolling_forecast': rolling_forecast
            }

def merge_metrics(historical_metrics, actual_data_baseline_metrics, actual_data_rolling_metrics):
    """Combine metrics from different forecast types into a structured dictionary."""

    return {
            'historical_metrics': historical_metrics,
            'actual_data_baseline_metrics': actual_data_baseline_metrics,
            'actual_data_rolling_metrics': actual_data_rolling_metrics
            }