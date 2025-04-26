import pandas as pd
from utils import data_manager

def data_prep(metric='order_volume'):
    """
    Prepare and forecast time series data for the specified metric.
    
    Loads model and data, creates features, generates forecasts (baseline, rolling, historical),
    calculates performance metrics, and returns consolidated results.
    
    Args:
        metric (str): Metric to analyze (default: 'order_volume')
        
    Returns:
        dict: Contains:
            - merged_data: Combined historical and actual data
            - merged_forecast: Consolidated forecast results
            - merged_future_dates_forecast: Future predictions
            - merged_metrics: Performance metrics
            - burn_in: Burn-in period used
            - split_date: Date separating train/test data
    """

    # Model Path
    metric_to_filename_eval = {'order_volume': 'order_volume_sarima_eval_v20250426_0803.joblib',
                               'revenue_trend': 'revenue_trend_sarima_eval_v20250426_0803.joblib'}
    metric_to_filename_full = {'order_volume': 'order_volume_sarima_full_v20250426_0803.joblib',
                               'revenue_trend': 'revenue_trend_sarima_full_v20250426_0803.joblib'}

    # Loading the model
    model, transformer = data_manager.load_model(metric, metric_to_filename_eval)
    full_model, full_transformer = data_manager.load_model(metric, metric_to_filename_full)   
    historical_data = data_manager.fetch_historical_data(metric)
    actual_data = data_manager.fetch_google_sheets_data(metric)
   
    # Set the forecast days
    forecast_days = 30

    # Combine datasets
    historical_data, actual_data, full_data = data_manager.combine_datasets(historical_data, actual_data, forecast_days)

    # Set the burn in periods, and train test split date
    burn_in = 10
    split_date = full_data['ds'].max() - pd.Timedelta(days=forecast_days)

    # Create a new feature
    historical_data = data_manager.create_features(historical_data)
    actual_data = data_manager.create_features(actual_data)
    full_data = data_manager.create_features(full_data)

    # Appyling Yeo-Johnson transformer
    historical_data, actual_data, full_data_future_dates = data_manager.apply_transformer(historical_data, actual_data, full_data, transformer, full_transformer)

    ##### Merge Data
    merged_data = data_manager.merge_data(historical_data, actual_data)

    # Regressor Features
    regressor_features = ['is_black_friday','is_black_friday_peak']

    # Baseline Eval Forecast
    baseline_df = data_manager.generate_baseline_forecast(model, actual_data, regressor_features, transformer)

    # Rolling Eval Forecast
    rolling_df = data_manager.generate_rolling_forecast(model, actual_data, regressor_features, transformer)

    # Historical Eval Forecast
    historical_df = data_manager.generate_historical_forecast(model, historical_data, transformer, burn_in)

    ##### Merge Forecast
    merged_forecast = data_manager.merge_forecast(historical_df, baseline_df, rolling_df)

    # Baseline Future Dates Forecast
    baseline_future_dates = data_manager.generate_baseline_future_dates_forecast(full_model, full_data_future_dates, full_transformer, regressor_features)

    # Rolling Futture Dates Forecast
    rolling_future_dates = data_manager.generate_rolling_future_dates_forecast(full_model, full_data_future_dates, full_transformer, regressor_features)

    ##### Merge Future Dates Forecast
    merged_future_dates_forecast = data_manager.merge_future_dates_forecast(baseline_df, rolling_df, baseline_future_dates, rolling_future_dates)

    # Eval Metrics
    historical_metrics = data_manager.calculate_metrics(historical_df['y_true'], 
                                                        historical_df['y_pred'], 
                                                        historical_df['historical_ci_lower'],
                                                        historical_df['historical_ci_upper'],
                                                        'Historical Data Forecast',
                                                        burn_in,
                                                        is_historical=True)
    
    actual_data_rolling_metrics = data_manager.calculate_metrics(actual_data['y_original'], 
                                                            rolling_df['rolling_pred'], 
                                                            rolling_df['rolling_ci_lower'],
                                                            rolling_df['rolling_ci_upper'],
                                                            'New Data Forecast (1-Day Rolling Window)',
                                                            burn_in)
    
    actual_data_baseline_metrics = data_manager.calculate_metrics(actual_data['y_original'], 
                                                            baseline_df['baseline_pred'], 
                                                            baseline_df['baseline_ci_lower'],
                                                            baseline_df['baseline_ci_upper'],
                                                            'New Data Forecast (Baseline Forecast)',
                                                            burn_in)

    ##### Merge Metrics
    merged_metrics = data_manager.merge_metrics(historical_metrics, actual_data_baseline_metrics, actual_data_rolling_metrics)

    return {
            'merged_data': merged_data, 
            'merged_forecast':merged_forecast, 
            'merged_future_dates_forecast': merged_future_dates_forecast, 
            'merged_metrics': merged_metrics, 
            'burn_in': burn_in, 
            'split_date': split_date
            }