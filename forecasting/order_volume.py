import streamlit as st
from utils import visualization, data_preparation, forecast_support
import warnings
warnings.filterwarnings('ignore')

def show_order_forecast():
    st.title("🛒 Order Volume Forecasting")

    # get data preparation for orders
    result = data_preparation.data_prep()
    merged_data = result['merged_data']
    merged_metrics = result['merged_metrics']
    merged_future_dates_forecast = result['merged_future_dates_forecast']

    # show forecast comparison plot for orders
    visualization.create_forecast_comparison_plot(merged_data, 
                                                  merged_metrics,
                                                  merged_future_dates_forecast)

    # show expected orders volume range table
    forecast_support.create_forecast_table(merged_data, merged_future_dates_forecast)

    # show recommended actions for orders
    forecast_support.display_recommendations()
    
    # show model performance metrics for orders
    forecast_support.display_model_performance(merged_metrics)