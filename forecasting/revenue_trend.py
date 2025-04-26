import streamlit as st
from utils import visualization, data_preparation, forecast_support
import warnings
warnings.filterwarnings('ignore')

def show_revenue_forecast():
    st.title("💰 Revenue Trend Forecasting")
    
    # get data preparation for revenue
    result = data_preparation.data_prep(metric='revenue_trend')
    merged_data = result['merged_data']
    merged_metrics = result['merged_metrics']
    merged_future_dates_forecast = result['merged_future_dates_forecast']

    # show forecast comparison plot for revenue
    visualization.create_forecast_comparison_plot(merged_data,
                                                merged_metrics, 
                                                merged_future_dates_forecast,
                                                title="30-Day Revenue Trend (R$) Forecast Comparison",
                                                legend_name="Revenue",
                                                units="R$")

    # show expected revenue volume range table
    forecast_support.create_forecast_table(merged_data, merged_future_dates_forecast, value_col='revenue', value_name='Revenue', units='R\$', col_name=' (R$)')

    # show recommended actions for revenue
    forecast_support.display_recommendations(metric_name="Revenue", value_col="revenue", metric_icon="💰")

    # show model performance metrics for revenue
    forecast_support.display_model_performance(merged_metrics, metric='revenue')