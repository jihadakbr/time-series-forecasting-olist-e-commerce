import streamlit as st
from utils import visualization, data_preparation

def performance_page():
    st.title("🎯 Forecast Performance")

    # get data preparation for orders    
    order = data_preparation.data_prep()
    merged_data_ord = order['merged_data']
    merged_forecast_ord = order['merged_forecast']

    # get data preparation for revenue
    revenue = data_preparation.data_prep(metric='revenue_trend')
    merged_data_rev = revenue['merged_data']
    merged_forecast_rev = revenue['merged_forecast']

    # Display forecast performance plot for orders
    visualization.create_forecast_performance_plot(merged_data_ord, merged_forecast_ord)

    # Display forecast performance plot for revenue
    visualization.create_forecast_performance_plot(merged_data_rev, merged_forecast_rev, 
                                                   name='revenue', plot_title='Revenue', 
                                                   units='R$', unique_key_slider='revenue')