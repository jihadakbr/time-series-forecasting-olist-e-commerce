import streamlit as st
from utils import visualization, data_preparation
import warnings
warnings.filterwarnings('ignore')

def dashboard_page():
    st.title("FutureSight Dashboard")

    # get data preparation for orders
    order = data_preparation.data_prep()
    merged_data_ord = order['merged_data']
    merged_forecast_ord = order['merged_forecast']
    merged_future_dates_forecast_ord = order['merged_future_dates_forecast']
    merged_metrics_ord = order['merged_metrics']
    burn_in_ord = order['burn_in']
    split_date_ord = order['split_date']

    # get data preparation for revenue
    revenue = data_preparation.data_prep(metric='revenue_trend')
    merged_data_rev = revenue['merged_data']
    merged_forecast_rev = revenue['merged_forecast']
    merged_future_dates_forecast_rev = revenue['merged_future_dates_forecast']
    merged_metrics_rev = revenue['merged_metrics']
    burn_in_rev = revenue['burn_in']
    split_date_rev = revenue['split_date']

    # KPI Cards    
    rolling_pred_ord = merged_future_dates_forecast_ord['rolling_forecast']['rolling_pred'].iloc[-1]
    rolling_pred_rev = merged_future_dates_forecast_rev['rolling_forecast']['rolling_pred'].iloc[-1]

    actual_order_today = merged_data_ord['actual_data']['y_original'].iloc[-1]
    actual_revenue_today = merged_data_rev['actual_data']['y_original'].iloc[-1]

    delta_ord = rolling_pred_ord - actual_order_today
    delta_rev = rolling_pred_rev - actual_revenue_today

    # Create a 2x2 grid
    col1, col2 = st.columns(2)

    # --- COLUMN 1 ---
    with col1:
        # Row 1: Next Day Orders (Top)
        current_pred_ord = rolling_pred_ord
        st.metric(
            label="📦 Next Day Orders Forecast",
            value=f"{current_pred_ord:,.0f}",
            delta=f"{delta_ord:,.0f} vs Today's Actual Orders"
        )
        
        # Row 2: Order Deviation (Bottom)
        lower_ord = merged_future_dates_forecast_ord['rolling_forecast']['rolling_ci_lower'].iloc[-1]
        upper_ord = merged_future_dates_forecast_ord['rolling_forecast']['rolling_ci_upper'].iloc[-1]
        
        st.metric(
            label="⚠️ Next Day Order Forecast Deviation",
            value=f"{lower_ord:,.0f} — {upper_ord:,.0f}"
        )

    # --- COLUMN 2 ---
    with col2:
        # Row 1: Next Day Revenue (Top)
        current_pred_rev = rolling_pred_rev
        st.metric(
            label="💰 Next Day Revenue Forecast",
            value=f"${current_pred_rev:,.0f}",
            delta=f"${delta_rev:,.0f} vs Today's Actual Revenue"
        )
        
        # Row 2: Revenue Deviation (Bottom)
        lower_rev = merged_future_dates_forecast_rev['rolling_forecast']['rolling_ci_lower'].iloc[-1]
        upper_rev = merged_future_dates_forecast_rev['rolling_forecast']['rolling_ci_upper'].iloc[-1]
        
        st.metric(
            label="⚠️ Next Day Revenue Forecast Deviation",
            value=f"${lower_rev:,.0f} — ${upper_rev:,.0f}"
        )

    # Display full forecast plot for orders
    visualization.create_full_forecast_plot(merged_data_ord, merged_forecast_ord, 
                                            merged_future_dates_forecast_ord,
                                            merged_metrics_ord, burn_in_ord, split_date_ord
                                            )

    # Display full forecast plot for revenue
    visualization.create_full_forecast_plot(merged_data_rev, merged_forecast_rev, 
                                            merged_future_dates_forecast_rev,
                                            merged_metrics_rev, burn_in_rev, split_date_rev,
                                            title="30-Day Revenue Trend Forecast",
                                            legend_name="Revenue", units='R$'
                                            )