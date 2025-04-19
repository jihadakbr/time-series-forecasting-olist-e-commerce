import streamlit as st
from utils import data_manager, visualization
import pandas as pd
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
import numpy as np
import plotly.graph_objects as go


def display_page():
    st.title("FutureSight Analytics Dashboard")

    ########################################################################################################
    test_kpi_orders = data_manager.fetch_google_sheets_data('order_volume')
    test_kpi_orders['ds'] = pd.to_datetime(test_kpi_orders['ds']).dt.date    

    ########################################################################################################

    test_kpi_revenue = data_manager.fetch_google_sheets_data('revenue_trend')
    test_kpi_revenue['ds'] = pd.to_datetime(test_kpi_revenue['ds']).dt.date
    

    # KPI Cards
    col1, col3 = st.columns(2)
    with col1:
        st.metric("Next Day Orders", f"{test_kpi_orders['y'].iloc[0]:,.0f}")
    with col3:
        st.metric("Next Day Revenue", f"${test_kpi_revenue['y'].iloc[0]:,.0f}")

    # Define path to saved model
    load_model_path = Path('saved_models/order_volume_sarima_v20250418_2116.joblib')
    
    # Load the artifact (model + transformer)
    deploy_artifact = joblib.load(load_model_path)

    # Extract model and transformer
    sarima_results = deploy_artifact['model']
    # pt = deploy_artifact['transformer']
    burn_in = 10
    # print(model, pt)

    test = data_manager.fetch_google_sheets_data('order_volume')
    test['ds'] = pd.to_datetime(test['ds']).dt.date
    # st.dataframe(test)
    
    train_path = Path('saved_csv/order_volume.csv')
    train = pd.read_csv(train_path)
    train = train[['date', 'order_volume']].rename(columns={'date': 'ds', 'order_volume': 'y'})
    train['ds'] = pd.to_datetime(train['ds']).dt.date
    # st.dataframe(train)

    daily_orders = pd.concat([train, test], axis=0).reset_index(drop=True)
    # st.dataframe(daily_orders)

    forecast_days = 30

    train = data_manager.create_features(train)
    test = data_manager.create_features(test)

    # Add features to daily_orders (for full-dataset retraining)
    daily_orders = data_manager.create_features(daily_orders.copy())
    # st.dataframe(daily_orders)

    split_date = daily_orders['ds'].max() - pd.Timedelta(days=forecast_days)

    # Regressor Features
    regressor_features = ['is_black_friday','is_black_friday_peak']

    # Initialize transformer
    pt = PowerTransformer(method='yeo-johnson')

    # Only fit to train set and transform to test and daily_orders (avoiding data leakage)
    train['y_transformed'] = pt.fit_transform(train[['y']]).flatten()
    test['y_transformed'] = pt.transform(test[['y']]).flatten()
    daily_orders['y_transformed'] = pt.transform(daily_orders[['y']]).flatten()

    # Keep original 'y' (unchanged)
    train['y_original'] = train['y']
    test['y_original'] = test['y']
    daily_orders['y_original'] = daily_orders['y']

    # Generate Forecast
    exog_test = test[regressor_features]

    # Baseline Forecast (without rolling window): generate forecast with exogenous variables
    sarima_forecast_baseline = sarima_results.get_forecast(
        steps=forecast_days,
        exog=exog_test
    )

    # Generate a Forecast with a 1-Day Rolling Window
    current_model = sarima_results
    sarima_pred_transformed = []
    sarima_ci_lower_transformed = []
    sarima_ci_upper_transformed = []

    # Iterate through each day in the test set
    for i in range(len(test)):
        # Get exogenous variables for the next step (as a DataFrame row)
        exog_next = exog_test.iloc[i:i+1]

        # Forecast the next step
        forecast = current_model.get_forecast(steps=1, exog=exog_next)
        pred_transformed = forecast.predicted_mean.iloc[0]
        sarima_pred_transformed.append(pred_transformed)

        # --- Capture confidence intervals ---
        ci = forecast.conf_int().iloc[0]
        sarima_ci_lower_transformed.append(ci[0])
        sarima_ci_upper_transformed.append(ci[1])

        # Append actual observation and exog to update the model
        new_endog = test['y_transformed'].iloc[i]
        current_model = current_model.append([new_endog], exog=exog_next.values)

    # Baseline Forecast
    sarima_pred_baseline = pt.inverse_transform(
        sarima_forecast_baseline.predicted_mean.values.reshape(-1, 1)
    ).flatten()

    # Inverse transform confidence intervals for test set
    sarima_baseline_ci = sarima_forecast_baseline.conf_int(alpha=0.30) # 100% - 70% CI → alpha=0.30

    # Inverse transform lower and upper bounds
    sarima_baseline_ci_lower = pt.inverse_transform(
        sarima_baseline_ci.iloc[:, 0].values.reshape(-1, 1)
    ).flatten()

    sarima_baseline_ci_upper = pt.inverse_transform(
        sarima_baseline_ci.iloc[:, 1].values.reshape(-1, 1)
    ).flatten()

    # Convert to array for inverse transform
    sarima_pred_transformed = np.array(sarima_pred_transformed)

    # Forecast with a 1-Day Rolling Window
    sarima_pred = pt.inverse_transform(
        sarima_pred_transformed.reshape(-1, 1)
    ).flatten()

    # Inverse transform confidence intervals for test set
    sarima_ci_lower = pt.inverse_transform(
        np.array(sarima_ci_lower_transformed).reshape(-1, 1)
    ).flatten()
    sarima_ci_upper = pt.inverse_transform(
        np.array(sarima_ci_upper_transformed).reshape(-1, 1)
    ).flatten()

    # Get training predictions AFTER burn-in (skip unstable initial points)
    pred = sarima_results.get_prediction(start=burn_in)
    sarima_train_pred_transformed = pred.predicted_mean

    # Capture training confidence intervals
    train_ci_transformed = pred.conf_int()

    # Align actuals (skip burn-in days)
    actuals_train = train['y_original'].iloc[burn_in:]

    # Inverse transform to original scale
    sarima_train_pred = pt.inverse_transform(
        sarima_train_pred_transformed.values.reshape(-1, 1)
    ).flatten().clip(min=0)

    # Inverse transform training confidence intervals
    sarima_train_ci_lower = pt.inverse_transform(
        train_ci_transformed.iloc[:, 0].values.reshape(-1, 1)
    ).flatten().clip(min=0)
    sarima_train_ci_upper = pt.inverse_transform(
        train_ci_transformed.iloc[:, 1].values.reshape(-1, 1)
    ).flatten().clip(min=0)

    d=1
    # Preparing the train set
    train_dates = train['ds'].iloc[d:]  # d = differencing order

    # Get aligned dates, actuals, and predictions AFTER burn-in
    train_dates = train['ds'].iloc[burn_in:burn_in + len(sarima_train_pred)]
    actuals_train = train['y_original'].iloc[burn_in:burn_in + len(sarima_train_pred)]

    # Align lengths between actuals and predictions
    if len(actuals_train) != len(sarima_train_pred):
        min_length = min(len(actuals_train), len(sarima_train_pred))
        actuals_train = actuals_train.iloc[:min_length]
        sarima_train_pred = sarima_train_pred[:min_length]

    # print(len(train_dates))
    # print(len(actuals_train))
    # print(len(sarima_train_pred))

    # Create DataFrame for train forecasts
    train_forecast_df = pd.DataFrame({
        'ds': train_dates,
        'y_true': actuals_train,
        'y_pred': sarima_train_pred
    })

    #####################################################

    # Create figure
    fig = go.Figure()

    # Add Historical Data trace
    fig.add_trace(go.Scatter(
        x=train['ds'].iloc[burn_in:],
        y=train['y_original'].iloc[burn_in:],
        name='Historical Data',
        line=dict(color='black', dash='dot'),
        mode='lines'
    ))

    # Add test data trace
    fig.add_trace(go.Scatter(
        x=test['ds'],
        y=test['y_original'],
        name='New Data',
        line=dict(color='red', dash='dot'),
        mode='lines'
    ))

    # Add train confidence interval
    fig.add_trace(go.Scatter(
        x=train['ds'].iloc[burn_in:].tolist() + train['ds'].iloc[burn_in:][::-1].tolist(),
        y=sarima_train_ci_upper.tolist() + sarima_train_ci_lower[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f'95% CI'
    ))

    # Add test confidence interval
    fig.add_trace(go.Scatter(
        x=test['ds'].tolist() + test['ds'][::-1].tolist(),
        y=sarima_ci_upper.tolist() + sarima_ci_lower[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f'95% CI'
        # name=f'95% CI (PIC: {pic_test:.1f}%, PIW: {piw_test:.1f})'
    ))

    # Add train forecasts
    fig.add_trace(go.Scatter(
        x=train_forecast_df['ds'],
        y=train_forecast_df['y_pred'],
        # name=f'SARIMA Train Forecast (MDA: {mda_train:.1f}%)',
        name=f'Historical Data Forecast',
        line=dict(color='blue'),
        mode='lines'
    ))

    # Add test forecasts
    fig.add_trace(go.Scatter(
        x=test['ds'],
        y=sarima_pred,
        # name=f'SARIMA Test Forecast (MDA: {mda_test:.1f}%)',
        name=f'New Data Forecast',
        line=dict(color='green'),
        mode='lines'
    ))

    # Convert split_date to timestamp in milliseconds
    split_date_dt = split_date.to_pydatetime()
    split_date_ts = split_date_dt.timestamp() * 1000

    # Add split date line
    fig.add_vline(
        x=split_date_ts,
        line=dict(color='brown', dash='dash', width=1),
        opacity=0.5,
        annotation_text="End of Historical Data",
        annotation_position="top right"
    )

    # Convert Black Friday date to timestamp and check presence
    bf_date = pd.to_datetime('2017-11-24').to_pydatetime()
    # Check if the date exists in the 'ds' column (date part only)
    if (train['ds'].dt.date == bf_date.date()).any():
        bf_date_ts = bf_date.timestamp() * 1000
        fig.add_vline(
            x=bf_date_ts,
            line=dict(color='magenta', dash='dash', width=1),
            opacity=0.5,
            annotation_text="Black Friday",
            annotation_position="top right"
        )

    # Update layout
    fig.update_layout(
    title={
        'text': '30-Day Order Volume Forecast Using a 1-Day Rolling Window',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24}
    },
    yaxis_title='Number of Orders',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    width=1200,
    height=700
    )

    st.plotly_chart(fig, use_container_width=True)
    ################################################################################################################################################

    # Define path to saved model
    load_model_path = Path('saved_models/revenue_trend_sarima_v20250418_2117.joblib')
    
    # Load the artifact (model + transformer)
    deploy_artifact = joblib.load(load_model_path)

    # Extract model and transformer
    sarima_results = deploy_artifact['model']
    # pt = deploy_artifact['transformer']
    burn_in = 10
    # print(model, pt)

    test = data_manager.fetch_google_sheets_data('revenue_trend')
    test['ds'] = pd.to_datetime(test['ds']).dt.date
    # st.dataframe(test)
    
    train_path = Path('saved_csv/revenue_trend.csv')
    train = pd.read_csv(train_path)
    train = train[['date', 'revenue_trend']].rename(columns={'date': 'ds', 'revenue_trend': 'y'})
    train['ds'] = pd.to_datetime(train['ds']).dt.date
    # st.dataframe(train)

    daily_revenue = pd.concat([train, test], axis=0).reset_index(drop=True)
    # st.dataframe(daily_revenue)

    forecast_days = 30

    train = data_manager.create_features(train)
    test = data_manager.create_features(test)

    # Add features to daily_revenue (for full-dataset retraining)
    daily_revenue = data_manager.create_features(daily_revenue.copy())
    # st.dataframe(daily_revenue)

    split_date = daily_revenue['ds'].max() - pd.Timedelta(days=forecast_days)

    # Regressor Features
    regressor_features = ['is_black_friday','is_black_friday_peak']

    # Initialize transformer
    pt = PowerTransformer(method='yeo-johnson')

    # Only fit to train set and transform to test and daily_revenue (avoiding data leakage)
    train['y_transformed'] = pt.fit_transform(train[['y']]).flatten()
    test['y_transformed'] = pt.transform(test[['y']]).flatten()
    daily_revenue['y_transformed'] = pt.transform(daily_revenue[['y']]).flatten()

    # Keep original 'y' (unchanged)
    train['y_original'] = train['y']
    test['y_original'] = test['y']
    daily_revenue['y_original'] = daily_revenue['y']

    # Generate Forecast
    exog_test = test[regressor_features]

    # Baseline Forecast (without rolling window): generate forecast with exogenous variables
    sarima_forecast_baseline = sarima_results.get_forecast(
        steps=forecast_days,
        exog=exog_test
    )

    # Generate a Forecast with a 1-Day Rolling Window
    current_model = sarima_results
    sarima_pred_transformed = []
    sarima_ci_lower_transformed = []
    sarima_ci_upper_transformed = []

    # Iterate through each day in the test set
    for i in range(len(test)):
        # Get exogenous variables for the next step (as a DataFrame row)
        exog_next = exog_test.iloc[i:i+1]

        # Forecast the next step
        forecast = current_model.get_forecast(steps=1, exog=exog_next)
        pred_transformed = forecast.predicted_mean.iloc[0]
        sarima_pred_transformed.append(pred_transformed)

        # --- Capture confidence intervals ---
        ci = forecast.conf_int().iloc[0]
        sarima_ci_lower_transformed.append(ci[0])
        sarima_ci_upper_transformed.append(ci[1])

        # Append actual observation and exog to update the model
        new_endog = test['y_transformed'].iloc[i]
        current_model = current_model.append([new_endog], exog=exog_next.values)

    # Baseline Forecast
    sarima_pred_baseline = pt.inverse_transform(
        sarima_forecast_baseline.predicted_mean.values.reshape(-1, 1)
    ).flatten()

    # Inverse transform confidence intervals for test set
    sarima_baseline_ci = sarima_forecast_baseline.conf_int(alpha=0.30) # 100% - 70% CI → alpha=0.30

    # Inverse transform lower and upper bounds
    sarima_baseline_ci_lower = pt.inverse_transform(
        sarima_baseline_ci.iloc[:, 0].values.reshape(-1, 1)
    ).flatten()

    sarima_baseline_ci_upper = pt.inverse_transform(
        sarima_baseline_ci.iloc[:, 1].values.reshape(-1, 1)
    ).flatten()

    # Convert to array for inverse transform
    sarima_pred_transformed = np.array(sarima_pred_transformed)

    # Forecast with a 1-Day Rolling Window
    sarima_pred = pt.inverse_transform(
        sarima_pred_transformed.reshape(-1, 1)
    ).flatten()

    # Inverse transform confidence intervals for test set
    sarima_ci_lower = pt.inverse_transform(
        np.array(sarima_ci_lower_transformed).reshape(-1, 1)
    ).flatten()
    sarima_ci_upper = pt.inverse_transform(
        np.array(sarima_ci_upper_transformed).reshape(-1, 1)
    ).flatten()

    # Get training predictions AFTER burn-in (skip unstable initial points)
    pred = sarima_results.get_prediction(start=burn_in)
    sarima_train_pred_transformed = pred.predicted_mean

    # Capture training confidence intervals
    train_ci_transformed = pred.conf_int()

    # Align actuals (skip burn-in days)
    actuals_train = train['y_original'].iloc[burn_in:]

    # Inverse transform to original scale
    sarima_train_pred = pt.inverse_transform(
        sarima_train_pred_transformed.values.reshape(-1, 1)
    ).flatten().clip(min=0)

    # Inverse transform training confidence intervals
    sarima_train_ci_lower = pt.inverse_transform(
        train_ci_transformed.iloc[:, 0].values.reshape(-1, 1)
    ).flatten().clip(min=0)
    sarima_train_ci_upper = pt.inverse_transform(
        train_ci_transformed.iloc[:, 1].values.reshape(-1, 1)
    ).flatten().clip(min=0)

    d=1
    # Preparing the train set
    train_dates = train['ds'].iloc[d:]  # d = differencing order

    # Get aligned dates, actuals, and predictions AFTER burn-in
    train_dates = train['ds'].iloc[burn_in:burn_in + len(sarima_train_pred)]
    actuals_train = train['y_original'].iloc[burn_in:burn_in + len(sarima_train_pred)]

    # Align lengths between actuals and predictions
    if len(actuals_train) != len(sarima_train_pred):
        min_length = min(len(actuals_train), len(sarima_train_pred))
        actuals_train = actuals_train.iloc[:min_length]
        sarima_train_pred = sarima_train_pred[:min_length]

    # print(len(train_dates))
    # print(len(actuals_train))
    # print(len(sarima_train_pred))

    # Create DataFrame for train forecasts
    train_forecast_df = pd.DataFrame({
        'ds': train_dates,
        'y_true': actuals_train,
        'y_pred': sarima_train_pred
    })

    #####################################################

    # Create figure
    fig2 = go.Figure()

    # Add Historical Data trace
    fig2.add_trace(go.Scatter(
        x=train['ds'].iloc[burn_in:],
        y=train['y_original'].iloc[burn_in:],
        name='Historical Data',
        line=dict(color='black', dash='dot'),
        mode='lines'
    ))

    # Add test data trace
    fig2.add_trace(go.Scatter(
        x=test['ds'],
        y=test['y_original'],
        name='New Data',
        line=dict(color='red', dash='dot'),
        mode='lines'
    ))

    # Add train confidence interval
    fig2.add_trace(go.Scatter(
        x=train['ds'].iloc[burn_in:].tolist() + train['ds'].iloc[burn_in:][::-1].tolist(),
        y=sarima_train_ci_upper.tolist() + sarima_train_ci_lower[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f'95% CI'
    ))

    # Add test confidence interval
    fig2.add_trace(go.Scatter(
        x=test['ds'].tolist() + test['ds'][::-1].tolist(),
        y=sarima_ci_upper.tolist() + sarima_ci_lower[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f'95% CI'
    ))

    # Add train forecasts
    fig2.add_trace(go.Scatter(
        x=train_forecast_df['ds'],
        y=train_forecast_df['y_pred'],
        # name=f'SARIMA Train Forecast (MDA: {mda_train:.1f}%)',
        name=f'Historical Data Forecast',
        line=dict(color='blue'),
        mode='lines'
    ))

    # Add test forecasts
    fig2.add_trace(go.Scatter(
        x=test['ds'],
        y=sarima_pred,
        # name=f'SARIMA Test Forecast (MDA: {mda_test:.1f}%)',
        name=f'New Data Forecast',
        line=dict(color='green'),
        mode='lines'
    ))

    # Convert split_date to timestamp in milliseconds
    split_date_dt = split_date.to_pydatetime()
    split_date_ts = split_date_dt.timestamp() * 1000

    # Add split date line
    fig2.add_vline(
        x=split_date_ts,
        line=dict(color='brown', dash='dash', width=1),
        opacity=0.5,
        annotation_text="End of Historical Data",
        annotation_position="top right"
    )

    # Convert Black Friday date to timestamp and check presence
    bf_date = pd.to_datetime('2017-11-24').to_pydatetime()
    # Check if the date exists in the 'ds' column (date part only)
    if (train['ds'].dt.date == bf_date.date()).any():
        bf_date_ts = bf_date.timestamp() * 1000
        fig.add_vline(
            x=bf_date_ts,
            line=dict(color='magenta', dash='dash', width=1),
            opacity=0.5,
            annotation_text="Black Friday",
            annotation_position="top right"
        )

    # Update layout
    fig2.update_layout(
    title={
        'text': '30-Day Revenue Forecast Using a 1-Day Rolling Window',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24}
    },
    yaxis_title='Revenue',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    width=1200,
    height=700
    )

    st.plotly_chart(fig2, use_container_width=True)
    ################################################################################################################################################