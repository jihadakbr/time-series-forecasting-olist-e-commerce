import pandas as pd
import streamlit as st

def color_alert(value_col, val):
    """
    Apply conditional formatting based on value thresholds for different metrics.
    
    Args:
        value_col: Metric type to evaluate ('orders' or 'revenue')
        val: Value to evaluate (numeric or string)
        
    Returns:
        For numeric values:
            dict: {
                'style': CSS style string for background color,
                'thresh': Formatted threshold description
            }
        For string values:
            str: CSS style string for background color
        Empty string if no conditions match
        
    Color Coding:
        - High values: Light green (#ccffcc)
        - Medium values: Light blue (#90D5FF)
        - Low values: Light red (#ffcccc)
        
    Thresholds of Orders and Revenue: 
        - High
        - Medium
        - Low
            
    Note:
        - Handles both numeric and string inputs
        - Returns different types based on input type
        - Includes formatted threshold information for numeric values
    """

    # Applying specific thresholds for orders or revenue
    if value_col=='orders':
        # Handle numeric values
        if isinstance(val, (int, float)):
            high_thresh = 300
            mod_thresh = [100, 300]
            low_thresh = 100
            if val > high_thresh:
                thresh = f"> {high_thresh}"
                style = 'background-color: #ccffcc'  # Light green
            elif val < low_thresh:
                thresh = f"< {low_thresh}"
                style = 'background-color: #ffcccc'  # Light red
            else:
                thresh = f"{mod_thresh[0]} - {mod_thresh[1]}"
                style = 'background-color: #90D5FF'  # Light blue
            return {'style': style, 'thresh': thresh}
        # Handle string values
        elif isinstance(val, str):
            if val == 'High Demand':
                return 'background-color: #ccffcc'  # Light green
            elif val == 'Low Demand':
                return 'background-color: #ffcccc'  # Light red
            else:
                return 'background-color: #90D5FF'  # Light blue
    elif value_col=='revenue':
        if isinstance(val, (int, float)):
            high_thresh = 50000
            mod_thresh = [30000, 50000]
            low_thresh = 30000
            if val > high_thresh:
                thresh = f"> R${high_thresh:,}"
                style = 'background-color: #ccffcc'  # Light green
            elif val < low_thresh:
                thresh = f"< R${low_thresh:,}"
                style = 'background-color: #ffcccc'  # Light red
            else:
                thresh = f"R${mod_thresh[0]:,} - R${mod_thresh[1]:,}"
                style = 'background-color: #90D5FF'  # Light blue
            return {'style': style, 'thresh': thresh}
        # Handle string values
        elif isinstance(val, str):
            if val == 'High Revenue':
                return 'background-color: #ccffcc'  # Light green
            elif val == 'Low Revenue':
                return 'background-color: #ffcccc'  # Light red
            else:
                return 'background-color: #90D5FF'  # Light blue
            
    # Default case
    return ''

def forecast_table_style(
    forecast_table,
    actual_data,
    value_col, 
    value_name,
    col_name,
    table_name='Rolling'
    ):
    """
    Apply consistent styling and formatting to forecast comparison tables.
    
    Args:
        forecast_table: DataFrame containing forecast data to style
        actual_data: DataFrame containing actual values with 'Date' column
        value_col: Internal metric name (e.g., 'orders', 'revenue')
        value_name: Display name for the metric (e.g., 'Orders')
        col_name: Column name suffix for special cases
        table_name: Forecast type name for labeling (default: 'Rolling')
        
    Returns:
        pd.io.formats.style.Styler: Styled DataFrame with:
            - Formatted numeric values (thousands separators)
            - Special date labeling (Today/Tomorrow)
            - Conditional coloring based on value thresholds
            - Highlighted current and future dates
            
    Styling Features:
        1. Numeric formatting with thousands separators
        2. Special date annotations (Today/Tomorrow)
        3. Value-based background colors via color_alert()
        4. Yellow highlight for today's date
        5. Orange highlight for future dates (tomorrow onward)
        6. Clean handling of missing/empty values
        
    Note:
        - Uses color_alert() for value-based coloring
        - Maintains consistent formatting across forecast types
        - Preserves original data while adding visual enhancements
    """

    # Replace None values with empty strings
    forecast_table = forecast_table.fillna('')

    # Format index to start at 1
    forecast_table = forecast_table.reset_index(drop=True)
    forecast_table.index = forecast_table.index + 1

    # Compute the key dates
    actual_max_date = actual_data['Date'].max()
    tomorrow_date = actual_max_date + pd.Timedelta(days=1)

    # Apply styling with the new logic
    styled_table = (
        forecast_table.style
        .format({
            f'Worst Case{col_name}': lambda x: '{:,.0f}'.format(x) if x != '' else '',
            f'Best Case{col_name}': lambda x: '{:,.0f}'.format(x) if x != '' else '',
            f'Actual {value_name}{col_name}': lambda x: '{:,.0f}'.format(x) if x != '' else '', 
            f'Expected {value_name} ({table_name}){col_name}': lambda x: '{:,.0f}'.format(x) if x != '' else '', 
            'Date': lambda v: (
                f"{v.strftime('%Y-%m-%d')} (Today!)" 
                if v == actual_max_date 
                else (f"{v.strftime('%Y-%m-%d')} (Tomorrow!)" 
                    if v == tomorrow_date 
                    else f"{v.strftime('%Y-%m-%d')}")
                if v != '' else ''
            )
        })
        .applymap(lambda x: color_alert(value_col, x)['style'] 
                if isinstance(color_alert(value_col, x), dict) and x != '' 
                else '', 
                subset=[f'Expected {value_name} ({table_name}){col_name}'])
        # Apply yellow background for today's date
        .apply(lambda s: ['background-color: yellow' if v == actual_max_date else '' for v in s], 
            subset=['Date'])
        # Apply orange background from tomorrow onwards in the 'Date' column
        .apply(lambda s: ['background-color: orange' if v >= tomorrow_date and v != '' else '' for v in s], 
            subset=['Date'])
    )

    return styled_table

def create_forecast_table(
    merged_data, 
    merged_future_dates_forecast,
    value_col='orders', 
    value_name='Orders',
    units='',
    col_name=''
    ):
    """
    Generate and display styled forecast comparison tables with summary statistics.
    
    Creates and displays two interactive tables (rolling and baseline forecasts) with:
    - Actual vs predicted values
    - Confidence intervals (worst/best case scenarios)
    - Automatic formatting and styling
    - Plain-language summary of key metrics

    Args:
        merged_data: Dictionary containing 'actual_data' DataFrame
        merged_future_dates_forecast: Dictionary containing forecast DataFrames
        value_col: Internal name for the metric (default: 'orders')
        value_name: Display name for the metric (default: 'Orders')
        units: Unit label for values (default: '')
        col_name: Column name suffix (default: '')

    Displays:
        - Two styled DataFrames (rolling and baseline forecasts)
        - Summary statistics section with key metrics

    Processing Steps:
        1. Extracts actual and forecast data from input dictionaries
        2. Creates comparison tables for both forecast methods
        3. Applies styling through forecast_table_style()
        4. Displays interactive tables with consistent formatting
        5. Generates plain-language summary of rolling forecast metrics
    """

    # Unpack the variables
    actual_data = merged_data['actual_data']
    baseline_forecast  = merged_future_dates_forecast['baseline_forecast']
    rolling_forecast  = merged_future_dates_forecast['rolling_forecast']

    # Rename it to Date column
    actual_data.rename(columns={'ds': 'Date'}, inplace=True)

    # Expected Volume Range Table
    st.subheader(f"Expected {value_name} Volume Range")
    st.caption(f"Next 30-day projections with possible variation")

    # Create forecast table - Rolling
    forecast_table_rol = pd.DataFrame({
        'Date': baseline_forecast['baseline_dates'],
        f'Actual {value_name}{col_name}': actual_data['y_original'],
        f'Expected {value_name} (Rolling){col_name}': rolling_forecast['rolling_pred'],
        f'Worst Case{col_name}': rolling_forecast['rolling_ci_lower'],
        f'Best Case{col_name}': rolling_forecast['rolling_ci_upper']
    })

    # Create forecast table - Baseline
    forecast_table_bas = pd.DataFrame({
        'Date': baseline_forecast['baseline_dates'],
        f'Actual {value_name}{col_name}': actual_data['y_original'],
        f'Expected {value_name} (Baseline){col_name}': baseline_forecast['baseline_pred'],
        f'Worst Case{col_name}': baseline_forecast['baseline_ci_lower'],
        f'Best Case{col_name}': baseline_forecast['baseline_ci_upper']
    })

    # Styling the table
    styled_table_rol = forecast_table_style(forecast_table_rol, actual_data, value_col, value_name, col_name)
    styled_table_bas = forecast_table_style(forecast_table_bas, actual_data, value_col, value_name, col_name, table_name='Baseline')

    # Display table - Rolling
    st.dataframe(
        styled_table_rol,
        height=250,
        hide_index=True,
        use_container_width=True
    )

    # Display table - Baseline
    st.dataframe(
        styled_table_bas,
        height=250,
        hide_index=True,
        use_container_width=True
    )

    # Plain-language summary
    st.markdown(f"""
    **A Quick Look at Rolling Forecast:**
    - Typical expected volume: **{units}{int(rolling_forecast['rolling_pred'].mean()):,.0f} {value_col}/day**  
    - Normal fluctuation range: **{units}{int(rolling_forecast['rolling_ci_lower'].mean().mean()):,.0f}–{units}{int(rolling_forecast['rolling_ci_upper'].mean()):,.0f} {value_col}/day**  
    """, unsafe_allow_html=True)

def action_table_order():
    """
    Generate an order volume action table with recommended operational responses.
    
    Returns:
        pd.DataFrame: Action table with columns:
            - Order Range: Threshold ranges for order volumes
            - Forecast: Demand level classification
            - Primary Action: Main operational response
            - Secondary Action: Supporting operational response
            
    Thresholds:
        - High Demand
        - Moderate Demand
        - Low Demand
        
    Color Coding:
        Matches color_alert() thresholds for visual consistency.
        
    Note:
        Used to guide staffing and operational decisions based on forecasted order volumes.
    """

    # Initialize the thresholds and their colors
    thresholds = [301, 201, 99]
    order_ranges = [color_alert('orders', t) for t in thresholds]

    # Create the table
    action_table = pd.DataFrame({
        'Order Range': [order_ranges[0]['thresh'], order_ranges[1]['thresh'], order_ranges[2]['thresh']],
        'Forecast': ['High Demand', 'Moderate Demand', 'Low Demand'], 
        'Primary Action': [
            'Hire 20% more staff',
            'Maintain current staffing',
            'Freeze hiring'
        ],
        'Secondary Action': [
            'Negotiate shipping contracts',
            'Optimize delivery routes',
            'Audit retention metrics'
        ]
    })

    return action_table

def action_table_revenue():
    """
    Generate a revenue forecast action table with recommended financial responses.
    
    Returns:
        pd.DataFrame: Action table with columns:
            - Revenue Range: Threshold ranges for revenue levels
            - Forecast: Revenue level classification
            - Primary Action: Main financial response
            - Secondary Action: Supporting financial response
            
    Thresholds:
        - High Revenue
        - Moderate Revenue
        - Low Revenue
        
    Color Coding:
        Matches color_alert() thresholds for visual consistency.
        
    Note:
        Used to guide marketing spend and cost management decisions based on forecasted revenue.
        Integrates with the same color coding system as order forecasts for dashboard consistency.
    """

    # Initialize the thresholds and their colors
    thresholds = [50001, 30001, 20001]
    revenue_ranges = [color_alert('revenue', t) for t in thresholds]

    # Create the table
    action_table = pd.DataFrame({
        'Revenue Range': [revenue_ranges[0]['thresh'], revenue_ranges[1]['thresh'], revenue_ranges[2]['thresh']],
        'Forecast': ['High Revenue', 'Moderate Revenue', 'Low Revenue'], 
        'Primary Action': [
            'Scale marketing campaigns',
            'Maintain current ad spend',
            'Reduce non-essential costs'
        ],
        'Secondary Action': [
            'Upsell premium products/services',
            'Run targeted promotions',
            'Analyze customer churn drivers'
        ]
    })

    return action_table

def display_recommendations(metric_name="Order Volume", value_col="orders", metric_icon="🛒"):
    """
    Display styled action recommendations based on forecast metric type.
    
    Args:
        metric_name: Name of the metric to display ('Order Volume' or 'Revenue Trend')
        value_col: Internal metric identifier ('orders' or 'revenue')
        metric_icon: Emoji icon to display with metric name
        
    Displays:
        - A styled Streamlit DataFrame showing:
            * Threshold ranges
            * Forecast conditions with color coding
            * Priority actions
            * Secondary recommendations
            
    Processing:
        1. Selects appropriate action table based on metric
        2. Applies conditional formatting to Forecast column
        3. Configures display columns with icons and headers
        4. Renders interactive table with full-width container
        
    Note:
        - Automatically switches between order and revenue action tables
        - Uses color_alert() for consistent conditional formatting
        - Column headers are dynamically configured with emojis
        - Table is responsive to container width
    """

    # Title 
    st.subheader("Recommended Actions")
    
    # Call the function based on the metric input
    if metric_name == "Order Volume":
        df_action = action_table_order()
    else:
        df_action = action_table_revenue()
    
    # Apply formatting to Forecast columns
    styled_table = (
        df_action.style
        .applymap(lambda x: color_alert(value_col, x), subset=['Forecast'])
    )

    # Display the table
    st.dataframe(
        styled_table,
        column_config={
            metric_name: st.column_config.TextColumn(f"{metric_icon} {metric_name}", width="auto"),
            'Forecast': st.column_config.TextColumn("📈 Forecast Condition", width="auto"),
            'Primary Action': st.column_config.TextColumn("✅ Priority Action", width="auto"),
            'Secondary Action': st.column_config.TextColumn("⚡ Next Step", width="auto")
        },
        hide_index=True,
        use_container_width=True
    )

def display_model_performance(merged_metrics, metric='orders'):
    """
    Display model evaluation metrics with plain-language interpretations.
    
    Args:
        merged_metrics: Dictionary containing forecast performance metrics
        metric: Type of metric being evaluated ('orders' or 'revenue')
        
    Displays:
        - Streamlit DataFrame showing:
            * Key performance metrics (RMSE, PIW, PIC, MDA, Bias)
            * Formatted metric values
            * Plain-language interpretations
            * Responsive column layouts with icons
            
    Metrics Shown:
        1. RMSE (Root Mean Squared Error) - Prediction accuracy
        2. PIW (Prediction Interval Width) - Forecast range size
        3. PIC (Prediction Interval Coverage) - % within confidence bounds
        4. MDA (Mean Directional Accuracy) - Trend prediction correctness
        5. Bias - Systematic over/under-prediction
        
    Note:
        - Automatically adapts currency formatting for revenue metrics
        - Provides contextual interpretations for each metric
        - Uses icons and responsive column sizing
        - Handles both order volume and revenue trend metrics
    """

    # Title
    st.subheader("Time Series Model Performance")
       
    # Unpack the variables   
    rmse = merged_metrics['actual_data_rolling_metrics']['RMSE']
    piw = merged_metrics['actual_data_rolling_metrics']['PIW']
    pic = merged_metrics['actual_data_rolling_metrics']['PIC']
    mda = merged_metrics['actual_data_rolling_metrics']['MDA']
    bias = merged_metrics['actual_data_rolling_metrics']['Bias']

    # Helper function to clean and convert numeric strings
    def clean_convert(value):
        if isinstance(value, str):
            return float(value.replace(',', ''))
        return float(value)

    # Create metrics table
    metrics_table = {
        'Metrics': ['Avg. Prediction Error (RMSE)', 
                    'Prediction Width (PIW)', 
                    'Prediction Coverage (PIC)', 
                    'Trend Accuracy (MDA)',
                    'Forecast Deviation'],
        'Values': [rmse, 
                   piw, 
                   pic, 
                   mda,
                   f"{clean_convert(bias):,.0f}"]
    }

    # Get the interpretations for orders or revenue
    if metric == 'orders':
        interpretations = [
            f"On average, predictions are off by {rmse} {metric}/day.",
            f"Prediction ranges are {piw} {metric} wide (e.g., 100–{100+clean_convert(piw):,.0f}).",
            f"{pic}% of actual sales fall within the predicted range.",
            f"{mda}% of trend predictions (up/down vs. yesterday) are correct.",
            f"Tends to {'overpredict' if clean_convert(bias) > 0 else 'underpredict'} by {abs(clean_convert(bias)):.0f} {metric}"
        ]
    else:
        interpretations = [
            f"On average, predictions are off by R${clean_convert(rmse):,.0f} {metric}/day.",
            f'Prediction ranges are R${clean_convert(piw):,.0f} {metric} wide (e.g., R$10,000–R${10000+clean_convert(piw):,.0f}).',
            f"{pic}% of actual sales fall within the predicted range.",
            f"{mda}% of trend predictions (up/down vs. yesterday) are correct.",
            f"Tends to {'overpredict' if clean_convert(bias) > 0 else 'underpredict'} by R${abs(clean_convert(bias)):,.0f} {metric}"
        ]

    # Save the interpretation
    metrics_table['Interpretation'] = interpretations
    
    # Display the table
    st.dataframe(
        metrics_table,
        column_config={
            'Metrics': st.column_config.TextColumn("📊 Metric", width="medium"),
            'Values': st.column_config.TextColumn("🎯 Value", width="small"),
            'Interpretation': st.column_config.TextColumn("📝 Interpretation", width="large")
        },
        hide_index=True,
        use_container_width=True
    )