# FutureSight: Revolutionizing E-Commerce with Data-Driven Forecasting

---

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Project Background](#project-background)
- [Business Objective](#business-objective)
- [Data Understanding](#data-understanding)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Findings and Results](#findings-and-results)
- [Business Impact Analysis](#business-impact-analysis)
- [Recommendations](#recommendations)
- [Dashboard](#dashboard)
- [License](#license)
- [Contact](#contact)

---

## Dataset Overview

### olist_orders_dataset.csv:
| No. | Column Name | Description |
|-----|-------------|-------------|
| 1   | order_id | Unique identifier for each order. |
| 2   | customer_id | Unique identifier for the customer who placed the order. |
| 3   | order_status | Current status of the order (e.g., `delivered`, `shipped`, `canceled`, etc.). |
| 4   | order_purchase_timestamp | Timestamp when the order was placed by the customer. |
| 5   | order_approved_at | Timestamp when payment for the order was approved. |
| 6   | order_delivered_carrier_date | Timestamp when the order was handed to the logistics partner for delivery. |
| 7   | order_delivered_customer_date | Timestamp when the order was delivered to the customer. |
| 8   | order_estimated_delivery_date | Estimated delivery date provided at the time of purchase. |

### olist_order_payments_dataset.csv:
| No. | Column Name | Description |
|-----|-------------|-------------|
| 1   | order_id | Unique identifier for the order. |
| 2   | payment_sequential | Number of payment attempts (e.g., `1` for a single payment, `2` if split into two transactions). |
| 3   | payment_type | Method of payment (e.g., `credit_card`, `boleto`, `voucher`, `debit_card`). |
| 4   | payment_installments | Number of installments the payment was divided into (e.g., `1` for lump sum). |
| 5   | payment_value | Total value of the payment for the order installment. |

---

## Project Background

Unpredictable order volumes and revenue volatility pose challenges for e-commerce businesses. Sudden fluctuations in order demand make it difficult to manage resources effectively, while shifting revenue trends complicate financial planning. **FutureSight** addresses these challenges by providing accurate, data-driven forecasts for order volumes and revenue trends.

---

## Business Objective

The primary goal of this project was to build a state-of-the-art time series forecasting system that accurately predicts key business metrics. This mission includes:
- **Accurately forecast order volumes** to optimize inventory and resource management, minimizing stockouts and overstocking.
- **Forecast revenue trends** to provide a clear financial roadmap, supporting strategic planning and long-term growth. 

This project, **FutureSight**, delivered an AI-driven forecasting engine that leverages advanced analytics to address these business-critical need.

---

## Data Understanding

The dataset was sourced from the [Kaggle – Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data), covering approximately 100,000 orders from October 2016 to September 2018 across Brazil. It includes various order details, customer and seller information, payment methods, shipping logistics, and customer reviews.

---

## Project Overview
![Project Overview Diagram](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/workflow-diagram-time-series-forecasting.png)


The project was deployed using Streamlit and structured as follows:
```
time-series-forecasting-olist-e-commerce/
├── futuresight_app.py    # Main entry point
├── custom_pages/
│   ├── home.py       # Landing page with radio navigation + Project background explanation
│   ├── dashboard.py  # Main forecasting dashboard
│   ├── contact.py    # Contact information
│   ├── overview.py   # Project overview (diagram)
│   └── performance.py # Time series model performance
├── utils/
│   ├── data_manager.py    # Data/Model loading, fetching, etc.
│   ├── data_preparation.py # Data preprocessing for daily orders or revenue
│   ├── forecast_support.py # Table data, recommendations, model performance
│   └── visualization.py   # Plotting utilities
├── forecasting/
│   ├── order_volume.py    # Order volume forecasting logic
│   └── revenue_trend.py   # Revenue trend forecasting logic
├── data/                  # Local CSV datasets
├── saved_models/          # Trained model binaries
├── saved_csv/             # Preprocessed CSV files
└── assets/                # Static files (images, styles)
```

Streamlit link: [time-series-forecasting-olist-e-commerce-jihadakbar.streamlit.app](https://time-series-forecasting-olist-e-commerce-jihadakbar.streamlit.app/)
<br>
<br>
<br>
The project files on GitHub are:
- **Data Science Final Project Presentation.pdf** — PowerPoint presentation in PDF format
- **Order Volume Forecasting.ipynb** — Preprocessing and training of the time series model for daily orders
- **Revenue Trend Forecasting.ipynb** — Preprocessing and training of the time series model for daily revenue

---

## Data Preprocessing

### Step 1: Combine Data
- Merged related CSV files for daily order and revenue forecasting

### Step 2: Filter Relevant Data
- Focused on "delivered" orders (order_status = 'delivered') for forecasting.

### Step 3: Remove Anomalous Dates
- Excluded dates with near-zero orders or revenue, retaining data from January 2017 to August 2018.

### Step 4: Exclude Outlier Dates
- Removed end-of-period dates with unusually low activity, likely due to in-transit orders.

### Step 5: Convert Data Types
- Reformatted 'date' columns to proper date type for time series analysis.

### Step 6: Address Outliers
- Identified Black Friday 2017 as an outlier, handling it as an exogenous variable in the model.

### Step 7: Analyze Seasonality
- Used ACF/PACF analysis to detect weekly and yearly seasonality patterns.

### Step 8: Assess Stationarity
- Data was assessed for stationarity, with a p-value close to 0.5, indicating near non-stationarity.

### Step 9: Save Clean Data
- Exported the clean dataset for use in the forecasting model and the dashboard.

---

## Findings and Results
### Best Model: SARIMA
- The SARIMA model was identified as the most effective for this dataset.
- The model diagnostics showed better performance compared to Prophet, although Prophet's residuals could be improved using SARIMA; however, it would be more difficult to maintain for future use.

![30-Day Order Volume Forecast Using a 1-Day Rolling Window](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Order%20Volume%20Forecast%20using%20a%201-Day%20Rolling%20Window.png)

![30-Day Revenue Forecast Using a 1-Day Rolling Window](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Revenue%20Forecast%20using%20a%201-Day%20Rolling%20Window.png)

### Insights:
- Actual and predicted daily orders and revenue closely aligned, with a notable peak during Black Friday 2017.
- The average RMSE for daily orders was 30, and for daily revenue, R$5,500.
- The prediction intervals for orders and revenue had widths of 126 orders and R$26,000, respectively.
- The prediction coverage was 93% for orders and 97% for revenue, indicating high accuracy.
- Trend prediction accuracy was 70% for daily orders and 67% for daily revenue.
- **Weekly Seasonality:** Orders peaked on weekends, but deliveries occurred during weekdays.
- **Yearly Seasonality:** Black Friday 2017 caused a sharp spike in orders and revenue, followed by a noticeable decline.

![30-Day Order Volume Forecast Comparison](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Order%20Volume%20Forecast%20Comparison.png)

![30-Day Revenue Forecast Comparison](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Revenue%20Forecast%20Comparison.png)

![forecast vs actual daily orders](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/forecast%20vs%20actual%20for%20daily%20orders.png)

![forecast vs actual daily revenue](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/forecast%20vs%20actual%20for%20daily%20revenue.png)

### Forecasting Models:
- **Baseline Forecast:** Provides a broad, less precise prediction over 30 days.
- **1-Day Rolling Window Forecast:** Uses actual data from today to predict tomorrow's values, offering greater accuracy.

---

## Business Impact Analysis

### Daily Order Volume
- **Average Daily Orders:** 255
- **Average Deviation:** 30 orders (11.9% of average)

#### Error Cost Analysis (Local Currency R$) [Using hypothetical numbers]
- **Under-prediction Cost:** R$50 per order (lost profit)
- **Over-prediction Cost:** R$30 per order (excess inventory)
- **Daily Average Cost:** R$958
- **Monthly Error Cost Impact:** R$28,728

#### ROI Calculation [Using hypothetical numbers]
- **Investment cost:** R$40,000 (one-time)
- **Ongoing monthly cost:** R$1,500
- **Projected monthly savings:** R$7,182
- **Annual net gain:** R$68,184
- **ROI after 1 year:** 170.5%
- **Breakeven point:** ~5.6 months

### Daily Revenue
- **Average Daily Revenue:** R$40,787
- **Average Deviation:** R$5,458 (13.4% of average)

#### Error Cost Analysis (Local Currency R$) [Using hypothetical numbers]
- **Under-prediction Cost:** R$50 per R$100 error (lost profit)
- **Over-prediction Cost:** R$30 per R$100 error (excess inventory)
- **Daily Average Cost:** R$1,680
- **Monthly Error Cost Impact:** R$50,415

#### ROI Calculation [Using hypothetical numbers]
- **One-Time Investment:** R$75,000
- **Ongoing Monthly Cost:** R$3,000
- **Projected Monthly Savings:** R$12,604
- **Annual Net Gain:** R$115,245
- **ROI After 1 Year:** 153.7%
- **Breakeven Period:** 6.0 months

---

## Recommendations
- Use daily order forecasts to adjust supply and inventory planning.
- Adjust revenue strategies based on forecast updates to align with actual sales.
- Monitor forecast accuracy regularly to adjust plans and minimize errors.
- Invest in model improvement for significant cost savings and high ROI.

---

## Dashboard
![Dashboard 1](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_1.png)

![Dashboard 2](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_2.png)

![Dashboard 3](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_3.png)

![Dashboard 4](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_4.png)

![Dashboard 5](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_5.png)

![Dashboard 6](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_6.png)

![Dashboard 7](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/dashboard_7.png)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, feel free to reach out:

- Email: [jihadakbr@gmail.com](mailto:jihadakbr@gmail.com)
- LinkedIn: [linkedin.com/in/jihadakbr](https://www.linkedin.com/in/jihadakbr)
- Portfolio: [jihadakbr.github.io](https://jihadakbr.github.io/)