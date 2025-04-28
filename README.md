# FutureSight: Revolutionizing E-Commerce with Data-Driven Forecasting
**FutureSight** is an AI-powered forecasting system designed to tackle unpredictability in e-commerce by predicting order volumes and revenue trends. Built on a Brazilian e-commerce dataset (100,000 orders from 2016–2018), it leverages SARIMA models to deliver actionable insights for inventory optimization and financial planning.

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

---

![30-Day Order Volume Forecast Using a 1-Day Rolling Window](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Order%20Volume%20Forecast%20using%20a%201-Day%20Rolling%20Window.png)

![30-Day Revenue Forecast Using a 1-Day Rolling Window](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Revenue%20Forecast%20using%20a%201-Day%20Rolling%20Window.png)

### Insights:
- The actual and predicted daily orders and revenue exhibit a **similar pattern throughout the year**, with a notable peak around the **Black Friday event (2017-11-24)**.
- **The average prediction error (RMSE)** is **30** for daily orders and **R$5,500** for daily revenue.
- **The prediction interval width (PIW)** ranges approximately **126 orders** (e.g., 100–226) and **R$26,000** for daily revenue (e.g., R$10,000–36,000).
- **The prediction coverage (PIC)** is **93%** for orders and **97%** for revenue, indicating that 93% of actual daily orders and 97% of actual daily revenue fall within their respective predicted ranges.
- **70%** of **trend predictions** or MDA (up/down compared to the previous day) are correct for daily orders, while **67%** of trend predictions are correct for daily revenue.
- **The seasonal ups and downs** in this dataset were caused by two main patterns: weekly seasonality and yearly seasonality.
- **Weekly seasonality** happens because customer behavior changes depending on the day of the week. For example, customers often place orders on Saturdays or Sundays, but since the dataset tracks the "delivered" status, these orders are recorded as deliveries during the weekdays.
- **Yearly seasonality** happens because of major shopping events throughout the year. For instance, the Black Friday event in late November causes a sharp increase in both orders and revenue, followed by a noticeable drop afterward.

---

![30-Day Order Volume Forecast Comparison](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Order%20Volume%20Forecast%20Comparison.png)

![30-Day Revenue Forecast Comparison](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/30-Day%20Revenue%20Forecast%20Comparison.png)

### Insights:
- The baseline forecast provides a rough prediction for orders or revenue over the next 30 days, giving a broad overview of future trends. In contrast, the 1-day rolling window forecast leverages today’s actual data to predict tomorrow’s values, making it significantly more accurate.
- As a result, while the baseline forecast is less precise than the 1-day rolling window forecast, it serves as a useful tool for initial preparation and planning

---

![forecast vs actual daily orders](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/forecast%20vs%20actual%20for%20daily%20orders.png)

![forecast vs actual daily revenue](https://raw.githubusercontent.com/jihadakbr/time-series-forecasting-olist-e-commerce/refs/heads/main/assets/images/forecast%20vs%20actual%20for%20daily%20revenue.png)

### Insights:
- This monitoring dashboard helps identify whether the predictions are too high or too low compared to the actual data (normalized to zero).
- The gray line represents the actual number of orders or revenue, serving as a benchmark for comparison.

---

## Business Impact Analysis

### Daily Order Volume
- **Average Daily Orders:** 255
- **Average Deviation:** 30 orders (11.9% of average)

#### Insights:
- The forecasting model deviates on average by 30 orders per day, which is roughly 12% of the daily volume.
- This level of deviation indicates moderate accuracy but with room for improvement to
reduce costs linked to forecast errors.

#### Error Cost Analysis (Local Currency R$) [Using hypothetical numbers]
- **Under-prediction Cost:** R$50 per order (lost profit)
- **Over-prediction Cost:** R$30 per order (excess inventory)
- **Daily Average Cost:** R$958
- **Monthly Error Cost Impact:** R$28,728

#### Insights:
- The forecasting errors translate into a significant daily cost of nearly R$1,000, accumulating to nearly R$29,000 monthly.
- Since under-prediction costs are higher, errors causing stockouts (lost sales) are more
expensive than excess inventory costs.

#### Return on Investment (ROI) [Using hypothetical numbers]
- **Investment cost:** R$40,000 (one-time)
- **Ongoing monthly cost:** R$1,500
- **Projected monthly savings:** R$7,182
- **Annual net gain:** R$68,184
- **ROI after 1 year:** 170.5%
- **Breakeven point:** ~5.6 months

#### Insights:
- Investing R$40k in improving the forecast model with an ongoing cost of R$1,500/month is highly financially justified.
- Achieving a 25% reduction in forecasting errors would generate substantial savings and lead to a very attractive ROI (>170%) within just over 5 months. This is a strong business case for funding forecasting improvements.

### Daily Revenue
- **Average Daily Revenue:** R$40,787
- **Average Deviation:** R$5,458 (13.4% of average)

#### Insights:
- The forecast error on revenue is about 13.4% of the daily revenue, meaning the model deviates on average by R$5,458 daily. This is a substantial variation given the revenue magnitude and directly impacts profitability and costs.

#### Error Cost Analysis (Local Currency R$) [Using hypothetical numbers]
- **Under-prediction Cost:** R$50 per R$100 error (lost profit)
- **Over-prediction Cost:** R$30 per R$100 error (excess inventory)
- **Daily Average Cost:** R$1,680
- **Monthly Error Cost Impact:** R$50,415

#### Insights:
- The costs associated with forecasting errors (lost profit + inventory holding) sum to over R$1,600 daily, leading to a monthly loss exceeding R$50,000. Under-prediction carries a higher penalty, reflecting lost sales/profit impact.

#### Return on Investment (ROI) [Using hypothetical numbers]
- **One-Time Investment:** R$75,000
- **Ongoing Monthly Cost:** R$3,000
- **Projected Monthly Savings:** R$12,604
- **Annual Net Gain:** R$115,245
- **ROI After 1 Year:** 153.7%
- **Breakeven Period:** 6.0 months

#### Insights:
- Investing R$75k in improving the revenue forecast model—with a higher ongoing
maintenance cost—is justified financially.
- Achieving a 25% error reduction delivers substantial monthly savings and a strong ROI of over 150% within a year, breaking even in just 6 months.

---

## Recommendations
- Use updated daily order forecasts to **adjust supply and inventory plans**, ensuring products are available without overstocking or understocking.
- **Base inventory and revenue strategies** on daily updates to forecasts, ensuring alignment with actual sales and preventing overproduction or lost opportunities.
- **Regularly monitor forecast accuracy** on the dashboard, adjusting orders and inventory levels when predictions are off to minimize excess or shortage costs.
- **Investing in model improvement** is financially advantageous with strong ROI and quick breakeven.

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