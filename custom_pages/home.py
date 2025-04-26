import streamlit as st

def home_page():
    st.title("FutureSight: Revolutionizing E-Commerce with Data-Driven Forecasting")
    st.markdown("""
    **Welcome to FutureSight!**
    
     ***Disclaimer**: The following story is fictional and created solely to illustrate the challenges addressed by this project.*

    #### 🎬 The Background: A Business on the Brink

    Olist, one of Brazil's largest e-commerce platforms, faces critical challenges despite its growing sales:  
    ⚠️ **Unpredictable Order Volumes**: Sudden fluctuations make it challenging to manage resources effectively.  
    ⚠️ **Revenue Volatility**: Shifting revenue trends are hard to track, hindering strategic planning.  

    The CEO of Olist has issued a bold challenge:
    """)

    _, col, _ = st.columns([1, 2, 1])  # Adjust ratios for spacing
    with col:
        # Display image
        st.image("assets/images/data-scientist-ghibli-illustration.png", caption="Ghibli-style Data Scientist Illustration created by ChatGPT")

    st.markdown("""
    #### 🎯 The Mission: Predict the Future, Optimize the Present
    
    The primary goal of this project is to build a state-of-the-art time series forecasting system that accurately predicts key business metrics, enabling Olist to anticipate future trends and support critical business decisions. This mission includes:  
    📈 **Accurately forecast order volumes** to optimize inventory and resource management, minimizing stockouts and overstocking.  
    📈 **Forecast revenue trends** to provide a clear financial roadmap, supporting strategic planning and long-term growth.  

    This project, **FutureSight**, will deliver an AI-driven forecasting engine that leverages advanced analytics to address these business-critical needs.
    
    *It is powered by Olist's real transaction data, comprising over 100,000 orders across Brazil's diverse regions, collected from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data).* 
    """)