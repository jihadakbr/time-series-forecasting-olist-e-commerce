import streamlit as st
from utils import data_manager

# This MUST be the very first Streamlit command
st.set_page_config(
    page_title="FutureSight Analytics", 
    page_icon="📈", 
    layout='wide'
)

def main():
    # Load external CSS
    def load_css():
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Load external JS
    def load_js(current_page):
        with open("assets/scripts.js") as f:
            js_code = f.read().replace("{{CURRENT_PAGE}}", current_page)
            st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)

    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    # Navigation options
    nav_options = [
        "🏠 Home",
        "📊 Dashboard",
        "🛒 Order Volume",
        "💰 Revenue Trend",
        "📞 Contact"
    ]

    # Load CSS
    load_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">📈 FutureSight Analytics</div>', 
            unsafe_allow_html=True
        )
        
        for option in nav_options:
            if st.button(option, key=option):
                st.session_state.current_page = option
                st.rerun()
        
        st.markdown(
            '<div class="sidebar-footer">Made with ❤️ by Jihad Akbar</div>', 
            unsafe_allow_html=True
        )

    # Load JS with current page
    load_js(st.session_state.current_page)

    # Page display logic
    if st.session_state.current_page == "🏠 Home":
        from custom_pages.home import display_page
        display_page()

    elif st.session_state.current_page == "📊 Dashboard":
        from custom_pages.dashboard import display_page
        display_page()
    
    elif st.session_state.current_page == "🛒 Order Volume":
        from forecasting.order_volume import show_order_forecast
        show_order_forecast()
        # Example 1: Sequential execution with data passing
        # with st.spinner("Preparing data..."):
        #     df = data_preparation()
        
        # with st.spinner("Processing data..."):
        #     processed_df = data_preprocessing(df)
        
        # OR Example 2: Independent execution
        # data_preparation()
        # data_preprocessing()
    
    elif st.session_state.current_page == "💰 Revenue Trend":
        from forecasting.revenue_trend import show_revenue_forecast
        show_revenue_forecast()
    
    else:
        from custom_pages.contact import display_page
        display_page()

if __name__ == "__main__":
    main()