import streamlit as st

def display_page():

    # Custom CSS for larger text and better spacing
    st.markdown("""
    <style>
    .contact-item {
        font-size: 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .contact-emoji {
        font-size: 24px;
        margin-right: 20px;
        min-width: 30px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Contact Information")
    st.markdown("<br>", unsafe_allow_html=True)

    # Contact data with proper Markdown links
    contacts = [
        ("👦", "Jihad Akbar"),
        ("📍", "Indonesia"),
        ("📱", "(+62) 8133 2326 785", "https://wa.me/6281332326785"),
        ("✉️", "jihadakbr@gmail.com", "mailto:jihadakbr@gmail.com"),
        ("🔗", "LinkedIn", "https://www.linkedin.com/in/jihadakbr"),
        ("💻", "GitHub", "https://github.com/jihadakbr")
    ]

    for item in contacts:
        if len(item) == 2:  # For plain text items (name, location)
            st.markdown(f"""
            <div class="contact-item">
                <span class="contact-emoji">{item[0]}</span>
                <span>{item[1]}</span>
            </div>
            """, unsafe_allow_html=True)
        else:  # For clickable links
            st.markdown(f"""
            <div class="contact-item">
                <span class="contact-emoji">{item[0]}</span>
                <span><a href="{item[2]}" target="_blank">{item[1]}</a></span>
            </div>
            """, unsafe_allow_html=True)

    st.write("---")
    st.markdown("#### Feel free to reach out!")