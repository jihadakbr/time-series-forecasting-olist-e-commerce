import streamlit as st

def contact_page():
    st.title("Contact Information")
    st.markdown("<br>", unsafe_allow_html=True)

    contacts = [
        ("👦", "Jihad Akbar"),
        ("📍", "Indonesia"),
        ("📱", "(+62) 8133 2326 785", "https://wa.me/6281332326785"),
        ("✉️", "jihadakbr@gmail.com", "mailto:jihadakbr@gmail.com"),
        ("🔗", "LinkedIn", "https://www.linkedin.com/in/jihadakbr"),
        ("💻", "GitHub", "https://github.com/jihadakbr")
    ]

    # Open links in a new tab
    for item in contacts:
        if len(item) == 2:
            st.markdown(f"""
            <div class="contact-item">
                <span class="contact-emoji">{item[0]}</span>
                <span>{item[1]}</span>
            </div>
            """, unsafe_allow_html=True)
        else: 
            st.markdown(f"""
            <div class="contact-item">
                <span class="contact-emoji">{item[0]}</span>
                <span><a href="{item[2]}" target="_blank">{item[1]}</a></span>
            </div>
            """, unsafe_allow_html=True)

    st.write("---")
    st.markdown("#### Feel free to reach out!")