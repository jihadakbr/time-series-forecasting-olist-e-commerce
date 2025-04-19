function setActiveNav() {
    const buttons = document.querySelectorAll('[data-testid="stSidebar"] button');
    buttons.forEach(button => {
        if (button.textContent === "{{CURRENT_PAGE}}") {
            button.classList.add('nav-active');
        }
    });
}

document.addEventListener('DOMContentLoaded', setActiveNav);