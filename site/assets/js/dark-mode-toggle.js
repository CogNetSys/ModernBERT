// dark-mode-toggle.js

function toggleDarkMode() {
    console.log('Toggling dark mode');
    // Get the body element
    var body = document.body;

    // Toggle dark mode class on body
    body.classList.toggle('dark-mode');
}

// Check for saved theme preference and apply it
window.onload = function () {
    console.log('Page loaded');
    var savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
    }
};
