let currentSection = 'main';
// DOM elements
const app = document.getElementById('app');
const backBtn = document.getElementById('backBtn');
const pageTitle = document.getElementById('pageTitle');
const pageSubtitle = document.getElementById('pageSubtitle');

// Section titles
const sectionTitles = {
    main: 'Settings',
    profile: 'User Profile',
    profileEdit: 'Edit Profile',
    version: 'Version',
    // theme: 'Dark Theme',
    feedback: 'Send Feedback',
};

// Initialize the app
function init() {
    setupEventListeners();
}

// Setup event listeners
function setupEventListeners() {
    // Menu item clicks
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', () => {
            const section = item.getAttribute('data-section');
            navigateToSection(section);
        });
    });

    // Back button
    backBtn.addEventListener('click', () => {
        navigateToSection('main');
    });

    // Password toggles
    document.querySelectorAll('.password-toggle').forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            const targetId = e.currentTarget.getAttribute('data-target');
            const input = document.getElementById(targetId);
            const eyeOpen = e.currentTarget.querySelector('.eye-open');
            const eyeClosed = e.currentTarget.querySelector('.eye-closed');
            
            if (input.type === 'password') {
                input.type = 'text';
                eyeOpen.classList.add('hidden');
                eyeClosed.classList.remove('hidden');
            } else {
                input.type = 'password';
                eyeOpen.classList.remove('hidden');
                eyeClosed.classList.add('hidden');
            }
        });
    });

    // Profile edit form
    const profileEditForm = document.getElementById('profileEditForm');
    profileEditForm.addEventListener('submit', handleProfileEditSubmit);

}

// Navigation
function navigateToSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(sec => {
        sec.classList.remove('active');
    });

    // Show target section
    const targetSection = document.getElementById(section === 'main' ? 'mainMenu' : `${section}Section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update header
    updateHeader(section);
    
    currentSection = section;
}

function updateHeader(section) {
    if (pageTitle) pageTitle.textContent = sectionTitles[section];

    if (section === 'main') {
        if (backBtn) backBtn.classList.add('hidden');
        if (pageSubtitle) pageSubtitle.classList.remove('hidden');
    } else {
        if (backBtn) backBtn.classList.remove('hidden');
        if (pageSubtitle) pageSubtitle.classList.add('hidden');
    }
}

    // Form submission handler
    document.getElementById('passwordChangeForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const currentPassword = document.getElementById('currentPassword').value;
        const newPassword = document.getElementById('newPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        // Client-side validation
        if (newPassword !== confirmPassword) {
            alertSystem.warning('Warning','New passwords do not match!');
            return;
        }

        // Send AJAX request to change password
        try {
            const response = await fetch('/auth/change_password/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                },
                body: JSON.stringify({
                    currentPassword,
                    newPassword
                })
            });

            const result = await response.json();
            if (response.ok) {
                alertSystem.success('Success','Password changed successfully!');
                document.getElementById('passwordChangeForm').reset();
            } else {
                alertSystem.error('Error', 'An error occurred while changing the password.');
            }
        } catch (error) {
            alertSystem.error('Error','Please try again later.');
        }
    });

function handleProfileEditSubmit(e) {
        e.preventDefault();

        const fullName = document.getElementById('editFullName').value.trim();
        const email = document.getElementById('editEmail').value.trim();
        const phone = document.getElementById('editPhone').value.trim();

        if (!fullName || !email || !phone) {
            alertSystem,warning('Warning','Please fill in all required fields!');
            return;
        }

        // Send data to backend
        fetch('/auth/update-profile/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: JSON.stringify({
                username: fullName,
                email: email,
                phone: phone
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update the profile view with new data
                const profileDetails = document.querySelector('#profileSection .profile-details');
                profileDetails.children[0].querySelector('p').textContent = fullName;
                profileDetails.children[1].querySelector('p').textContent = email;
                profileDetails.children[2].querySelector('p').textContent = phone;


                alertSystem.success('Profile updated successfully!');
                navigateToSection('profile');
            } else {
                alertSystem.error('Error','An error occurred while updating the profile.');
            }
        })
    }

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);