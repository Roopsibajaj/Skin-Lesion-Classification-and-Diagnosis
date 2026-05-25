// ===========================================================================
// Deep Learning Based Skin Lesion Classification and Diagnosis
// Frontend JavaScript
// ===========================================================================
// Handles:
//   - Drag-and-drop file upload
//   - Image preview on file select
//   - Form validation (file type, size)
//   - Loading spinner during prediction
//   - Smooth scroll navigation
//   - Animated particle background
// ===========================================================================

document.addEventListener('DOMContentLoaded', function() {

    // =====================================================================
    // Element References
    // =====================================================================
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const fileName = document.getElementById('file-name');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadForm = document.getElementById('upload-form');

    // Allowed file types
    const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB

    // =====================================================================
    // Drag and Drop Handling
    // =====================================================================
    if (dropZone) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop zone when dragging over
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            }, false);
        });

        // Remove highlight when leaving or dropping
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            }, false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            // Set the file input's files property
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    }

    // =====================================================================
    // File Input Change Handler
    // =====================================================================
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                handleFileSelect(this.files[0]);
            }
        });
    }

    // =====================================================================
    // File Selection & Validation
    // =====================================================================
    function handleFileSelect(file) {
        // Validate file type
        if (!ALLOWED_TYPES.includes(file.type)) {
            showNotification('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'error');
            resetUpload();
            return;
        }

        // Validate file size
        if (file.size > MAX_FILE_SIZE) {
            showNotification('File too large. Maximum file size is 16 MB.', 'error');
            resetUpload();
            return;
        }

        // Display image preview
        displayPreview(file);

        // Enable the analyze button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
    }

    // =====================================================================
    // Image Preview Display
    // =====================================================================
    function displayPreview(file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            if (previewImg) {
                previewImg.src = e.target.result;
            }
            if (fileName) {
                fileName.textContent = file.name;
            }
            if (imagePreview) {
                imagePreview.classList.remove('hidden');
            }
            // Hide the drop zone content
            const dropContent = document.querySelector('.drop-zone-content');
            if (dropContent) {
                dropContent.style.display = 'none';
            }
        };

        reader.readAsDataURL(file);
    }

    // =====================================================================
    // Remove Image
    // =====================================================================
    if (removeBtn) {
        removeBtn.addEventListener('click', function() {
            resetUpload();
        });
    }

    function resetUpload() {
        // Reset file input
        if (fileInput) {
            fileInput.value = '';
        }

        // Hide preview
        if (imagePreview) {
            imagePreview.classList.add('hidden');
        }

        // Show drop zone content
        const dropContent = document.querySelector('.drop-zone-content');
        if (dropContent) {
            dropContent.style.display = 'block';
        }

        // Disable analyze button
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
        }
    }

    // =====================================================================
    // Form Submission & Loading Spinner
    // =====================================================================
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Show loading state
            if (analyzeBtn) {
                const btnText = analyzeBtn.querySelector('.btn-text');
                const btnLoading = analyzeBtn.querySelector('.btn-loading');

                if (btnText) btnText.classList.add('hidden');
                if (btnLoading) btnLoading.classList.remove('hidden');

                analyzeBtn.disabled = true;
            }
        });
    }

    // =====================================================================
    // Notification System
    // =====================================================================
    function showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `flash-message flash-${type}`;
        notification.innerHTML = `
            <span class="flash-icon">${type === 'error' ? '⚠️' : '✅'}</span>
            ${message}
            <button class="flash-close" onclick="this.parentElement.remove()">×</button>
        `;

        // Find or create container
        let container = document.querySelector('.flash-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'flash-container';
            const main = document.querySelector('.upload-section') || document.body;
            main.parentNode.insertBefore(container, main);
        }

        container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(20px)';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }

    // =====================================================================
    // Smooth Scroll for Navigation Links
    // =====================================================================
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // =====================================================================
    // Animated Particles Background
    // =====================================================================
    const particlesContainer = document.getElementById('particles');
    if (particlesContainer) {
        createParticles(particlesContainer, 30);
    }

    function createParticles(container, count) {
        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 4 + 1}px;
                height: ${Math.random() * 4 + 1}px;
                background: rgba(0, 212, 170, ${Math.random() * 0.3 + 0.1});
                border-radius: 50%;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
                animation: float ${Math.random() * 6 + 4}s ease-in-out infinite;
                animation-delay: ${Math.random() * 4}s;
                pointer-events: none;
            `;
            container.appendChild(particle);
        }
    }

    // =====================================================================
    // Scroll Animations (Intersection Observer)
    // =====================================================================
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe cards for scroll animation
    document.querySelectorAll('.step-card, .lesion-card, .stat-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // =====================================================================
    // Lesion Types Detailed Descriptions DB
    // =====================================================================
    const LESION_DETAILS_DB = {
        'akiec': {
            desc: "Pre-cancerous skin lesions caused by chronic sun exposure that can progress to squamous cell carcinoma.",
            symptoms: "Rough, scaly, or crusty patches, sand-paper feel, pink/red color.",
            rec: "Pre-cancerous. Treatment recommended (cryotherapy, topical creams) to prevent malignancy."
        },
        'bcc': {
            desc: "The most common type of skin cancer. It is slow-growing and rarely metastasizes but can be locally destructive.",
            symptoms: "Pearly or waxy bump, visible blood vessels (telangiectasia), non-healing sore that bleeds.",
            rec: "Surgical excision or topical treatments recommended. Schedule dermatologist visit."
        },
        'bkl': {
            desc: "Benign skin growths that include seborrheic keratoses, solar lentigines, and lichen planus-like keratoses.",
            symptoms: "Waxy, 'stuck-on' appearance, brown or black color, rough surface.",
            rec: "Benign growth. No treatment required unless irritated or cosmetically desired."
        },
        'df': {
            desc: "A common benign skin nodule, typically occurring on the lower legs, representing a fibrous reaction.",
            symptoms: "Firm, brown-to-pink nodule, 'dimple sign' (dimples inward when pinched).",
            rec: "Benign lesion. No treatment required unless painful or changing."
        },
        'mel': {
            desc: "A highly malignant form of skin cancer that originates in melanocytes (pigment-producing cells).",
            symptoms: "Asymmetrical border, color variations, diameter >6mm, evolving size or shape (ABCDE criteria).",
            rec: "Urgent clinical evaluation and surgical excision/biopsy required."
        },
        'nv': {
            desc: "Common moles, which are benign proliferations of melanocytes.",
            symptoms: "Uniform color, symmetric shape, smooth borders, stable over time.",
            rec: "Benign lesion. Monitor periodically for any changes using the ABCDE rules."
        },
        'vasc': {
            desc: "Benign growths of blood vessels including cherry angiomas, angiokeratomas, and pyogenic granulomas.",
            symptoms: "Bright red, purple, or blue bumps, compressible, may bleed easily.",
            rec: "Benign vascular growth. No medical treatment necessary unless bleeding or cosmetically bothersome."
        }
    };

    // Populate details in HTML & Add Expand Click Event
    document.querySelectorAll('.lesion-card').forEach(card => {
        const abbr = card.getAttribute('data-abbr');
        if (abbr && LESION_DETAILS_DB[abbr]) {
            const data = LESION_DETAILS_DB[abbr];
            const descEl = card.querySelector('.desc-text');
            const symEl = card.querySelector('.sym-text');
            const recEl = card.querySelector('.rec-text');
            if (descEl) descEl.textContent = data.desc;
            if (symEl) symEl.textContent = data.symptoms;
            if (recEl) recEl.textContent = data.rec;
        }

        card.addEventListener('click', function(e) {
            this.classList.toggle('expanded');
            const details = this.querySelector('.lesion-details');
            const indicator = this.querySelector('.expand-indicator');

            if (details) {
                if (this.classList.contains('expanded')) {
                    details.style.maxHeight = details.scrollHeight + "px";
                    details.style.opacity = "1";
                    details.style.marginTop = "15px";
                    if (indicator) indicator.textContent = "▲ Click to collapse";
                } else {
                    details.style.maxHeight = "0";
                    details.style.opacity = "0";
                    details.style.marginTop = "0";
                    if (indicator) indicator.textContent = "▼ Click to expand";
                }
            }
        });
    });

    // =====================================================================
    // Animate Probability Bars on Results Page
    // =====================================================================
    const probBars = document.querySelectorAll('.prob-bar, .confidence-bar');
    if (probBars.length > 0) {
        setTimeout(function() {
            probBars.forEach(function(bar) {
                const width = bar.getAttribute('data-width');
                if (width) {
                    bar.style.width = width + '%';
                }
            });
        }, 300);
    }
});
