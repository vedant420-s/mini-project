/**
 * ========================================
 * SCRIPT.JS - Frontend JavaScript Logic
 * ========================================
 * 
 * Handles:
 * - File upload and validation
 * - Drag and drop functionality
 * - Image preview
 * - API communication with Flask backend
 * - Result display
 * - UI interaction
 * 
 * ========================================
 */

// ============================================
// GLOBAL VARIABLES
// ============================================

let selectedFile = null;  // Store selected image file

// ============================================
// DOM ELEMENTS
// ============================================

const imageInput = document.getElementById('imageInput');
const uploadForm = document.getElementById('uploadForm');
const dragDropZone = document.getElementById('dragDropZone');
const analyzeButton = document.getElementById('analyzeButton');
const resetButton = document.getElementById('resetButton');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const predictionSection = document.getElementById('predictionSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultContainer = document.getElementById('resultContainer');
const normalResult = document.getElementById('normalResult');
const pneumoniaResult = document.getElementById('pneumoniaResult');
const errorResult = document.getElementById('errorResult');

// ============================================
// EVENT LISTENERS
// ============================================

/**
 * File input change event
 * Triggered when user selects a file through the file dialog
 */
imageInput.addEventListener('change', function(event) {
    handleFileSelect(event.target.files[0]);
});

/**
 * Drag over event - shows visual feedback
 */
document.addEventListener('dragover', function(event) {
    event.preventDefault();
    dragDropZone.classList.add('dragover');
});

/**
 * Drag leave event - removes visual feedback
 */
document.addEventListener('dragleave', function(event) {
    dragDropZone.classList.remove('dragover');
});

/**
 * Drop event - handles dropped file
 */
document.addEventListener('drop', function(event) {
    event.preventDefault();
    dragDropZone.classList.remove('dragover');
    
    if (event.dataTransfer.files.length > 0) {
        handleFileSelect(event.dataTransfer.files[0]);
    }
});

/**
 * Analyze button click event
 * Sends image to Flask backend for prediction
 */
analyzeButton.addEventListener('click', function() {
    analyzeImage();
});

/**
 * Reset button click event
 * Clears selection and returns to initial state
 */
resetButton.addEventListener('click', function() {
    resetUI();
});

// ============================================
// FILE HANDLING FUNCTIONS
// ============================================

/**
 * Handle file selection
 * Validates file type and size
 * 
 * @param {File} file - Selected file object
 */
function handleFileSelect(file) {
    if (!file) {
        showError('No file selected');
        return;
    }

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload JPG, PNG, GIF, or BMP image.');
        return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;  // 16MB in bytes
    if (file.size > maxSize) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }

    // Store selected file
    selectedFile = file;
    console.log('✓ File selected:', file.name, file.size, 'bytes');

    // Show drag drop zone
    dragDropZone.classList.add('show');

    // Display preview
    showPreview(file);

    // Show analyze button
    analyzeButton.style.display = 'inline-block';
    resetButton.style.display = 'inline-block';
}

/**
 * Display image preview
 * Reads file and displays in preview section
 * 
 * @param {File} file - Image file
 */
function showPreview(file) {
    const reader = new FileReader();

    /**
     * When file is loaded, display it
     */
    reader.onload = function(e) {
        // Set preview image source
        previewImage.src = e.target.result;

        // Display file information
        const fileSizeKB = (file.size / 1024).toFixed(2);
        fileName.textContent = `File: ${file.name} (${fileSizeKB} KB)`;

        // Show preview section
        previewSection.style.display = 'block';

        // Scroll to preview
        previewSection.scrollIntoView({ behavior: 'smooth' });
    };

    // Read file as data URL
    reader.readAsDataURL(file);
}

// ============================================
// IMAGE ANALYSIS FUNCTIONS
// ============================================

/**
 * Send image to Flask backend for analysis
 * Makes POST request with FormData
 */
function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    console.log('🔬 Starting image analysis...');

    // Show loading spinner
    loadingSpinner.style.display = 'block';
    resultContainer.style.display = 'none';
    predictionSection.style.display = 'block';

    // Scroll to prediction section
    predictionSection.scrollIntoView({ behavior: 'smooth' });

    // Create FormData object
    const formData = new FormData();
    formData.append('file', selectedFile);

    /**
     * Send request to Flask backend
     * POST /predict endpoint
     */
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('✓ Response received:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('✓ Data parsed:', data);
        
        // Hide loading spinner
        loadingSpinner.style.display = 'none';
        resultContainer.style.display = 'block';

        if (data.success) {
            // Display prediction result
            displayResult(data);
        } else {
            // Display error
            showError(data.error || 'An error occurred during analysis');
        }
    })
    .catch(error => {
        console.error('❌ Error:', error);
        
        // Hide loading
        loadingSpinner.style.display = 'none';
        resultContainer.style.display = 'block';

        // Show error message
        showError('Network error or server unavailable. Please try again.');
    });
}

/**
 * Display prediction result
 * Updates UI with prediction class and confidence
 * 
 * @param {Object} data - Response from Flask backend
 */
function displayResult(data) {
    // Hide all result cards first
    normalResult.style.display = 'none';
    pneumoniaResult.style.display = 'none';
    errorResult.style.display = 'none';

    const prediction = data.prediction;
    const confidence = data.confidence;

    console.log(`✓ Prediction: ${prediction}, Confidence: ${confidence}%`);

    if (prediction === 'NORMAL') {
        // Show normal result
        normalResult.style.display = 'block';
        document.getElementById('normalConfidence').textContent = confidence;
        
        // Animate confidence bar
        const normalBar = document.getElementById('normalBar');
        normalBar.style.width = '0%';
        setTimeout(() => {
            normalBar.style.width = confidence + '%';
        }, 100);

    } else if (prediction === 'PNEUMONIA') {
        // Show pneumonia result
        pneumoniaResult.style.display = 'block';
        document.getElementById('pneumoniaConfidence').textContent = confidence;
        
        // Animate confidence bar
        const pneumoniaBar = document.getElementById('pneumoniaBar');
        pneumoniaBar.style.width = '0%';
        setTimeout(() => {
            pneumoniaBar.style.width = confidence + '%';
        }, 100);

    } else {
        // Show error
        showError(data.message || 'Unknown prediction');
    }
}

/**
 * Show error message
 * Displays error in result card
 * 
 * @param {String} errorMessage - Error message to display
 */
function showError(errorMessage) {
    console.error('❌ Error:', errorMessage);

    // Hide other results
    normalResult.style.display = 'none';
    pneumoniaResult.style.display = 'none';

    // Show error result
    errorResult.style.display = 'block';
    document.getElementById('errorMessage').textContent = errorMessage;

    // Hide loading spinner
    loadingSpinner.style.display = 'none';
    resultContainer.style.display = 'block';

    // Show prediction section
    predictionSection.style.display = 'block';
}

// ============================================
// UI MANAGEMENT FUNCTIONS
// ============================================

/**
 * Reset UI to initial state
 * Clears all selections and hides sections
 */
function resetUI() {
    console.log('🔄 Resetting UI...');

    // Clear file input
    imageInput.value = '';
    selectedFile = null;

    // Hide sections
    previewSection.style.display = 'none';
    predictionSection.style.display = 'none';
    dragDropZone.classList.remove('show');
    dragDropZone.classList.remove('dragover');

    // Hide buttons
    analyzeButton.style.display = 'none';
    resetButton.style.display = 'none';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    console.log('✓ UI reset complete');
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('✓ Page loaded and ready');
    console.log('✓ Medical Image Classification System initialized');
});

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Log messages to console with timestamp
 * 
 * @param {String} message - Message to log
 * @param {String} type - Log type (log, warn, error)
 */
function log(message, type = 'log') {
    const timestamp = new Date().toLocaleTimeString();
    console[type](`[${timestamp}] ${message}`);
}

/**
 * Display informational toast/notification
 * (Can be implemented with a toast library if needed)
 * 
 * @param {String} message - Message to display
 * @param {String} type - Type (success, error, info)
 */
function showNotification(message, type = 'info') {
    // Simple implementation - can be enhanced
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// ============================================
// KEYBOARD SHORTCUTS
// ============================================

/**
 * Keyboard shortcuts
 * Enter: Analyze image
 * Escape: Reset UI
 */
document.addEventListener('keydown', function(event) {
    // Enter key - analyze image
    if (event.key === 'Enter' && selectedFile && 
        analyzeButton.style.display !== 'none') {
        analyzeImage();
    }

    // Escape key - reset UI
    if (event.key === 'Escape') {
        resetUI();
    }
});

// ============================================
// END OF SCRIPT
// ============================================

console.log('✓ Script.js loaded successfully');
