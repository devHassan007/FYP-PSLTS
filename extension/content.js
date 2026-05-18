// SignLink Content Script - Runs on Google Meet / Zoom pages

let videoElement = null;
let canvas = null;
let isRunning = false;
let captureInterval = null;
let overlay = null;

// Cache the Urdu voice once voices are ready
let urduVoice = null;

function loadUrduVoice() {
    const voices = window.speechSynthesis.getVoices();
    urduVoice = voices.find(v => v.lang === 'ur-PK' || v.lang === 'ur') || null;
    console.log('🔊 Urdu voice:', urduVoice ? urduVoice.name : 'not found on this device');
}

// Voices load asynchronously in Chrome — this event fires when they are ready
window.speechSynthesis.onvoiceschanged = loadUrduVoice;

// Also try immediately in case they are already loaded (Firefox / some builds)
loadUrduVoice();

console.log(' SignLink content script loaded');

// Create overlay for displaying subtitles
function createOverlay() {
    if (document.getElementById('signlink-overlay')) {
        return document.getElementById('signlink-overlay');
    }

    const div = document.createElement('div');
    div.id = 'signlink-overlay';
    div.style.cssText = `
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.85);
        color: white;
        padding: 20px 40px;
        border-radius: 15px;
        font-size: 28px;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif;
        direction: rtl;
        z-index: 999999;
        display: none;
        min-width: 350px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
    `;
    document.body.appendChild(div);
    return div;
}

// Speak the prediction — uses cached urduVoice
function speakPrediction(englishText, urduText) {
    window.speechSynthesis.cancel();

    const utterance  = new SpeechSynthesisUtterance(urduVoice ? urduText : englishText);
    utterance.voice  = urduVoice || null;
    utterance.lang   = urduVoice ? 'ur-PK' : 'en-US';
    utterance.rate   = 0.95;
    utterance.pitch  = 1.0;
    utterance.volume = 1.0;

    window.speechSynthesis.speak(utterance);
}

// Show Urdu subtitle on screen
function showSubtitle(urduText, englishText, confidence) {
    if (!overlay) overlay = createOverlay();

    overlay.innerHTML = `
        <div style="font-size: 32px; font-weight: 600;">${urduText}</div>
    `;
    overlay.style.display = 'block';

    speakPrediction(englishText, urduText);

    setTimeout(() => {
        if (overlay) overlay.style.display = 'none';
    }, 3000);
}

// Find the main video element
function findVideoElement() {
    const videos = document.querySelectorAll('video');
    if (videos.length === 0) {
        console.warn(' No video elements found');
        return null;
    }
    for (let video of videos) {
        if (video.videoWidth > 0 && video.videoHeight > 0) return video;
    }
    return videos[0];
}

// Capture frame and send to background service worker
function captureFrame() {
    videoElement = findVideoElement();
    if (!videoElement || videoElement.videoWidth === 0) return;

    if (!canvas) canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const frameData = canvas.toDataURL('image/jpeg', 0.7);

    chrome.runtime.sendMessage({ action: 'frame', data: frameData });
}

// Connect via background (background owns the WebSocket)
function connectWebSocket() {
    console.log('🔌 Requesting background to connect WebSocket...');
    chrome.runtime.sendMessage({ action: 'connect' });
}

// Start capturing video
function startCapture() {
    if (isRunning) {
        console.log(' Already running');
        return;
    }

    console.log('🎥 Starting capture...');
    isRunning = true;
    overlay = createOverlay();
    connectWebSocket();

    // Capture at 5 FPS
    captureInterval = setInterval(captureFrame, 200);
    console.log(' Capture started');
}

// Stop capturing video
function stopCapture() {
    if (!isRunning) return;

    console.log(' Stopping capture...');
    isRunning = false;

    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    window.speechSynthesis.cancel();

    chrome.runtime.sendMessage({ action: 'stop' });

    if (overlay) {
        overlay.remove();
        overlay = null;
    }

    console.log(' Capture stopped');
}

// Listen for messages from popup AND background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log(' Message received:', request);

    if (request.action === 'start') {
        startCapture();
        sendResponse({ success: true });
    } else if (request.action === 'stop') {
        stopCapture();
        sendResponse({ success: true });
    } else if (request.action === 'prediction') {
        if (request.urdu && request.confidence > 0.5) {
            showSubtitle(request.urdu, request.english, request.confidence);
        }
    } else if (request.action === 'updateStatus') {
        console.log('Status:', request.message);
    }

    return true;
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isRunning) stopCapture();
});