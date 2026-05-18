// SignLink Popup Script

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');

// Check if backend server is running
async function checkServer() {
    try {
        const response = await fetch('http://localhost:8000/', {
            method: 'GET',
            mode: 'cors'
        });
        
        if (response.ok) {
            status.className = 'status connected';
            status.innerHTML = ' سرور منسلک ہے';
            startBtn.disabled = false;
            return true;
        }
    } catch (e) {
        status.className = 'status disconnected';
        status.innerHTML = ' سرور سے منقطع<br><small>کیا Python سرور چل رہا ہے؟</small>';
        startBtn.disabled = true;
        return false;
    }
}

// Start button click
startBtn.addEventListener('click', async () => {
    const serverRunning = await checkServer();
    
    if (!serverRunning) {
        alert('براہ کرم پہلے Python سرور شروع کریں:\n\npython backend/server.py\n\nپھر دوبارہ کوشش کریں۔');
        return;
    }
    
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Check if we're on a supported site
    const url = tab.url;
    if (!url.includes('meet.google.com') && !url.includes('zoom.us')) {
        alert('براہ کرم Google Meet یا Zoom میں جائیں!');
        return;
    }
    
    // Send message to content script to start capture
    chrome.tabs.sendMessage(tab.id, { action: 'start' }, (response) => {
        if (chrome.runtime.lastError) {
            console.error('Error:', chrome.runtime.lastError);
            alert('خرابی: صفحہ کو دوبارہ لوڈ کریں اور کوشش کریں۔');
            return;
        }
        
        if (response && response.success) {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'block';
            status.className = 'status connected';
            status.innerHTML = '🎥 کیپچر چل رہا ہے...';
        }
    });
});

// Stop button click
stopBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.tabs.sendMessage(tab.id, { action: 'stop' }, (response) => {
        if (chrome.runtime.lastError) {
            console.error('Error:', chrome.runtime.lastError);
        }
        
        stopBtn.style.display = 'none';
        startBtn.style.display = 'block';
        checkServer(); // Update status
    });
});

// Check server status on load
checkServer();

// Check server status every 3 seconds
setInterval(checkServer, 3000);

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'updateStatus') {
        status.innerHTML = request.message;
    }
});