// SignLink Background Service Worker
console.log(' SignLink background service worker loaded');

let ws = null;

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        console.log(' WebSocket already open or connecting');
        return;
    }

    console.log('🔌 Background connecting to WebSocket...');
    ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
        console.log(' Background WebSocket connected');
        broadcastToTabs({ action: 'updateStatus', message: ' منسلک و چل رہا ہے' });
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'prediction') {
                broadcastToTabs({
                    action: 'prediction',
                    urdu: data.urdu,
                    english: data.english,
                    confidence: data.confidence
                });
            }
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        broadcastToTabs({ action: 'updateStatus', message: ' سرور سے منسلک نہیں - کیا server.py چل رہا ہے؟' });
    };

    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected, retrying in 3s...');
        ws = null;
        setTimeout(connectWebSocket, 3000);
    };
}

function broadcastToTabs(message) {
    chrome.tabs.query({ url: ['https://meet.google.com/*', 'https://*.zoom.us/*'] }, (tabs) => {
        tabs.forEach(tab => {
            chrome.tabs.sendMessage(tab.id, message).catch(() => {});
        });
    });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'connect') {
        connectWebSocket();
        sendResponse({ success: true });
    }

    else if (request.action === 'frame') {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'frame', data: request.data }));
        } else {
            // Try to reconnect if socket dropped
            connectWebSocket();
        }
        sendResponse({ success: true });
    }

    else if (request.action === 'stop') {
        if (ws) {
            try {
                ws.send(JSON.stringify({ type: 'reset' }));
                ws.close();
            } catch (e) {}
            ws = null;
        }
        sendResponse({ success: true });
    }

    return true;
});

chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason === 'install') {
        console.log(' SignLink extension installed');
    }
});