// Doom Arena - Game client
// Handles canvas rendering, Pointer Lock, keyboard/mouse capture, and WebSocket I/O

const sessionId = window.location.pathname.split('/')[2];
const urlParams = new URLSearchParams(window.location.search);
const slotId = urlParams.get('slot');
const joinToken = urlParams.get('token');

const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const clickOverlay = document.getElementById('click-overlay');
const hud = document.getElementById('hud');
const scoreboard = document.getElementById('scoreboard');
const connStatus = document.getElementById('connection-status');

// Input state
const INPUT_MSG_TYPE = 0x02;
const keys = {};
let mouseDx = 0;
let mouseDown = false;
let pointerLocked = false;

// Key → bitmask bit mapping
const KEY_MAP = {
    'KeyW': 0, 'ArrowUp': 0,
    'KeyS': 1, 'ArrowDown': 1,
    'KeyD': 2, 'ArrowRight': 2,
    'KeyA': 3, 'ArrowLeft': 3,
    'Digit1': 4, 'Digit2': 5, 'Digit3': 6, 'Digit4': 7,
    'Digit5': 8, 'Digit6': 9, 'Digit7': 10,
    'ShiftLeft': 12, 'ShiftRight': 12,
};

// WebSocket
let ws = null;
let inputInterval = null;

// Frame rendering
const frameImage = new Image();
let lastFrameBlob = null;
let lastFrameUrl = null;

function connect() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws/game/${sessionId}/${slotId}?token=${joinToken}`;
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        connStatus.textContent = 'Connected - click to play';
        connStatus.className = 'status connected';
        startInputLoop();
    };

    ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            handleTextMessage(JSON.parse(event.data));
        } else {
            handleBinaryMessage(event.data);
        }
    };

    ws.onclose = () => {
        connStatus.textContent = 'Disconnected';
        connStatus.className = 'status error';
        stopInputLoop();
    };

    ws.onerror = () => {
        connStatus.textContent = 'Connection error';
        connStatus.className = 'status error';
    };
}

function handleBinaryMessage(buffer) {
    // Parse frame header
    const view = new DataView(buffer);
    const msgType = view.getUint8(0);
    if (msgType !== 0x01) return;

    // const tic = view.getUint32(1, true);
    const health = view.getInt16(5, true);
    const frags = view.getInt16(7, true);
    const deaths = view.getInt16(9, true);

    // Update HUD
    document.getElementById('hud-health').textContent = `HP: ${health}`;
    document.getElementById('hud-frags').textContent = `Frags: ${frags}`;
    document.getElementById('hud-deaths').textContent = `Deaths: ${Math.abs(deaths)}`;
    hud.style.display = 'flex';

    // Extract JPEG data and render
    const jpegData = new Uint8Array(buffer, 11);
    const blob = new Blob([jpegData], { type: 'image/jpeg' });

    // Clean up previous URL
    if (lastFrameUrl) {
        URL.revokeObjectURL(lastFrameUrl);
    }
    lastFrameUrl = URL.createObjectURL(blob);

    frameImage.onload = () => {
        ctx.drawImage(frameImage, 0, 0, canvas.width, canvas.height);
    };
    frameImage.src = lastFrameUrl;
}

function handleTextMessage(msg) {
    switch (msg.type) {
        case 'scores':
            showScoreboard(
                `Episode ${msg.episode}/${msg.total_episodes}`,
                msg.scores,
                msg.next_in ? `Next episode in ${msg.next_in}s...` : ''
            );
            // Auto-hide after delay
            if (msg.next_in) {
                setTimeout(() => {
                    scoreboard.classList.remove('active');
                }, (msg.next_in - 1) * 1000);
            }
            break;

        case 'game_over':
            showScoreboard('Game Over', msg.scores, `${msg.episodes_played} episode(s) played`);
            stopInputLoop();
            connStatus.textContent = 'Game finished';
            connStatus.className = 'status';
            break;
    }
}

function showScoreboard(title, scores, info) {
    document.getElementById('scoreboard-title').textContent = title;
    document.getElementById('scoreboard-body').innerHTML = scores.map(s => `
        <tr>
            <td>${escapeHtml(s.name)}</td>
            <td><span class="player-type ${s.type}">${s.type}</span></td>
            <td style="color: var(--warning)">${s.frags}</td>
            <td style="color: var(--accent)">${s.deaths}</td>
        </tr>
    `).join('');
    document.getElementById('scoreboard-info').textContent = info || '';
    scoreboard.classList.add('active');
}

// --- Input Handling ---

function requestPointerLock() {
    canvas.requestPointerLock();
}

document.addEventListener('pointerlockchange', () => {
    pointerLocked = !!document.pointerLockElement;
    clickOverlay.classList.toggle('hidden', pointerLocked);
    if (!pointerLocked) {
        // Reset input state when pointer unlocked
        for (const k in keys) keys[k] = false;
        mouseDown = false;
        mouseDx = 0;
    }
});

document.addEventListener('keydown', (e) => {
    if (!pointerLocked) return;
    e.preventDefault();
    keys[e.code] = true;
    // ESC exits pointer lock (browser default), no need to handle
});

document.addEventListener('keyup', (e) => {
    if (!pointerLocked) return;
    e.preventDefault();
    keys[e.code] = false;
});

document.addEventListener('mousemove', (e) => {
    if (!pointerLocked) return;
    mouseDx += e.movementX;
});

document.addEventListener('mousedown', (e) => {
    if (!pointerLocked) return;
    if (e.button === 0) mouseDown = true;
});

document.addEventListener('mouseup', (e) => {
    if (!pointerLocked) return;
    if (e.button === 0) mouseDown = false;
});

// Prevent right-click context menu
document.addEventListener('contextmenu', (e) => e.preventDefault());

function buildInputMessage() {
    // Build key bitmask
    let keyState = 0;
    for (const [code, bit] of Object.entries(KEY_MAP)) {
        if (keys[code]) keyState |= (1 << bit);
    }
    // Attack from mouse
    if (mouseDown) keyState |= (1 << 11);

    // Build binary message: [type(1), keyState(2), mouseDx(4)]
    const buffer = new ArrayBuffer(7);
    const view = new DataView(buffer);
    view.setUint8(0, INPUT_MSG_TYPE);
    view.setUint16(1, keyState, true);
    view.setFloat32(3, mouseDx, true);

    // Reset accumulated mouse delta after sending
    mouseDx = 0;

    return buffer;
}

function startInputLoop() {
    // Send input at ~35 Hz (matching game FPS)
    inputInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN && pointerLocked) {
            const msg = buildInputMessage();
            ws.send(msg);
        }
    }, 1000 / 35);
}

function stopInputLoop() {
    if (inputInterval) {
        clearInterval(inputInterval);
        inputInterval = null;
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Start connection
connect();
