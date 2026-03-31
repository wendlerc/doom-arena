// Doom Arena - Lobby WebSocket client

const sessionId = window.location.pathname.split('/').pop();
const urlParams = new URLSearchParams(window.location.search);
const adminToken = urlParams.get('admin');

let ws = null;
let mySlotId = null;
let myJoinToken = null;

function joinGame() {
    const name = document.getElementById('player-name').value.trim();
    if (!name) {
        document.getElementById('player-name').focus();
        return;
    }

    // Connect WebSocket
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${window.location.host}/ws/lobby/${sessionId}`);

    ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'join', name: name }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };

    ws.onclose = () => {
        document.getElementById('status').className = 'status error';
        document.getElementById('status').textContent = 'Disconnected from server';
    };

    ws.onerror = () => {
        document.getElementById('status').className = 'status error';
        document.getElementById('status').textContent = 'Connection error';
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'joined':
            mySlotId = msg.slot_id;
            myJoinToken = msg.join_token;

            // Switch to lobby view
            document.getElementById('join-card').style.display = 'none';
            document.getElementById('lobby-card').style.display = 'block';

            // Set share link
            const shareUrl = `${window.location.origin}/lobby/${sessionId}`;
            document.getElementById('lobby-share-link').value = shareUrl;

            // Show admin controls if admin
            if (adminToken) {
                document.getElementById('admin-controls').style.display = 'block';
            }

            // Load session config
            loadConfig();
            break;

        case 'player_list':
            updatePlayerList(msg.players);
            break;

        case 'game_started':
            // Redirect to game page
            document.getElementById('lobby-card').style.display = 'none';
            document.getElementById('starting-card').style.display = 'block';

            const gameUrl = `${msg.game_url}?slot=${mySlotId}&token=${myJoinToken}`;
            setTimeout(() => {
                window.location.href = gameUrl;
            }, 1000);
            break;

        case 'error':
            alert(msg.message);
            break;
    }
}

function updatePlayerList(players) {
    const list = document.getElementById('player-list');
    list.innerHTML = players.map(p => `
        <li class="player-item">
            <span class="player-name">${escapeHtml(p.name)}${p.slot_id === mySlotId ? ' (you)' : ''}</span>
            <span class="player-type ${p.type}">${p.type}</span>
        </li>
    `).join('');

    const humanCount = players.filter(p => p.type === 'human').length;
    document.getElementById('status').textContent = `${humanCount} player${humanCount !== 1 ? 's' : ''} in lobby`;
    document.getElementById('status').className = 'status connected';
}

async function loadConfig() {
    try {
        const resp = await fetch(`/api/sessions/${sessionId}`);
        const session = await resp.json();
        const c = session.config;
        document.getElementById('config-info').innerHTML = `
            <strong>${c.wad}</strong> &mdash;
            ${c.timelimit} min &bull;
            ${c.num_bots} bots &bull;
            ${c.num_episodes} episode${c.num_episodes > 1 ? 's' : ''} &bull;
            Max ${c.max_human_players} players
        `;
    } catch (e) {}
}

async function startGame() {
    if (!adminToken) return;

    const btn = document.getElementById('start-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    try {
        const resp = await fetch(`/api/sessions/${sessionId}/start?admin_token=${adminToken}`, {
            method: 'POST',
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || 'Failed to start');
            btn.disabled = false;
            btn.textContent = 'Start Game';
        }
    } catch (err) {
        alert('Error: ' + err.message);
        btn.disabled = false;
        btn.textContent = 'Start Game';
    }
}

function copyLobbyLink() {
    const input = document.getElementById('lobby-share-link');
    input.select();
    navigator.clipboard.writeText(input.value);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Keep connection alive
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 15000);
