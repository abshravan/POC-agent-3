/**
 * Speaker Identification System - Frontend Application
 * Black and White Theme
 */

const API_BASE = '/api/v1';

// State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioContext = null;
let analyser = null;
let currentThreshold = 0.55;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeRecorder();
    initializeFileUploads();
    initializeThresholdSliders();
    checkApiStatus();
    loadSpeakers();
    loadSystemInfo();
});

// Tab Navigation
function initializeTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(targetId).classList.add('active');

            if (targetId === 'speakers') {
                loadSpeakers();
            }
        });
    });
}

// Audio Recorder
function initializeRecorder() {
    const recordBtn = document.getElementById('recordBtn');
    const recordStatus = document.getElementById('recordStatus');

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    });

    // Initialize visualizer bars
    const visualizer = document.getElementById('visualizer');
    for (let i = 0; i < 32; i++) {
        const bar = document.createElement('div');
        bar.className = 'visualizer-bar';
        visualizer.appendChild(bar);
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Setup audio context for visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 64;

        // Setup media recorder - prefer WAV format
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            await identifySpeaker(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;

        document.getElementById('recordBtn').classList.add('recording');
        document.getElementById('recordStatus').textContent = 'Recording... Click to stop';

        updateVisualizer();

    } catch (error) {
        console.error('Error starting recording:', error);
        showMessage('identifyResult', 'Error: Could not access microphone', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        document.getElementById('recordBtn').classList.remove('recording');
        document.getElementById('recordStatus').textContent = 'Processing...';

        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }
}

function updateVisualizer() {
    if (!isRecording || !analyser) return;

    const bars = document.querySelectorAll('.visualizer-bar');
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);

    bars.forEach((bar, i) => {
        const value = dataArray[i] || 0;
        const height = Math.max(4, (value / 255) * 50);
        bar.style.height = `${height}px`;
    });

    requestAnimationFrame(updateVisualizer);
}

async function identifySpeaker(audioBlob) {
    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'recording.webm');

    const threshold = document.getElementById('thresholdSlider').value / 100;

    try {
        const response = await fetch(`${API_BASE}/identify?threshold=${threshold}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            displayIdentifyResult(result);
        } else {
            showMessage('identifyResult', `Error: ${result.detail || 'Identification failed'}`, 'error');
        }
    } catch (error) {
        console.error('Identification error:', error);
        showMessage('identifyResult', `Error: ${error.message}`, 'error');
    }

    document.getElementById('recordStatus').textContent = 'Click to start recording';
}

function displayIdentifyResult(result) {
    const container = document.getElementById('identifyResult');

    if (result.is_identified) {
        container.innerHTML = `
            <div class="result-box success">
                <div class="result-speaker">${result.speaker_name || 'Unknown'}</div>
                <div class="result-confidence">
                    Employee ID: ${result.employee_id || 'N/A'}
                </div>
                <div class="result-confidence">
                    Confidence: ${(result.confidence * 100).toFixed(1)}%
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-secondary);">
                    Processing time: ${result.processing_time_ms.toFixed(0)}ms
                </div>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="result-box">
                <div class="result-speaker" style="color: var(--text-secondary);">No Match Found</div>
                <div class="result-confidence">
                    Highest confidence: ${(result.confidence * 100).toFixed(1)}%
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-secondary);">
                    Threshold: ${(result.threshold_used * 100).toFixed(0)}% |
                    Processing time: ${result.processing_time_ms.toFixed(0)}ms
                </div>
            </div>
        `;
    }
}

// File Uploads
function initializeFileUploads() {
    // Identify file upload
    const identifyUpload = document.getElementById('identifyUpload');
    const identifyFile = document.getElementById('identifyFile');
    const identifyFileBtn = document.getElementById('identifyFileBtn');

    identifyUpload.addEventListener('click', () => identifyFile.click());
    identifyUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        identifyUpload.classList.add('dragover');
    });
    identifyUpload.addEventListener('dragleave', () => {
        identifyUpload.classList.remove('dragover');
    });
    identifyUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        identifyUpload.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            identifyFile.files = e.dataTransfer.files;
            identifyFileBtn.disabled = false;
        }
    });
    identifyFile.addEventListener('change', () => {
        identifyFileBtn.disabled = !identifyFile.files.length;
    });
    identifyFileBtn.addEventListener('click', async () => {
        if (identifyFile.files.length) {
            await identifySpeaker(identifyFile.files[0]);
        }
    });

    // Enroll file upload
    const enrollUpload = document.getElementById('enrollUpload');
    const enrollFiles = document.getElementById('enrollFiles');
    const enrollFileList = document.getElementById('enrollFileList');

    enrollUpload.addEventListener('click', () => enrollFiles.click());
    enrollUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        enrollUpload.classList.add('dragover');
    });
    enrollUpload.addEventListener('dragleave', () => {
        enrollUpload.classList.remove('dragover');
    });
    enrollUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        enrollUpload.classList.remove('dragover');
        handleEnrollFiles(e.dataTransfer.files);
    });
    enrollFiles.addEventListener('change', () => {
        handleEnrollFiles(enrollFiles.files);
    });

    // Enroll form
    document.getElementById('enrollForm').addEventListener('submit', handleEnroll);
}

let enrollmentFiles = [];

function handleEnrollFiles(files) {
    enrollmentFiles = Array.from(files);
    const list = document.getElementById('enrollFileList');
    list.innerHTML = enrollmentFiles.map((f, i) => `
        <div class="file-item">
            <span>${f.name}</span>
            <button class="btn btn-secondary" onclick="removeEnrollFile(${i})" style="padding: 0.25rem 0.5rem;">
                Remove
            </button>
        </div>
    `).join('');
}

function removeEnrollFile(index) {
    enrollmentFiles.splice(index, 1);
    handleEnrollFiles(enrollmentFiles);
}

async function handleEnroll(e) {
    e.preventDefault();

    if (enrollmentFiles.length < 3) {
        showMessage('enrollResult', 'Please provide at least 3 audio samples', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('employee_id', document.getElementById('employeeId').value);
    formData.append('name', document.getElementById('speakerName').value);
    formData.append('department', document.getElementById('department').value);

    enrollmentFiles.forEach(file => {
        formData.append('audio_files', file);
    });

    document.getElementById('enrollBtn').disabled = true;
    document.getElementById('enrollBtn').textContent = 'Enrolling...';

    try {
        const response = await fetch(`${API_BASE}/enroll`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showMessage('enrollResult', `Successfully enrolled ${result.speaker_name}!`, 'success');
            document.getElementById('enrollForm').reset();
            enrollmentFiles = [];
            document.getElementById('enrollFileList').innerHTML = '';
            loadSpeakers();
        } else {
            showMessage('enrollResult', `Error: ${result.message || result.detail}`, 'error');
        }
    } catch (error) {
        showMessage('enrollResult', `Error: ${error.message}`, 'error');
    }

    document.getElementById('enrollBtn').disabled = false;
    document.getElementById('enrollBtn').textContent = 'Enroll Speaker';
}

// Speakers List
async function loadSpeakers() {
    try {
        const [speakersRes, statsRes] = await Promise.all([
            fetch(`${API_BASE}/speakers`),
            fetch(`${API_BASE}/admin/stats`)
        ]);

        if (speakersRes.ok) {
            const data = await speakersRes.json();
            displaySpeakers(data.speakers);
        }

        if (statsRes.ok) {
            const stats = await statsRes.json();
            document.getElementById('totalSpeakers').textContent = stats.total_speakers;
            document.getElementById('activeSpeakers').textContent = stats.active_speakers;
            document.getElementById('totalEmbeddings').textContent = stats.total_embeddings;
        }
    } catch (error) {
        console.error('Error loading speakers:', error);
    }
}

function displaySpeakers(speakers) {
    const tbody = document.getElementById('speakersList');

    if (!speakers || speakers.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; color: var(--text-secondary);">
                    No speakers enrolled
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = speakers.map(s => `
        <tr>
            <td><strong>${s.name}</strong></td>
            <td>${s.employee_id}</td>
            <td>${s.department || '-'}</td>
            <td>${s.embedding_count}</td>
            <td>${new Date(s.enrolled_at).toLocaleDateString()}</td>
            <td>
                <button class="btn btn-danger" onclick="removeSpeaker('${s.id}', '${s.name}')"
                        style="padding: 0.25rem 0.5rem; font-size: 0.85rem;">
                    Remove
                </button>
            </td>
        </tr>
    `).join('');
}

async function removeSpeaker(id, name) {
    if (!confirm(`Are you sure you want to remove ${name}?`)) return;

    try {
        const response = await fetch(`${API_BASE}/speakers/${id}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadSpeakers();
        } else {
            const error = await response.json();
            alert(`Error: ${error.detail || 'Failed to remove speaker'}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Search speakers
document.getElementById('searchSpeakers')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#speakersList tr');

    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
});

document.getElementById('refreshSpeakers')?.addEventListener('click', loadSpeakers);

// Threshold Sliders
function initializeThresholdSliders() {
    const slider = document.getElementById('thresholdSlider');
    const value = document.getElementById('thresholdValue');

    slider.addEventListener('input', () => {
        value.textContent = (slider.value / 100).toFixed(2);
    });

    const globalSlider = document.getElementById('globalThreshold');
    const globalValue = document.getElementById('globalThresholdValue');

    globalSlider?.addEventListener('input', () => {
        globalValue.textContent = (globalSlider.value / 100).toFixed(2);
    });

    document.getElementById('saveThreshold')?.addEventListener('click', saveThreshold);
}

async function saveThreshold() {
    const threshold = document.getElementById('globalThreshold').value / 100;

    try {
        const response = await fetch(`${API_BASE}/admin/thresholds`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ threshold })
        });

        if (response.ok) {
            showMessage('settings', 'Threshold saved successfully', 'success');
        } else {
            showMessage('settings', 'Failed to save threshold', 'error');
        }
    } catch (error) {
        showMessage('settings', `Error: ${error.message}`, 'error');
    }
}

// System Info
async function loadSystemInfo() {
    try {
        const response = await fetch(`${API_BASE}/admin/stats`);
        if (response.ok) {
            const stats = await response.json();
            document.getElementById('infoModel').textContent = stats.model_name || '-';
            document.getElementById('infoDevice').textContent = stats.device || '-';
            document.getElementById('infoDimension').textContent = stats.embedding_dimension || '-';
        }

        const healthRes = await fetch('/health');
        if (healthRes.ok) {
            const health = await healthRes.json();
            document.getElementById('infoVersion').textContent = health.version || '-';
        }
    } catch (error) {
        console.error('Error loading system info:', error);
    }
}

// Clear Cache
document.getElementById('clearCache')?.addEventListener('click', async () => {
    try {
        const response = await fetch(`${API_BASE}/admin/cache/clear`, {
            method: 'POST'
        });

        if (response.ok) {
            const result = await response.json();
            alert(`Cache cleared: ${result.entries_cleared} entries removed`);
        } else {
            alert('Failed to clear cache');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
});

// API Status Check
async function checkApiStatus() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        document.getElementById('statusDot').classList.remove('connected');
        document.getElementById('statusText').textContent = 'Disconnected';
    }

    // Check again in 30 seconds
    setTimeout(checkApiStatus, 30000);
}

// Utility: Show Message
function showMessage(containerId, message, type = 'info') {
    const container = document.getElementById(containerId);
    if (!container) return;

    const msgEl = document.createElement('div');
    msgEl.className = `message ${type}`;
    msgEl.textContent = message;

    // For result containers, replace content
    if (containerId.includes('Result')) {
        container.innerHTML = '';
        container.appendChild(msgEl);
    } else {
        // For other containers, insert at top
        container.insertBefore(msgEl, container.firstChild);
        setTimeout(() => msgEl.remove(), 5000);
    }
}
