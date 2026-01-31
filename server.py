#!/usr/bin/env python3
"""
ytdl-web - YouTube Downloader
Supports multiple concurrent downloads
Cross-platform: Windows, macOS, Linux
"""

import os
import re
import sys
import uuid
import json
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse, unquote, quote
import subprocess
import shutil

# Get FFmpeg path from imageio-ffmpeg (cross-platform)
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[OK] FFmpeg found: {FFMPEG_PATH}")
except ImportError:
    FFMPEG_PATH = None
    print("[WARN] imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 8080
EXTERNAL_PORT = int(os.environ.get('EXTERNAL_PORT', PORT))
DOWNLOAD_DIR = Path(__file__).parent / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Track active downloads
active_downloads = {}
downloads_lock = threading.Lock()


def parse_progress(line):
    """Parse yt-dlp progress output and return percentage (0-100)."""
    # yt-dlp output format: [download]  45.2% of 10.5MiB at 2.3MiB/s ETA 00:05
    if '[download]' in line and '%' in line:
        try:
            # Extract percentage
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))
        except (ValueError, AttributeError):
            pass
    # Merger progress
    if '[Merger]' in line or '[ExtractAudio]' in line:
        return 95  # Near completion during merge/extract
    return None

# HTML Frontend - YouTube Dark Mode Style
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Downloader</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0f0f0f;
            color: #f1f1f1;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 700px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid #272727;
        }

        .logo {
            width: 40px;
            height: 40px;
            background: #ff0000;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .logo svg { width: 24px; height: 24px; fill: white; }

        .header h1 {
            font-size: 20px;
            font-weight: 500;
            color: #f1f1f1;
        }

        /* Tab Navigation */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .tab {
            padding: 10px 24px;
            background: transparent;
            border: none;
            color: #aaa;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 20px;
            transition: all 0.2s;
        }

        .tab:hover { background: #272727; color: #f1f1f1; }
        .tab.active { background: #272727; color: #f1f1f1; }

        /* Content Sections */
        .content-section { display: none; }
        .content-section.active { display: block; }

        /* Input Section */
        .input-section {
            background: #272727;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        .input-group input {
            flex: 1;
            padding: 12px 16px;
            background: #121212;
            border: 1px solid #3f3f3f;
            border-radius: 8px;
            color: #f1f1f1;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
            min-width: 0;
        }

        .input-group input:focus { border-color: #3ea6ff; }
        .input-group input::placeholder { color: #717171; }

        .btn {
            padding: 12px 24px;
            background: #3ea6ff;
            border: none;
            border-radius: 8px;
            color: #0f0f0f;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .btn:hover { background: #65b8ff; }
        .btn:disabled { background: #3f3f3f; color: #717171; cursor: not-allowed; }

        .btn-convert {
            background: #ff0000;
            color: white;
        }

        .btn-convert:hover { background: #cc0000; }

        .btn-danger {
            background: #cc0000;
            color: white;
            padding: 8px 12px;
            font-size: 12px;
        }

        .btn-danger:hover { background: #aa0000; }

        /* Video Info Card */
        .video-card {
            display: none;
            background: #272727;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        .video-card.show { display: block; }

        .thumbnail-container {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #1a1a1a;
        }

        .thumbnail {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .duration-badge {
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: rgba(0,0,0,0.8);
            color: #f1f1f1;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }

        .video-details {
            padding: 16px;
        }

        .video-title {
            font-size: 16px;
            font-weight: 500;
            color: #f1f1f1;
            margin-bottom: 16px;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        /* Quality Selection */
        .quality-section {
            margin-bottom: 16px;
        }

        .quality-label {
            font-size: 13px;
            color: #aaa;
            margin-bottom: 8px;
        }

        .quality-select {
            width: 100%;
            padding: 12px 16px;
            background: #121212;
            border: 1px solid #3f3f3f;
            border-radius: 8px;
            color: #f1f1f1;
            font-size: 14px;
            cursor: pointer;
            outline: none;
        }

        .quality-select:focus { border-color: #3ea6ff; }

        .estimated-size {
            font-size: 12px;
            color: #3ea6ff;
            margin-top: 8px;
        }

        .convert-section {
            display: flex;
            justify-content: center;
            padding-top: 8px;
        }

        .convert-section .btn { padding: 12px 48px; }

        /* Status Section */
        .status-section {
            display: none;
            background: #272727;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .status-section.show { display: block; }

        .status-text {
            text-align: center;
            color: #f1f1f1;
            margin-bottom: 12px;
            font-size: 14px;
        }

        .progress-bar {
            height: 4px;
            background: #3f3f3f;
            border-radius: 2px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: #ff0000;
            width: 0%;
            transition: width 0.3s;
            border-radius: 2px;
        }

        /* Download Button */
        .download-section {
            display: none;
            text-align: center;
            margin-bottom: 20px;
        }

        .download-section.show { display: block; }

        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 14px 32px;
            background: #2ba640;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
        }

        .download-btn:hover { background: #239936; }

        .download-btn svg {
            width: 20px;
            height: 20px;
            fill: currentColor;
        }

        /* Error */
        .error {
            display: none;
            background: rgba(255, 0, 0, 0.1);
            border: 1px solid #ff4444;
            color: #ff6b6b;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }

        .error.show { display: block; }

        /* Server Status */
        .server-status {
            text-align: center;
            padding: 12px;
            background: #272727;
            border-radius: 8px;
            font-size: 13px;
            color: #aaa;
        }

        .server-status span { color: #3ea6ff; font-weight: 500; }

        /* Supported URLs */
        .supported-urls {
            margin-top: 20px;
            padding: 16px;
            background: #1a1a1a;
            border-radius: 8px;
            border: 1px solid #272727;
        }

        .supported-urls h3 {
            font-size: 13px;
            color: #aaa;
            margin-bottom: 12px;
            font-weight: 500;
        }

        .supported-urls ul {
            list-style: none;
        }

        .supported-urls li {
            font-size: 12px;
            color: #717171;
            padding: 4px 0;
        }

        .supported-urls code {
            background: #272727;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
            color: #aaa;
        }

        /* History Section */
        .history-section {
            background: #272727;
            border-radius: 12px;
            padding: 16px;
        }

        .history-empty {
            text-align: center;
            color: #717171;
            padding: 40px 20px;
            font-size: 14px;
        }

        .history-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .history-item {
            display: flex;
            gap: 12px;
            background: #1a1a1a;
            border-radius: 8px;
            padding: 12px;
            position: relative;
        }

        .history-item.deleted {
            opacity: 0.6;
        }

        .history-thumb {
            width: 120px;
            height: 68px;
            border-radius: 6px;
            object-fit: cover;
            background: #0f0f0f;
            flex-shrink: 0;
        }

        .history-info {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .history-title {
            font-size: 14px;
            font-weight: 500;
            color: #f1f1f1;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            cursor: pointer;
            text-decoration: none;
        }

        .history-title:hover {
            color: #3ea6ff;
        }

        a.history-title:visited {
            color: #f1f1f1;
        }

        a.history-title:visited:hover {
            color: #3ea6ff;
        }

        .history-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin-top: 4px;
        }

        .history-badge {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 500;
        }

        .badge-mp3 {
            background: #ff5722;
            color: white;
        }

        .badge-mp4 {
            background: #2196f3;
            color: white;
        }

        .badge-quality {
            background: #3f3f3f;
            color: #aaa;
        }

        .history-date {
            font-size: 11px;
            color: #717171;
        }

        .history-actions {
            display: flex;
            gap: 8px;
            align-items: flex-start;
            flex-shrink: 0;
        }

        .history-download-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            background: #2ba640;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            transition: all 0.2s;
        }

        .history-download-btn:hover { background: #239936; }
        .history-download-btn:disabled {
            background: #3f3f3f;
            cursor: not-allowed;
        }

        .history-download-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }

        .history-delete-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            background: #cc0000;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            transition: all 0.2s;
        }

        .history-delete-btn:hover { background: #aa0000; }

        .history-delete-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }

        .deleted-tooltip {
            font-size: 11px;
            color: #ff6b6b;
            margin-top: 4px;
        }

        /* Mobile Responsive */
        @media (max-width: 600px) {
            body {
                padding: 12px;
            }

            .header h1 {
                font-size: 18px;
            }

            .tabs {
                gap: 4px;
            }

            .tab {
                padding: 8px 14px;
                font-size: 13px;
            }

            .input-group {
                flex-direction: column;
            }

            .input-group input {
                width: 100%;
            }

            .btn {
                width: 100%;
                padding: 14px 20px;
            }

            .convert-section .btn {
                width: 100%;
                padding: 14px 20px;
            }

            .video-details {
                padding: 12px;
            }

            .video-title {
                font-size: 14px;
            }

            .download-btn {
                width: 100%;
                justify-content: center;
                padding: 14px 20px;
            }

            .history-item {
                flex-direction: column;
            }

            .history-thumb {
                width: 100%;
                height: auto;
                aspect-ratio: 16/9;
            }

            .history-actions {
                flex-direction: row;
                justify-content: flex-end;
                margin-top: 8px;
            }

            .supported-urls {
                padding: 12px;
            }

            .supported-urls li {
                font-size: 11px;
            }
        }

        @media (max-width: 400px) {
            .tab {
                padding: 8px 10px;
                font-size: 12px;
            }

            .history-meta {
                flex-direction: column;
                align-items: flex-start;
                gap: 4px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <svg viewBox="0 0 24 24"><path d="M10 15l5.19-3L10 9v6m11.56-7.83c.13.47.22 1.1.28 1.9.07.8.1 1.49.1 2.09L22 12c0 2.19-.16 3.8-.44 4.83-.25.9-.83 1.48-1.73 1.73-.47.13-1.33.22-2.65.28-1.3.07-2.49.1-3.59.1L12 19c-4.19 0-6.8-.16-7.83-.44-.9-.25-1.48-.83-1.73-1.73-.13-.47-.22-1.1-.28-1.9-.07-.8-.1-1.49-.1-2.09L2 12c0-2.19.16-3.8.44-4.83.25-.9.83-1.48 1.73-1.73.47-.13 1.33-.22 2.65-.28 1.3-.07 2.49-.1 3.59-.1L12 5c4.19 0 6.8.16 7.83.44.9.25 1.48.83 1.73 1.73z"/></svg>
            </div>
            <h1>YouTube Downloader</h1>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('audio')" id="tabAudio">Audio (MP3)</button>
            <button class="tab" onclick="switchTab('video')" id="tabVideo">Video (MP4)</button>
            <button class="tab" onclick="switchTab('history')" id="tabHistory">History</button>
        </div>

        <!-- Download Content Section -->
        <div class="content-section active" id="downloadContent">
            <div class="input-section">
                <div class="input-group">
                    <input type="text" id="url" placeholder="Paste YouTube URL here..." autofocus>
                    <button class="btn" id="fetchBtn" onclick="fetchInfo()">Get Info</button>
                </div>
            </div>

            <div class="error" id="error"></div>

            <div class="video-card" id="videoCard">
                <div class="thumbnail-container">
                    <img class="thumbnail" id="thumbnail" src="" alt="Video thumbnail">
                    <div class="duration-badge" id="durationBadge">0:00</div>
                </div>
                <div class="video-details">
                    <div class="video-title" id="videoTitle"></div>

                    <div class="quality-section">
                        <div class="quality-label">Select Quality:</div>
                        <!-- Audio Quality Options -->
                        <select id="audioQuality" class="quality-select" onchange="updateEstimatedSize()">
                            <option value="320">320 kbps - Best Quality</option>
                            <option value="256">256 kbps - High Quality</option>
                            <option value="192">192 kbps - Good Quality</option>
                            <option value="128" selected>128 kbps - Standard</option>
                            <option value="96">96 kbps - Low</option>
                            <option value="64">64 kbps - Smallest</option>
                        </select>
                        <!-- Video Quality Options -->
                        <select id="videoQuality" class="quality-select" style="display:none;" onchange="updateEstimatedSize()">
                            <option value="2160">4K (2160p)</option>
                            <option value="1440">2K (1440p)</option>
                            <option value="1080" selected>Full HD (1080p)</option>
                            <option value="720">HD (720p)</option>
                            <option value="480">SD (480p)</option>
                            <option value="360">Low (360p)</option>
                        </select>
                        <div class="estimated-size" id="estimatedSize"></div>
                    </div>

                    <div class="convert-section">
                        <button class="btn btn-convert" id="convertBtn" onclick="convert()">
                            <span id="convertBtnText">Download MP3</span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="status-section" id="status">
                <div class="status-text" id="statusText">Starting...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>

            <div class="download-section" id="downloadSection">
                <a href="#" class="download-btn" id="downloadBtn">
                    <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
                    <span id="downloadText">Download</span>
                </a>
            </div>

            <div class="server-status">
                Active conversions: <span id="activeCount">0</span>
            </div>

            <div class="supported-urls">
                <h3>Supported URL Formats:</h3>
                <ul>
                    <li><code>youtube.com/watch?v=...</code></li>
                    <li><code>youtu.be/...</code></li>
                    <li><code>youtube.com/shorts/...</code></li>
                </ul>
            </div>
        </div>

        <!-- History Content Section -->
        <div class="content-section" id="historyContent">
            <div class="history-section">
                <div class="history-list" id="historyList">
                    <div class="history-empty">No download history yet. Convert some videos to see them here.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTaskId = null;
        let pollInterval = null;
        let videoDuration = 0;
        let currentUrl = '';
        let currentThumbnail = '';
        let currentTitle = '';
        let currentMode = 'audio';
        let currentView = 'download';

        // Initialize
        setInterval(updateActiveCount, 3000);
        updateActiveCount();
        loadHistory();

        function switchTab(mode) {
            // Handle history tab separately
            if (mode === 'history') {
                currentView = 'history';
                document.getElementById('tabAudio').classList.remove('active');
                document.getElementById('tabVideo').classList.remove('active');
                document.getElementById('tabHistory').classList.add('active');
                document.getElementById('downloadContent').classList.remove('active');
                document.getElementById('historyContent').classList.add('active');
                loadHistory();
                return;
            }

            // Audio/Video tabs
            currentView = 'download';
            currentMode = mode;
            document.getElementById('tabAudio').classList.toggle('active', mode === 'audio');
            document.getElementById('tabVideo').classList.toggle('active', mode === 'video');
            document.getElementById('tabHistory').classList.remove('active');
            document.getElementById('downloadContent').classList.add('active');
            document.getElementById('historyContent').classList.remove('active');
            document.getElementById('audioQuality').style.display = mode === 'audio' ? 'block' : 'none';
            document.getElementById('videoQuality').style.display = mode === 'video' ? 'block' : 'none';
            document.getElementById('convertBtnText').textContent = mode === 'audio' ? 'Download MP3' : 'Download MP4';
            updateEstimatedSize();
        }

        function updateActiveCount() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => document.getElementById('activeCount').textContent = data.active_count)
                .catch(() => {});
        }

        function formatDuration(seconds) {
            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            if (hrs > 0) return `${hrs}:${mins.toString().padStart(2,'0')}:${secs.toString().padStart(2,'0')}`;
            return `${mins}:${secs.toString().padStart(2,'0')}`;
        }

        function formatFileSize(bytes) {
            if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
            if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
            return (bytes / 1024).toFixed(1) + ' KB';
        }

        function formatDate(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;

            if (diff < 60000) return 'Just now';
            if (diff < 3600000) return Math.floor(diff / 60000) + ' min ago';
            if (diff < 86400000) return Math.floor(diff / 3600000) + ' hours ago';
            if (diff < 604800000) return Math.floor(diff / 86400000) + ' days ago';

            return date.toLocaleDateString();
        }

        function updateEstimatedSize() {
            if (videoDuration <= 0) return;
            let bytes;
            if (currentMode === 'audio') {
                const bitrate = parseInt(document.getElementById('audioQuality').value);
                bytes = (bitrate * 1000 / 8) * videoDuration;
            } else {
                const resolution = parseInt(document.getElementById('videoQuality').value);
                const bitrateMap = {2160: 15000, 1440: 8000, 1080: 4000, 720: 2000, 480: 1000, 360: 400};
                bytes = ((bitrateMap[resolution] || 2000) * 1000 / 8) * videoDuration;
            }
            document.getElementById('estimatedSize').textContent = 'Estimated size: ~' + formatFileSize(bytes);
        }

        // History Management
        function getHistory() {
            try {
                return JSON.parse(localStorage.getItem('downloadHistory') || '[]');
            } catch (e) {
                return [];
            }
        }

        function saveHistory(history) {
            localStorage.setItem('downloadHistory', JSON.stringify(history));
        }

        function addToHistory(item) {
            const history = getHistory();
            // Avoid duplicates by task_id
            const existingIndex = history.findIndex(h => h.task_id === item.task_id);
            if (existingIndex >= 0) {
                history[existingIndex] = item;
            } else {
                history.unshift(item);
            }
            // Keep only last 50 items
            if (history.length > 50) history.pop();
            saveHistory(history);
        }

        function removeFromHistory(taskId) {
            const history = getHistory().filter(h => h.task_id !== taskId);
            saveHistory(history);
            loadHistory();
        }

        async function checkFileExists(taskId) {
            try {
                const response = await fetch('/api/file-exists/' + taskId);
                const data = await response.json();
                return data.exists;
            } catch (e) {
                return false;
            }
        }

        async function deleteHistoryItem(taskId) {
            if (!confirm('Delete this file from server and history?')) return;

            try {
                await fetch('/api/delete/' + taskId, { method: 'DELETE' });
            } catch (e) {
                // File might already be deleted, continue anyway
            }
            removeFromHistory(taskId);
        }

        async function loadHistory() {
            const historyList = document.getElementById('historyList');
            const history = getHistory();

            if (history.length === 0) {
                historyList.innerHTML = '<div class="history-empty">No download history yet. Convert some videos to see them here.</div>';
                return;
            }

            // Check file existence for all items
            const existsPromises = history.map(item => checkFileExists(item.task_id));
            const existsResults = await Promise.all(existsPromises);

            let html = '';
            history.forEach((item, index) => {
                const exists = existsResults[index];
                const isMP3 = item.type === 'audio';
                const qualityLabel = isMP3 ? item.quality + ' kbps' : item.quality + 'p';

                const youtubeUrl = item.youtube_url || '';
                html += `
                    <div class="history-item ${exists ? '' : 'deleted'}">
                        <a href="${youtubeUrl}" target="_blank" rel="noopener" style="flex-shrink:0;">
                            <img class="history-thumb" src="${item.thumbnail || ''}" alt="" style="cursor:pointer;" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 9%22><rect fill=%22%23272727%22 width=%2216%22 height=%229%22/></svg>'">
                        </a>
                        <div class="history-info">
                            <a href="${youtubeUrl}" target="_blank" rel="noopener" class="history-title" title="Open on YouTube">
                                ${escapeHtml(item.title)}
                            </a>
                            <div class="history-meta">
                                <span class="history-badge ${isMP3 ? 'badge-mp3' : 'badge-mp4'}">${isMP3 ? 'MP3' : 'MP4'}</span>
                                <span class="history-badge badge-quality">${qualityLabel}</span>
                                <span class="history-date">${formatDate(item.date)}</span>
                            </div>
                            ${!exists ? '<div class="deleted-tooltip">File no longer available on server</div>' : ''}
                        </div>
                        <div class="history-actions">
                            <button class="history-download-btn" ${exists ? '' : 'disabled'}
                                    onclick="${exists ? `window.location.href='${item.download_url}'` : ''}"
                                    title="${exists ? 'Download' : 'File unavailable'}">
                                <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
                            </button>
                            <button class="history-delete-btn" onclick="deleteHistoryItem('${item.task_id}')" title="Delete">
                                <svg viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                            </button>
                        </div>
                    </div>
                `;
            });

            historyList.innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function fetchInfo() {
            const url = document.getElementById('url').value.trim();
            if (!url) { showError('Please enter a YouTube URL'); return; }

            hideError();
            hideDownload();
            hideVideoCard();
            hideStatus();
            document.getElementById('fetchBtn').disabled = true;
            document.getElementById('fetchBtn').textContent = 'Loading...';

            fetch('/api/info', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: url})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('fetchBtn').disabled = false;
                document.getElementById('fetchBtn').textContent = 'Get Info';

                if (data.error) { showError(data.error); return; }

                currentUrl = url;
                videoDuration = data.duration || 0;
                currentThumbnail = data.thumbnail || '';
                currentTitle = data.title || 'Unknown';

                document.getElementById('videoTitle').textContent = data.title;
                document.getElementById('durationBadge').textContent = formatDuration(videoDuration);
                document.getElementById('thumbnail').src = currentThumbnail;

                updateEstimatedSize();
                showVideoCard();
            })
            .catch(err => {
                document.getElementById('fetchBtn').disabled = false;
                document.getElementById('fetchBtn').textContent = 'Get Info';
                showError('Failed to get video info: ' + err.message);
            });
        }

        function convert() {
            if (!currentUrl) { showError('Please fetch video info first'); return; }

            const quality = currentMode === 'audio'
                ? document.getElementById('audioQuality').value
                : document.getElementById('videoQuality').value;

            hideError();
            hideDownload();
            showStatus('Starting download...');
            setProgress(0);
            document.getElementById('convertBtn').disabled = true;

            const endpoint = currentMode === 'audio' ? '/api/convert' : '/api/convert-video';

            fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: currentUrl, bitrate: quality, resolution: quality})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                    hideStatus();
                    document.getElementById('convertBtn').disabled = false;
                    return;
                }

                currentTaskId = data.task_id;
                // Store current conversion info for history
                window.currentConversion = {
                    task_id: data.task_id,
                    title: currentTitle,
                    thumbnail: currentThumbnail,
                    youtube_url: currentUrl,
                    type: currentMode,
                    quality: quality,
                    date: Date.now()
                };
                showStatus('Downloading...');
                pollInterval = setInterval(checkStatus, 500);
            })
            .catch(err => {
                showError('Failed to start: ' + err.message);
                hideStatus();
                document.getElementById('convertBtn').disabled = false;
            });
        }

        function checkStatus() {
            if (!currentTaskId) return;

            fetch('/api/check/' + currentTaskId)
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        setProgress(100);
                        showStatus('Complete!');
                        showDownload(data.filename, data.download_url);
                        document.getElementById('convertBtn').disabled = false;
                        updateActiveCount();

                        // Add to history
                        if (window.currentConversion) {
                            addToHistory({
                                ...window.currentConversion,
                                filename: data.filename,
                                download_url: data.download_url
                            });
                            window.currentConversion = null;
                        }
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        showError(data.error || 'Download failed');
                        hideStatus();
                        document.getElementById('convertBtn').disabled = false;
                        updateActiveCount();
                        window.currentConversion = null;
                    } else if (data.status === 'processing') {
                        const progress = data.progress || 0;
                        setProgress(Math.min(99, progress));
                        if (data.message) showStatus(data.message);
                    }
                })
                .catch(() => {});
        }

        function showVideoCard() { document.getElementById('videoCard').classList.add('show'); }
        function hideVideoCard() { document.getElementById('videoCard').classList.remove('show'); }
        function showStatus(text) {
            document.getElementById('status').classList.add('show');
            document.getElementById('statusText').textContent = text;
        }
        function hideStatus() { document.getElementById('status').classList.remove('show'); }
        function setProgress(p) { document.getElementById('progressFill').style.width = p + '%'; }
        function showDownload(filename, url) {
            document.getElementById('downloadSection').classList.add('show');
            document.getElementById('downloadBtn').href = url;
            document.getElementById('downloadBtn').download = filename;
            document.getElementById('downloadText').textContent = filename;
        }
        function hideDownload() { document.getElementById('downloadSection').classList.remove('show'); }
        function showError(msg) {
            document.getElementById('error').textContent = msg;
            document.getElementById('error').classList.add('show');
        }
        function hideError() { document.getElementById('error').classList.remove('show'); }

        document.getElementById('url').addEventListener('keypress', e => { if (e.key === 'Enter') fetchInfo(); });
    </script>
</body>
</html>
'''


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_video_info(url):
    """Get video title and duration using yt-dlp."""
    cmd = [
        'yt-dlp',
        '--dump-json',
        '--no-playlist',
        '--js-runtimes', 'nodejs',
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise Exception(f"Failed to get video info: {result.stderr}")

    info = json.loads(result.stdout)
    return {
        'title': info.get('title', 'Unknown'),
        'duration': info.get('duration', 0),  # Duration in seconds
        'thumbnail': info.get('thumbnail', '')
    }


def download_and_convert(task_id, url, bitrate='320'):
    """Download YouTube video and convert to MP3."""
    try:
        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting download...'
            }

        # Create task-specific directory
        task_dir = DOWNLOAD_DIR / task_id
        task_dir.mkdir(exist_ok=True)

        # Output template
        output_template = str(task_dir / '%(title)s.%(ext)s')

        # Map bitrate to yt-dlp audio quality (0=best, 9=worst)
        # We'll use --audio-quality with specific bitrate instead
        bitrate_map = {
            '320': '0',   # Best quality
            '256': '1',
            '192': '2',
            '128': '5',
            '96': '7',
            '64': '9'     # Lowest quality
        }
        audio_quality = bitrate_map.get(bitrate, '0')

        # yt-dlp command with progress output
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', audio_quality,
            '--output', output_template,
            '--no-playlist',
            '--newline',  # Output progress on new lines for parsing
            '--progress',
            '--js-runtimes', 'nodejs',
        ]

        # Add FFmpeg location if available from imageio-ffmpeg
        if FFMPEG_PATH:
            cmd.extend(['--ffmpeg-location', str(Path(FFMPEG_PATH).parent)])

        cmd.append(url)

        with downloads_lock:
            active_downloads[task_id]['message'] = 'Downloading...'
            active_downloads[task_id]['progress'] = 0

        # Run yt-dlp with real-time progress capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Read output line by line for real-time progress
        last_lines = []
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            last_lines.append(line.strip())
            if len(last_lines) > 15:
                last_lines.pop(0)
            progress = parse_progress(line)
            if progress is not None:
                with downloads_lock:
                    active_downloads[task_id]['progress'] = progress
                    if progress < 100:
                        active_downloads[task_id]['message'] = f'Downloading... {progress:.1f}%'
                    else:
                        active_downloads[task_id]['message'] = 'Converting to MP3...'

        process.wait(timeout=300)

        if process.returncode != 0:
            error_detail = '\n'.join(last_lines[-5:]) if last_lines else 'No output captured'
            raise Exception(f"yt-dlp download failed:\n{error_detail}")

        # Find the MP3 file
        mp3_files = list(task_dir.glob('*.mp3'))
        if not mp3_files:
            raise Exception("No MP3 file was created")

        mp3_file = mp3_files[0]
        filename = mp3_file.name

        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'completed',
                'filename': filename,
                'filepath': str(mp3_file),
                'download_url': f'/download/{task_id}/{filename}'
            }

    except Exception as e:
        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'error',
                'error': str(e)
            }


def download_video(task_id, url, resolution='1080'):
    """Download YouTube video as MP4."""
    try:
        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting video download...'
            }

        # Create task-specific directory
        task_dir = DOWNLOAD_DIR / task_id
        task_dir.mkdir(exist_ok=True)

        # Output template
        output_template = str(task_dir / '%(title)s.%(ext)s')

        # Build format selector based on resolution
        # Try to get video+audio merged, fallback to best available
        format_selector = f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]/best'

        # yt-dlp command for video with progress output
        cmd = [
            'yt-dlp',
            '-f', format_selector,
            '--merge-output-format', 'mp4',
            '--postprocessor-args', 'ffmpeg:-c:v copy -c:a aac -b:a 192k',
            '--output', output_template,
            '--no-playlist',
            '--newline',  # Output progress on new lines for parsing
            '--progress',
            '--js-runtimes', 'nodejs',
        ]

        # Add FFmpeg location if available
        if FFMPEG_PATH:
            cmd.extend(['--ffmpeg-location', str(Path(FFMPEG_PATH).parent)])

        cmd.append(url)

        with downloads_lock:
            active_downloads[task_id]['message'] = 'Downloading video...'
            active_downloads[task_id]['progress'] = 0

        # Run yt-dlp with real-time progress capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Read output line by line for real-time progress
        last_lines = []
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            last_lines.append(line.strip())
            if len(last_lines) > 15:
                last_lines.pop(0)
            progress = parse_progress(line)
            if progress is not None:
                with downloads_lock:
                    active_downloads[task_id]['progress'] = progress
                    if progress < 100:
                        active_downloads[task_id]['message'] = f'Downloading video... {progress:.1f}%'
                    else:
                        active_downloads[task_id]['message'] = 'Merging video and audio...'

        process.wait(timeout=600)

        if process.returncode != 0:
            error_detail = '\n'.join(last_lines[-5:]) if last_lines else 'No output captured'
            raise Exception(f"yt-dlp download failed:\n{error_detail}")

        # Find the video file
        video_files = list(task_dir.glob('*.mp4')) + list(task_dir.glob('*.mkv')) + list(task_dir.glob('*.webm'))
        if not video_files:
            raise Exception("No video file was created")

        video_file = video_files[0]
        filename = video_file.name

        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'completed',
                'filename': filename,
                'filepath': str(video_file),
                'download_url': f'/download/{task_id}/{filename}'
            }

    except Exception as e:
        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'error',
                'error': str(e)
            }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


class RequestHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        # Serve main page
        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
            return
        
        # API: Get server status
        if path == '/api/status':
            with downloads_lock:
                active_count = sum(1 for d in active_downloads.values() 
                                   if d.get('status') == 'processing')
            self.send_json({'active_count': active_count})
            return
        
        # API: Check task status
        if path.startswith('/api/check/'):
            task_id = path.split('/')[-1]
            with downloads_lock:
                task = active_downloads.get(task_id, {'status': 'not_found'})
            self.send_json(task)
            return

        # API: Check if file exists for a task
        if path.startswith('/api/file-exists/'):
            task_id = path.split('/')[-1]
            task_dir = DOWNLOAD_DIR / task_id
            exists = task_dir.exists() and any(task_dir.iterdir()) if task_dir.exists() else False
            self.send_json({'exists': exists})
            return

        # Serve download files
        if path.startswith('/download/'):
            parts = path.split('/')
            if len(parts) >= 4:
                task_id = parts[2]
                filename = unquote('/'.join(parts[3:]))
                filepath = DOWNLOAD_DIR / task_id / filename
                
                if filepath.exists():
                    self.send_response(200)
                    # Determine content type based on extension
                    ext = filepath.suffix.lower()
                    content_types = {
                        '.mp3': 'audio/mpeg',
                        '.mp4': 'video/mp4',
                        '.mkv': 'video/x-matroska',
                        '.webm': 'video/webm',
                    }
                    content_type = content_types.get(ext, 'application/octet-stream')
                    self.send_header('Content-Type', content_type)
                    # Use RFC 5987 encoding for non-ASCII filenames
                    ascii_filename = filename.encode('ascii', 'ignore').decode('ascii') or f'download{ext}'
                    encoded_filename = quote(filename)
                    self.send_header('Content-Disposition',
                        f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}")
                    self.send_header('Content-Length', str(filepath.stat().st_size))
                    self.end_headers()

                    with open(filepath, 'rb') as f:
                        shutil.copyfileobj(f, self.wfile)
                    return
            
            self.send_error(404, 'File not found')
            return
        
        self.send_error(404, 'Not found')
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        # API: Get video info
        if self.path == '/api/info':
            try:
                data = json.loads(body)
                url = data.get('url', '')

                # Validate URL
                video_id = extract_video_id(url)
                if not video_id:
                    self.send_json({'error': 'Invalid YouTube URL'}, 400)
                    return

                # Get video info
                info = get_video_info(url)
                self.send_json(info)

            except json.JSONDecodeError:
                self.send_json({'error': 'Invalid JSON'}, 400)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
            return

        # API: Convert video
        if self.path == '/api/convert':
            try:
                data = json.loads(body)
                url = data.get('url', '')
                bitrate = data.get('bitrate', '320')

                # Validate URL
                video_id = extract_video_id(url)
                if not video_id:
                    self.send_json({'error': 'Invalid YouTube URL'}, 400)
                    return

                # Validate bitrate
                valid_bitrates = ['320', '256', '192', '128', '96', '64']
                if bitrate not in valid_bitrates:
                    bitrate = '320'

                # Create task
                task_id = str(uuid.uuid4())[:8]

                # Start conversion in background thread
                thread = threading.Thread(
                    target=download_and_convert,
                    args=(task_id, url, bitrate)
                )
                thread.daemon = True
                thread.start()

                self.send_json({'task_id': task_id, 'status': 'started'})

            except json.JSONDecodeError:
                self.send_json({'error': 'Invalid JSON'}, 400)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
            return

        # API: Convert video to MP4
        if self.path == '/api/convert-video':
            try:
                data = json.loads(body)
                url = data.get('url', '')
                resolution = data.get('resolution', '1080')

                # Validate URL
                video_id = extract_video_id(url)
                if not video_id:
                    self.send_json({'error': 'Invalid YouTube URL'}, 400)
                    return

                # Validate resolution
                valid_resolutions = ['2160', '1440', '1080', '720', '480', '360']
                if resolution not in valid_resolutions:
                    resolution = '1080'

                # Create task
                task_id = str(uuid.uuid4())[:8]

                # Start video download in background thread
                thread = threading.Thread(
                    target=download_video,
                    args=(task_id, url, resolution)
                )
                thread.daemon = True
                thread.start()

                self.send_json({'task_id': task_id, 'status': 'started'})

            except json.JSONDecodeError:
                self.send_json({'error': 'Invalid JSON'}, 400)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
            return

        self.send_error(404, 'Not found')
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_DELETE(self):
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # API: Delete a task's files
        if path.startswith('/api/delete/'):
            task_id = path.split('/')[-1]
            task_dir = DOWNLOAD_DIR / task_id

            try:
                if task_dir.exists():
                    shutil.rmtree(task_dir)

                with downloads_lock:
                    if task_id in active_downloads:
                        del active_downloads[task_id]

                self.send_json({'success': True, 'message': 'File deleted'})
            except Exception as e:
                self.send_json({'success': False, 'error': str(e)}, 500)
            return

        self.send_error(404, 'Not found')


def cleanup_old_downloads():
    """Clean up downloads older than 1 hour."""
    import time
    while True:
        time.sleep(3600)  # Check every hour
        try:
            cutoff = time.time() - 3600
            for task_dir in DOWNLOAD_DIR.iterdir():
                if task_dir.is_dir():
                    if task_dir.stat().st_mtime < cutoff:
                        shutil.rmtree(task_dir)
                        with downloads_lock:
                            task_id = task_dir.name
                            if task_id in active_downloads:
                                del active_downloads[task_id]
        except Exception:
            pass


def main():
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_downloads, daemon=True)
    cleanup_thread.start()
    
    # Start server
    server = ThreadedHTTPServer((HOST, PORT), RequestHandler)
    
    # Get local IP for display
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("=" * 50)
    print("ytdl-web - YouTube Downloader")
    print("=" * 50)
    print(f"\nServer running!")
    print(f"\nAccess the web interface at:")
    print(f"   Local:   http://localhost:{EXTERNAL_PORT}")
    print(f"   Network: http://{local_ip}:{EXTERNAL_PORT}")
    print(f"\nShare the Network URL with others on your network")
    print(f"\nPress Ctrl+C to stop the server\n")
    print("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        server.shutdown()


if __name__ == '__main__':
    main()
