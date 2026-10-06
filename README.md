# ytdl-web

A self-hosted web server that downloads YouTube videos as MP3 audio or MP4 video files. Features a modern dark-mode UI, real-time progress tracking, download history, and AI-powered transcription using Whisper.

| Audio | Video | History |
|-------|-------|---------|
| ![Audio](assets/audio.png) | ![Video](assets/videos.png) | ![History](assets/history.png) |

## Features

- **Audio & Video Downloads** - Download as MP3 (audio) or MP4 (video)
- **Quality Selection** - Choose bitrate (64-320 kbps) or resolution (360p-4K)
- **AI Transcription** - Extract spoken text using Whisper Large V3 Turbo (GPU-accelerated)
- **Timestamps / SRT** - Optionally include timestamps in SRT format for captions
- **Real-time Progress** - Live progress bar with percentage updates
- **Download History** - Track past downloads with thumbnails, persists across page refreshes
- **Mobile Responsive** - Works great on phones and tablets
- **Multi-user Support** - Handle concurrent downloads
- **Video Preview** - Thumbnail, title, and duration before downloading
- **File Size Estimates** - See approximate file size before download
- **Auto-cleanup** - Old downloads removed after 1 hour
- **GPU Optimized** - Runs on NVIDIA GPUs via CUDA for fast transcription
- **MCP Server** - AI assistants such as Claude can download and transcribe videos over the [Model Context Protocol](#mcp-server-ai-assistants)

## Quick Start

### Docker Compose (Recommended)

Requires an NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

```bash
git clone https://github.com/hypersniper05/ytdl-web.git
cd ytdl-web
docker compose up -d
```

Open `http://localhost:6080` in your browser.

### Docker Run

```bash
docker build -t ytdl-web .
docker run -d -p 6080:8080 --gpus 1 --name ytdl-web ytdl-web
```

### Manual Installation

```bash
# Install ffmpeg
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: winget install FFmpeg

# yt-dlp needs a JavaScript runtime plus the player-challenge solver, or every
# download fails with "HTTP Error 403: Forbidden". Both come from these extras:
#   [default] -> yt-dlp-ejs, the solver
#   [deno]    -> the Deno runtime (sandboxes the untrusted player JS)
pip install "yt-dlp[default,deno]"
# For transcription support:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.40.0" accelerate

python server.py
```

## Usage

1. Open `http://localhost:6080` in your browser
2. Select **Audio (MP3)** or **Video (MP4)** tab
3. Paste a YouTube URL and click **Get Info**
4. Choose quality/bitrate from the dropdown
5. *(Optional)* Check **Extract transcription** to generate a text transcript
6. *(Optional)* Check **Include timestamps (SRT format)** for subtitle-ready output
7. Click **Download MP3** or **Download MP4**
8. Watch the real-time progress bar
9. Click the download button when complete - separate buttons for media and transcript

### Transcription

When **Extract transcription** is enabled:

- The audio is transcribed using [Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) running locally on your GPU
- **Without timestamps**: Produces a plain `.txt` file with the full transcription
- **With timestamps**: Produces an `.srt` file that can be used as subtitles/captions
- The Whisper model (~1.6 GB) downloads automatically on first use and is cached
- The model loads on demand and unloads after transcription to free GPU memory
- Transcription works with both MP3 and MP4 downloads (no separate audio extraction needed)
- Transcript download buttons also appear in the **History** tab

### History Tab

- View all your past downloads with real-time status
- **Persistent progress tracking** - downloads are saved to history immediately when started, so if you close or refresh the page mid-download, the item appears with a "Processing..." badge and automatically updates when complete
- Re-download files if still available
- Download transcripts for items that had transcription enabled
- Delete files from server
- Shows file type, quality, and date

## Audio Quality Options

| Bitrate | Quality | Use Case |
|---------|---------|----------|
| 320 kbps | Best | Audiophiles, Hi-Fi systems |
| 256 kbps | High | Quality listening |
| 192 kbps | Good | General use |
| 128 kbps | Standard | Casual listening, smaller files |
| 96 kbps | Low | Voice content, podcasts |
| 64 kbps | Smallest | Minimum quality, tiny files |

## Video Quality Options

| Resolution | Quality | Use Case |
|------------|---------|----------|
| 2160p (4K) | Ultra HD | Large screens, archival |
| 1440p (2K) | Quad HD | High-end monitors |
| 1080p | Full HD | Standard quality |
| 720p | HD | Good balance of size/quality |
| 480p | SD | Smaller files |
| 360p | Low | Minimum quality, mobile data |

## Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/shorts/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

## Network Access

To access from other devices on your network:

1. Find your IP: `hostname -I` (Linux/Mac) or `ipconfig` (Windows)
2. Share: `http://YOUR_IP:6080`
3. Ensure firewall allows port 6080

## Configuration

Change the port in `docker-compose.yml`:

```yaml
ports:
  - "0.0.0.0:YOUR_PORT:8080"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTERNAL_PORT` | `8080` | Port displayed in server startup message |
| `WHISPER_MODEL_ID` | `openai/whisper-large-v3-turbo` | Whisper model to use |
| `WHISPER_MODEL_DIR` | `/app/whisper-model` | Directory to cache model weights |
| `YTDLP_AUTO_UPDATE` | `1` | Refresh yt-dlp on every container start |
| `MCP_ENABLED` | `1` | Serve the MCP endpoint at `/mcp` |
| `MCP_AUTH` | `off` | `off`, `token` or `oauth` - see [Authorization](#authorization) |
| `MCP_AUTH_TOKEN` | - | Bearer token for `MCP_AUTH=token` |
| `PUBLIC_BASE_URL` | - | External URL, e.g. `https://ytdl.example.com`. Used for download links; required for `oauth` |
| `MCP_OAUTH_ISSUER` | - | Authorization server issuer URL (`oauth`) |
| `MCP_OAUTH_INTROSPECTION_URL` | - | Token introspection endpoint (`oauth`) |
| `MCP_OAUTH_CLIENT_ID`, `MCP_OAUTH_CLIENT_SECRET` | - | Credentials ytdl-web uses to call the introspection endpoint (`oauth`) |
| `MCP_OAUTH_SCOPES` | - | Space-separated scopes a token must have (`oauth`) |
| `MCP_ALLOWED_ORIGINS` | - | Comma-separated extra browser origins allowed to call `/mcp` |
| `MCP_MAX_ACTIVE_JOBS` | `4` | Jobs that may run at once before MCP clients get "Server busy" |

### Available Whisper Models

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| `openai/whisper-large-v3-turbo` | Fast | High | ~2 GB |
| `openai/whisper-large-v3` | Slow | Highest | ~3 GB |
| `openai/whisper-medium` | Medium | Good | ~1.5 GB |
| `openai/whisper-small` | Fast | Fair | ~1 GB |
| `openai/whisper-base` | Fastest | Basic | ~0.5 GB |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/info` | POST | Get video info (title, duration, thumbnail) |
| `/api/convert` | POST | Start MP3 conversion (supports `transcribe` and `timestamps` flags) |
| `/api/convert-video` | POST | Start MP4 download (supports `transcribe` and `timestamps` flags) |
| `/api/check/:id` | GET | Check task status/progress (includes `transcription_url` when available) |
| `/api/status` | GET | Get active conversion count |
| `/api/file-exists/:id` | GET | Check if download file exists |
| `/api/delete/:id` | DELETE | Delete a download |
| `/download/:id/:file` | GET | Download completed file (MP3, MP4, TXT, or SRT) |
| `/mcp` | POST | MCP server endpoint - see [MCP Server](#mcp-server-ai-assistants) |

## MCP Server (AI Assistants)

ytdl-web is also an [MCP](https://modelcontextprotocol.io) server, so AI assistants such as Claude can look up,
download and transcribe videos for you. It listens at `/mcp` on the same port as the web UI.

- **Protocol** - Streamable HTTP, revision `2026-07-28`, plus the earlier `initialize`-based revisions
  (`2025-03-26` to `2025-11-25`) that most clients still use. Stateless: no sessions are kept.
- **Long jobs** - Downloads and transcription run in the background. Each tool waits up to `wait_seconds`
  (default 45, max 600) and streams progress notifications while it waits. If the job is still running after
  that, the tool returns a `task_id`, and the assistant polls it with `get_task_status`. Transcription can take
  several minutes for a long video when Whisper runs on the CPU.

### Tools

| Tool | What it does |
|------|--------------|
| `get_video_info` | Title, channel, duration, thumbnail, and the resolutions and bitrates you can request |
| `download_audio` | MP3 at 64-320 kbps, optionally transcribed to plain text or SRT |
| `download_video` | MP4 at 360p-4K, optionally transcribed to plain text or SRT |
| `transcribe_video` | Transcript only: SRT subtitles (default) or plain text, returned inline with a link to the file |
| `get_task_status` | Check on, or wait for, a running job |
| `get_transcript` | Full transcript text of a finished job |
| `list_downloads` | Jobs whose files are still on the server, newest first |
| `delete_download` | Delete a finished job's files |

Every tool returns structured output that matches its published `outputSchema`, with direct download links.
Files are deleted 1-2 hours after the job finishes.

### Connecting a client

Claude Code:

```bash
claude mcp add --transport http ytdl http://localhost:6080/mcp
```

With a bearer token (see [Authorization](#authorization)):

```bash
claude mcp add --transport http ytdl http://localhost:6080/mcp --header "Authorization: Bearer YOUR_TOKEN"
```

Any MCP client that supports Streamable HTTP can connect to `http://<host>:6080/mcp`.

### Authorization

Off by default. Choose a mode with `MCP_AUTH`:

| Mode | Use it when | Settings |
|------|-------------|----------|
| `off` | The server stays on a trusted LAN | - |
| `token` | You configure the client yourself, e.g. Claude Code's `--header` | `MCP_AUTH_TOKEN` (32+ random characters) |
| `oauth` | The client discovers auth on its own, e.g. claude.ai custom connectors | `PUBLIC_BASE_URL`, `MCP_OAUTH_ISSUER`, `MCP_OAUTH_INTROSPECTION_URL`, `MCP_OAUTH_CLIENT_ID`, `MCP_OAUTH_CLIENT_SECRET`, optionally `MCP_OAUTH_SCOPES` |

Generate a token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

In `oauth` mode ytdl-web is an OAuth 2.1 *resource server*; you run the authorization server (Keycloak,
Authentik, Auth0, ...). ytdl-web:

- publishes Protected Resource Metadata (RFC 9728) at `/.well-known/oauth-protected-resource/mcp`
- validates each token at the authorization server's introspection endpoint (RFC 7662), caching the answer
  for up to 60 seconds
- accepts only tokens issued for `PUBLIC_BASE_URL/mcp` or `PUBLIC_BASE_URL` (audience check, RFC 8707)
- answers `403 insufficient_scope` when a token lacks a scope listed in `MCP_OAUTH_SCOPES`

If the chosen mode is missing a setting, `/mcp` refuses every request and the startup log says what is
missing. The web UI keeps working.

> **`MCP_AUTH` protects `/mcp` only.** The web UI, `/api/*` and `/download/` have no authentication, as
> before. Before exposing the server beyond your LAN, put it behind a reverse proxy that forwards only `/mcp`,
> `/.well-known/oauth-protected-resource*` and `/download/`. Download links carry no credentials so that MCP
> clients can fetch them directly; they use random task ids and expire after 1-2 hours.

Browser-based clients such as MCP Inspector must come from a loopback origin, from `PUBLIC_BASE_URL`, or from
an origin in `MCP_ALLOWED_ORIGINS`. Any other origin gets 403, which blocks DNS-rebinding attacks. Native
clients send no `Origin` header and are not affected.

## GPU Requirements

- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host
- Minimum ~2 GB VRAM for transcription (only used when transcription is active)

Transcription falls back to CPU if no GPU is available, but will be significantly slower.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | Change port in docker-compose.yml |
| Can't access from other devices | Check firewall, ensure same network |
| Conversion fails | Video may be private, age-restricted, or unavailable |
| `HTTP Error 403: Forbidden` | yt-dlp is stale or missing its JS player-challenge solver. Rebuild with a bumped refresh arg - see [Updating](#updating) |
| `No supported JavaScript runtime could be found` | No Deno/Node in the environment. The Docker image ships Deno; for a manual install see [Manual Installation](#manual-installation) |
| Still 403 after a restart | Not a stale-yt-dlp problem. YouTube also blocks some datacenter/VPN IPs and gates certain videos behind sign-in, which needs cookies or a PO token — neither is wired up here |
| No audio in MP4 | Already fixed - audio is converted to AAC |
| Progress stuck | Large files take time, check network connection |
| Transcription fails with CUDA error | Ensure PyTorch version matches your GPU architecture |
| Model download slow | First run downloads ~1.6 GB; subsequent runs use cache |
| GPU not detected | Check `nvidia-smi` works and Container Toolkit is installed |
| MCP client gets 401 | `MCP_AUTH` is on: send `Authorization: Bearer <token>` (`token`), or let the client complete the OAuth flow (`oauth`) |
| MCP tool returns `"status": "processing"` | Normal for long jobs: the assistant calls `get_task_status` with the `task_id` |

## Updating

```bash
git pull
docker compose up -d --build
```

### Keeping yt-dlp current

YouTube changes its extraction every few weeks, which leaves a stale yt-dlp failing
every download with `403 Forbidden`. The container refreshes yt-dlp on **every start**,
so recovering is just:

```bash
docker compose restart
```

Set `YTDLP_AUTO_UPDATE=0` in `docker-compose.yml` to disable that and pin to whatever
version is baked into the image. If PyPI is unreachable at boot the update is skipped
with a warning and the container still starts. The entrypoint logs the yt-dlp and Deno
versions it ends up with — check with `docker logs ytdl-web`.

To refresh the baked-in version too, note that a plain `--build` will **not** do it:
the yt-dlp layer is a cache hit. Bump `YTDLP_REFRESH` to any new value:

```bash
docker compose build --build-arg YTDLP_REFRESH=2
docker compose up -d
```

Use a different number each time (3, 4, ...). Works as-is in PowerShell, CMD and bash.

## Tech Stack

- **Backend**: Python 3.11, http.server
- **Video Processing**: yt-dlp, FFmpeg
- **Transcription**: OpenAI Whisper Large V3 Turbo (via HuggingFace Transformers)
- **GPU**: NVIDIA CUDA 12.8, PyTorch
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Storage**: Local filesystem, localStorage for history

## Disclaimer

For personal use only. Respect YouTube's Terms of Service and copyright laws.

## License

MIT License
