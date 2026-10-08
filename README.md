# <img src="static/icon-192.png" alt="" height="36"> YouTube Downloader

A self-hosted server that downloads YouTube videos as MP3 or MP4 files. It can transcribe the audio to text or SRT subtitles with Whisper, and it can split the vocals from the background with UVR. Use it from the web page, or let an AI assistant such as Claude use it through MCP.

| Download | Results |
|----------|---------|
| ![Download](assets/audio.png) | ![Results](assets/videos.png) |
| **History** | **Settings** |
| ![History](assets/history.png) | ![Settings](assets/settings.png) |

## Features

- MP3 audio (64-320 kbps) or MP4 video (360p-4K)
- Transcription to plain text or SRT subtitles with Whisper Large V3 Turbo
- Vocal separation with UVR MDX-Net: keep only the vocals, only the background, or both
- A Settings page to pick the Whisper and UVR models, and the GPU or CPU for each
- Live progress, and a download history that stays after a page refresh
- A job queue: 4 jobs run at the same time, and more jobs wait their turn
- An MCP server, so AI assistants can download and transcribe videos
- An optional login for the web page and for MCP
- Automatic deletion of files 1-2 hours after a job finishes

## Quick start

You need Docker. An NVIDIA GPU makes transcription and vocal separation much faster, but it is not necessary.

```bash
git clone https://github.com/hypersniper05/YouTube-Downloader.git
cd YouTube-Downloader
docker compose up -d
```

Open `http://localhost:6080`. Other devices on your network can use `http://YOUR_IP:6080`.

The default setup uses all NVIDIA GPUs. If you have no NVIDIA GPU, or the container does not start because of a GPU error, use the CPU setup instead:

```bash
docker compose -f docker-compose.cpu.yml up -d
```

## Use the web page

1. Select **Audio (MP3)** or **Video (MP4)**.
2. Paste a YouTube link and click **Get Info**.
3. Select the quality.
4. To get a transcript, select **Extract transcription**. To get an SRT file, also select **Include timestamps**.
5. To split the vocals from the background, select **Separate vocals and background**. Then select **Vocals only**, **Background only**, or **Both**.
6. Click **Download**.
7. When the job is done, click the buttons to save each file.

The **History** tab shows your past downloads. From there, you can download each file again or delete them.

## Separate vocals and background

The server uses the UVR MDX-Net models from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui). You always get the original file too.

| Option | You get |
|--------|---------|
| **Vocals only** | The voice without the music or background sound |
| **Background only** | The music and background sound without the voice, like karaoke |
| **Both** | One file of each |

For a video, each new file is an MP4 with the same picture. For audio, each new file is an MP3.

On a CPU with 2 cores, the separation takes about as long as the song itself. A GPU is much faster: a GTX 1650 separates a 3.5-minute song in about 30 seconds. The first job downloads the model, which is about 67 MB.

## Settings page

The **Settings** tab lets you choose:

- The Whisper model. Larger models are more accurate but slower.
- The UVR model. **Inst HQ 3** (the default) and **Inst HQ 4** are best at the background. **Kim Vocal 2** and **Voc FT** are best at the vocals. Each model makes both files.
- Where each model runs: **Automatic** (the GPU if there is one), **CPU**, or a specific GPU. The page lists every NVIDIA GPU it finds.

The settings are saved on the server and stay after a restart or an update.

## Use with an AI assistant (MCP)

The server has an MCP endpoint at `/mcp`. To add it to Claude Code, run:

```bash
claude mcp add --transport http youtube http://localhost:6080/mcp
```

Other MCP clients can connect to `http://YOUR_IP:6080/mcp` with the Streamable HTTP transport.

| Tool | What it does |
|------|--------------|
| `get_video_info` | Gets the title, the length, and the available qualities |
| `download_audio` | Downloads an MP3, with an optional transcript and vocal separation |
| `download_video` | Downloads an MP4, with an optional transcript and vocal separation |
| `transcribe_video` | Returns the transcript as SRT subtitles or plain text |
| `get_task_status` | Waits for a long job to finish |
| `get_transcript` | Returns the transcript text of a finished job |
| `list_downloads` | Lists the files on the server |
| `delete_download` | Deletes the files of a job |

To separate the audio, add `"remove_vocals": true`, `"remove_background": true`, or both to `download_audio` or `download_video`. The result then has a link for each file: `file` (the original), `vocals_file`, `background_file`, and `transcript_file` if you asked for a transcript.

A long job, such as the transcription of a long video, can take several minutes. If the job is not done in time, the tool returns a `task_id`. The assistant then uses `get_task_status` to wait for the result.

## Login

There is no login by default. Anyone who can reach port 6080 can use the server. This is fine on a trusted home network.

To add a login, set these values in `docker-compose.yml`:

| To protect | Set |
|------------|-----|
| The web page and `/api` | `WEB_AUTH_PASSWORD`. The user name is `admin`. Change it with `WEB_AUTH_USER`. |
| The MCP endpoint | `MCP_AUTH=token` and `MCP_AUTH_TOKEN` |

To make a strong token, run `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Then add the token to your MCP client:

```bash
claude mcp add --transport http youtube http://localhost:6080/mcp --header "Authorization: Bearer YOUR_TOKEN"
```

Download links (`/download/...`) do not need a login, so AI assistants can get the files. The links are random, and they expire after 1-2 hours. If you open the server to the internet, use HTTPS through a reverse proxy.

### OAuth (advanced)

Set `MCP_AUTH=oauth` to accept tokens from your own OAuth server, such as Keycloak or Authentik. Clients such as claude.ai connectors use this sign-in method. Also set these values:

| Setting | Value |
|---------|-------|
| `PUBLIC_BASE_URL` | The public URL of this server, for example `https://ytdl.example.com` |
| `MCP_OAUTH_ISSUER` | The URL of your OAuth server |
| `MCP_OAUTH_INTROSPECTION_URL` | The token introspection URL of your OAuth server |
| `MCP_OAUTH_CLIENT_ID`, `MCP_OAUTH_CLIENT_SECRET` | The client that this server uses to check tokens |
| `MCP_OAUTH_SCOPES` | Optional. The scopes that a token must have |

## Settings

Set these values in the `environment:` section of `docker-compose.yml`. The Whisper and UVR values are only the defaults: a choice that you save on the Settings page replaces them.

| Setting | Default | What it does |
|---------|---------|--------------|
| `WEB_AUTH_PASSWORD` | not set | Adds a login to the web page and `/api` |
| `WEB_AUTH_USER` | `admin` | The user name for the web login |
| `MCP_ENABLED` | `1` | Set to `0` to disable the MCP endpoint |
| `MCP_AUTH` | `off` | `off`, `token`, or `oauth` |
| `MCP_AUTH_TOKEN` | not set | The token for `MCP_AUTH=token` |
| `MAX_ACTIVE_JOBS` | `4` | The number of jobs that run at the same time. More jobs wait in a queue |
| `MAX_QUEUED_JOBS` | `100` | The maximum number of jobs that can wait in the queue |
| `MCP_ALLOWED_ORIGINS` | not set | More browser origins that can call `/mcp`, separated by commas |
| `PUBLIC_BASE_URL` | not set | The public URL for download links, if the server is behind a proxy |
| `YTDLP_AUTO_UPDATE` | `1` | Updates yt-dlp each time the container starts |
| `YTDLP_RETRIES` | `3` | The number of retries when YouTube refuses a download or yt-dlp crashes |
| `WHISPER_MODEL_ID` | `openai/whisper-large-v3-turbo` | The Whisper model. Smaller models, such as `openai/whisper-small`, are faster |
| `WHISPER_DEVICE` | `auto` | `auto` uses the first GPU if there is one. Also `cpu`, or a GPU such as `cuda:1` |
| `WHISPER_FALLBACK_MODEL_ID` | `openai/whisper-small` | The model used on the CPU when the GPU fails. It is about 2 times faster on a CPU than the default model |
| `UVR_MODEL` | `inst_hq_3` | The UVR model for vocal separation: `inst_hq_3`, `inst_hq_4`, `kim_vocal_2`, or `voc_ft` |
| `UVR_DEVICE` | `auto` | Like `WHISPER_DEVICE`, for vocal separation |
| `UVR_MAX_MINUTES` | `60` | The longest audio that vocal separation accepts. Longer audio gets an error, because it needs more memory |

To change the port, edit `ports:` in `docker-compose.yml`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `HTTP Error 403: Forbidden` | YouTube sometimes refuses a download link. The server tries again with a new link. If it still fails, run `docker compose restart` to update yt-dlp. |
| The status says "Waiting in queue" | Other jobs are running. The job starts automatically when one finishes. |
| The status says "Waiting for another AI job" | Transcription and vocal separation run one at a time. This one starts when the current one finishes. |
| The container does not start, with `nvidia-container-runtime-hook` or `SIGSEGV` in the error | Docker cannot use the GPU. Restart the computer. Until then, use `docker compose -f docker-compose.cpu.yml up -d`. |
| The download fails for one video | The video may be private, age-restricted, or blocked in your region. |
| The browser asks for a password | `WEB_AUTH_PASSWORD` is set. Log in as `admin`, or as the user in `WEB_AUTH_USER`. |
| An MCP client gets error 401 | `MCP_AUTH` is on. Add the token to the client. |
| Transcription is slow | Without a GPU, Whisper runs on the CPU. Use a smaller `WHISPER_MODEL_ID`. |
| "Transcription failed: Whisper returned an empty transcript", or the server restarts during a transcription | Some GPUs cannot run Whisper. The server then switches to the CPU by itself, with the faster `WHISPER_FALLBACK_MODEL_ID`. To skip the GPU, select **CPU** on the Settings page. |
| Vocal separation is slow | Without a GPU, UVR runs on the CPU and takes about as long as the audio. |
| The first transcription is slow | The first run downloads the Whisper model, which is about 1.6 GB. |
| Other devices cannot connect | Make sure that your firewall allows port 6080. |

## Update

```bash
git pull
docker compose up -d --build
```

The container updates yt-dlp each time it starts. If downloads stop working, run `docker compose restart` first.

## API

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/api/info` | POST | Gets video information. Body: `{"url": "..."}` |
| `/api/convert` | POST | Starts an MP3 job. Body: `url`, `bitrate`, `transcribe`, `timestamps`, `remove_vocals`, `remove_background` |
| `/api/convert-video` | POST | Starts an MP4 job. Body: `url`, `resolution`, `transcribe`, `timestamps`, `remove_vocals`, `remove_background` |
| `/api/check/:id` | GET | Gets the status of a job |
| `/api/settings` | GET, POST | Gets or saves the Settings page values: `whisper_model`, `whisper_device`, `uvr_model`, `uvr_device` |
| `/api/delete/:id` | DELETE | Deletes the files of a job |
| `/download/:id/:file` | GET | Downloads a finished file |
| `/mcp` | POST | The MCP endpoint |

If `WEB_AUTH_PASSWORD` is set, send the login with each `/api` call, for example `curl -u admin:PASSWORD ...`.

## Run without Docker

Install Python 3.11 and FFmpeg. Then run:

```bash
pip install "yt-dlp[default,deno]" torch "transformers>=4.40.0" accelerate onnxruntime
python server.py
```

For the GPU, install `torch` from `https://download.pytorch.org/whl/cu128`, and install `onnxruntime-gpu` instead of `onnxruntime`.

## Credits

Vocal separation uses the MDX-Net models of [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) (MIT License).

## Disclaimer

For personal use only. Obey the YouTube Terms of Service and copyright law.

## License

MIT License
