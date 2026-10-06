# ytdl-web - YouTube Downloader with Whisper Transcription
# NVIDIA CUDA base for GPU-accelerated transcription

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, Node.js, and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        curl \
        ffmpeg && \
    # Install Node.js 22.x
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    # Set python3.11 as default
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    # Cleanup
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Ubuntu 22.04's pip (22.0.2) can no longer parse current PyPI metadata: its resolver crashes
# with "'int' object has no attribute 'lower'" while installing transformers. Upgrade it first.
RUN python -m pip install --no-cache-dir --upgrade pip && pip --version

# Install Python dependencies
# PyTorch nightly with CUDA 12.8 support (required for sm_120 / RTX 5090 Blackwell)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    accelerate

# yt-dlp goes stale quickly as YouTube changes; keep it in its own layer.
#
# YouTube serves an obfuscated JavaScript player challenge that must be executed to
# derive the signature / nsig values in a stream URL. Without a JS runtime and the
# solver, yt-dlp cannot build a usable URL and every download fails with
# "HTTP Error 403: Forbidden". Both pieces come from yt-dlp's own extras:
#   [default] -> yt-dlp-ejs, the solver yt-dlp feeds to the runtime (exact-pinned by yt-dlp)
#   [deno]    -> the Deno binary, as a platform-correct PyPI wheel
# Deno is preferred over Node because it sandboxes the untrusted player JS. Taking both
# from the extras keeps their versions consistent with the yt-dlp release and keeps this
# layer architecture-independent.
#
# Bump YTDLP_REFRESH (or build with --no-cache) to pull a newer yt-dlp.
ARG YTDLP_REFRESH=1
RUN pip install --no-cache-dir -U \
    "yt-dlp[default,deno]>=2026.8.19" && \
    yt-dlp --version && \
    deno --version

# Copy application
COPY server.py .

# Refresh yt-dlp on container start so a `docker compose restart` recovers from a
# YouTube change without a rebuild. Opt out with YTDLP_AUTO_UPDATE=0.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create directories
RUN mkdir -p /app/downloads /app/whisper-model

# Expose port
EXPOSE 8080

# Run server
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "server.py"]
