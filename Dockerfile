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

# Install Python dependencies
# PyTorch nightly with CUDA 12.8 support (required for sm_120 / RTX 5090 Blackwell)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    yt-dlp \
    "transformers>=4.40.0" \
    accelerate

# Copy application
COPY server.py .

# Create directories
RUN mkdir -p /app/downloads /app/whisper-model

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "server.py"]
