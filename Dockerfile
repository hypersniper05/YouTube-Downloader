# ytdl-web - YouTube Downloader
# Lightweight Python image with ffmpeg

FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir yt-dlp

# Copy application
COPY server.py .

# Create downloads directory
RUN mkdir -p /app/downloads

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "server.py"]
