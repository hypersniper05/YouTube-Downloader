#!/bin/sh
# YouTube changes its extraction regularly, which leaves the yt-dlp baked into the
# image stale and makes every download fail with "HTTP Error 403: Forbidden".
# Refreshing on start means a plain `docker compose restart` recovers from that,
# with no rebuild. Set YTDLP_AUTO_UPDATE=0 to pin to the version in the image.
if [ "${YTDLP_AUTO_UPDATE:-1}" = "1" ]; then
    echo "[entrypoint] Updating yt-dlp (set YTDLP_AUTO_UPDATE=0 to skip)..."
    if ! pip install --no-cache-dir -q -U "yt-dlp[default,deno]"; then
        echo "[entrypoint] WARNING: yt-dlp update failed (offline?). Using the version baked into the image."
    fi
fi

echo "[entrypoint] yt-dlp $(yt-dlp --version 2>/dev/null || echo 'MISSING')"
echo "[entrypoint] deno   $(deno --version 2>/dev/null | head -1 || echo 'MISSING')"

exec "$@"
