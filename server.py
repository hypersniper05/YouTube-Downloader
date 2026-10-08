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
import gc
import collections
import time
import select
import socket
import hmac
import base64
import hashlib
import tempfile
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse, unquote, quote, quote_plus, urlencode
import subprocess
import shutil

# Get FFmpeg path from imageio-ffmpeg (cross-platform)
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[OK] FFmpeg found: {FFMPEG_PATH}")
except ImportError:
    FFMPEG_PATH = None
    if shutil.which('ffmpeg'):
        print(f"[OK] FFmpeg found: {shutil.which('ffmpeg')}")
    else:
        print("[WARN] FFmpeg not found. Install ffmpeg, or run: pip install imageio-ffmpeg")

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 8080
EXTERNAL_PORT = int(os.environ.get('EXTERNAL_PORT', PORT))
DOWNLOAD_DIR = Path(__file__).parent / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Optional HTTP Basic auth for the web UI and /api - off unless WEB_AUTH_PASSWORD is set.
# /download/ stays open so MCP clients can fetch the links they are handed (random task ids,
# deleted after 1-2 hours); /mcp has its own MCP_AUTH setting.
WEB_AUTH_USER = os.environ.get('WEB_AUTH_USER', 'admin')
WEB_AUTH_PASSWORD = os.environ.get('WEB_AUTH_PASSWORD', '')

# Track active downloads
active_downloads = {}
downloads_lock = threading.Lock()

# --- Settings --------------------------------------------------------------------------
# Environment variables give the defaults. The Settings page saves overrides to SETTINGS_FILE,
# which sits on the model volume so it survives rebuilds.
WHISPER_MODEL_DIR = os.environ.get('WHISPER_MODEL_DIR', str(Path(__file__).parent / 'whisper-model'))
UVR_MODEL_DIR = os.environ.get('UVR_MODEL_DIR', str(Path(WHISPER_MODEL_DIR) / 'uvr'))
SETTINGS_FILE = Path(os.environ.get('SETTINGS_FILE', str(Path(WHISPER_MODEL_DIR) / 'settings.json')))
# Used on the CPU after the GPU failed: a server sized for GPU transcription would otherwise
# take minutes per video. A CPU-only install keeps the chosen model.
WHISPER_FALLBACK_MODEL_ID = os.environ.get('WHISPER_FALLBACK_MODEL_ID', 'openai/whisper-small')

WHISPER_MODELS = [
    ('openai/whisper-large-v3-turbo', 'Large v3 Turbo (recommended)'),
    ('openai/whisper-large-v3', 'Large v3 (most accurate, slowest)'),
    ('openai/whisper-medium', 'Medium'),
    ('openai/whisper-small', 'Small (fast on a CPU)'),
    ('openai/whisper-base', 'Base (fastest, least accurate)'),
]
def _device_setting(value):
    """auto, cpu or cuda:N. 'cuda' (the 1.2.x WHISPER_DEVICE value) means the first GPU."""
    value = value.strip().lower()
    return 'cuda:0' if value == 'cuda' else value


# auto: the first GPU when PyTorch sees one, else the CPU. cpu / cuda:N: always that device.
DEFAULT_SETTINGS = {
    'whisper_model': os.environ.get('WHISPER_MODEL_ID', 'openai/whisper-large-v3-turbo'),
    'whisper_device': _device_setting(os.environ.get('WHISPER_DEVICE', 'auto')),
    'uvr_model': os.environ.get('UVR_MODEL', 'inst_hq_3'),
    'uvr_device': _device_setting(os.environ.get('UVR_DEVICE', 'auto')),
}
_settings_lock = threading.Lock()       # guards SETTINGS
_settings_save_lock = threading.Lock()  # one save at a time: check, write, apply


def _read_settings():
    settings = dict(DEFAULT_SETTINGS)
    if not SETTINGS_FILE.exists():
        return settings
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        settings.update({k: v for k, v in saved.items() if k in settings and isinstance(v, str)})
    except (OSError, ValueError, AttributeError) as e:
        print(f"[Settings] Ignoring {SETTINGS_FILE}, which cannot be read ({e}); using the defaults")
    for key in ('whisper_device', 'uvr_device'):
        settings[key] = _device_setting(settings[key])
    return settings


SETTINGS = _read_settings()


def setting(name):
    with _settings_lock:
        return SETTINGS[name]


_gpus = None


def detect_gpus():
    """Every NVIDIA GPU PyTorch can use: [{'id': 'cuda:0', 'name': ..., 'memory_gb': ...}]."""
    global _gpus
    if _gpus is None:
        found = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    found.append({'id': f'cuda:{i}', 'name': props.name,
                                  'memory_gb': round(props.total_memory / 2**30, 1)})
        except Exception as e:
            print(f"[GPU] Detection failed: {e}")
        _gpus = found
        print("[GPU] " + (', '.join(f"{g['id']} {g['name']} ({g['memory_gb']} GB)" for g in found)
                          or 'No NVIDIA GPU found: AI models run on the CPU'))
    return _gpus


def resolve_device(choice):
    """Turn a device setting (auto, cpu, cuda, cuda:N) into 'cpu' or 'cuda:N'."""
    gpus = [g['id'] for g in detect_gpus()]
    if choice == 'cpu' or not gpus:
        return 'cpu'
    if choice in gpus:
        return choice
    if choice not in ('auto', 'cuda'):
        print(f"[GPU] {choice} is not available; using {gpus[0]}")
    return gpus[0]


def settings_options():
    """The choices the Settings page offers."""
    devices = [{'id': 'auto', 'label': 'Automatic: GPU if there is one, else CPU'},
               {'id': 'cpu', 'label': 'CPU'}]
    gpus = detect_gpus()
    devices += [{'id': g['id'], 'label': f"GPU {g['id'][5:]}: {g['name']} ({g['memory_gb']} GB)"}
                for g in gpus]
    # A saved or configured GPU that is gone stays selectable, so saving the page does not
    # quietly turn it into Automatic.
    with _settings_lock:
        configured = (SETTINGS['whisper_device'], SETTINGS['uvr_device'], SETTINGS['whisper_model'])
    for device in (DEFAULT_SETTINGS['whisper_device'], DEFAULT_SETTINGS['uvr_device'], *configured[:2]):
        if device not in {d['id'] for d in devices}:
            devices.append({'id': device, 'label': f'{device} (not found: uses the first GPU or the CPU)'})
    whisper = [{'id': m, 'label': label} for m, label in WHISPER_MODELS]
    for model in (DEFAULT_SETTINGS['whisper_model'], configured[2]):
        if model not in {w['id'] for w in whisper}:
            whisper.append({'id': model, 'label': model})
    uvr = [{'id': key, 'label': f"{m['label'].replace('UVR-MDX-NET ', '')}: {m['primary']}"
                                 + (' (recommended)' if key == 'inst_hq_3' else '')}
           for key, m in UVR_MODELS.items()]
    return {'whisper_models': whisper, 'uvr_models': uvr, 'devices': devices, 'gpus': gpus}


def save_settings(changes):
    """Check and store new settings. Returns an error message, or None.

    Only values that differ from the defaults (the environment variables) go into the file, so
    a field the user leaves alone keeps following docker-compose.yml.
    """
    global _whisper_gpu_failed
    with _settings_save_lock:
        options = settings_options()
        allowed = {'whisper_model': options['whisper_models'], 'uvr_model': options['uvr_models'],
                   'whisper_device': options['devices'], 'uvr_device': options['devices']}
        for name, value in changes.items():
            if name not in allowed:
                return f'Unknown setting: {name}'
            if not isinstance(value, str) or value not in {o['id'] for o in allowed[name]}:
                return f'{name}: {value!r} is not one of the options'
        with _settings_lock:
            new = {**SETTINGS, **changes}
            old_device = SETTINGS['whisper_device']
        stored = {k: v for k, v in new.items() if v != DEFAULT_SETTINGS[k]}
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=SETTINGS_FILE.parent, prefix='.settings-', suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(stored, f, indent=2)
                os.replace(tmp, SETTINGS_FILE)
            finally:
                Path(tmp).unlink(missing_ok=True)
        except OSError as e:
            return f'Could not save the settings: {e}'
        with _settings_lock:
            SETTINGS.update(new)
    if new['whisper_device'] != old_device:
        # A new device choice is a fresh start for the GPU. A GPU transcription that is running
        # now removes its own marker when it ends; if it crashes the server, the marker must stay.
        _whisper_gpu_failed = False
        if not (model_lock.locked() and _whisper_on_gpu):
            _set_gpu_marker(None)
    print(f"[Settings] {', '.join(f'{k}={v}' for k, v in changes.items())}")
    return None


def settings_payload():
    """GET /api/settings: the current settings, the defaults, and the choices."""
    with _settings_lock:
        current = dict(SETTINGS)
    running_on_gpu = model_lock.locked() and _whisper_on_gpu  # its marker is not a failure
    return {
        'settings': current,
        'defaults': DEFAULT_SETTINGS,
        'options': settings_options(),
        'uvr_gpu_supported': uvr_gpu_supported(),
        'whisper_gpu_failed': _whisper_gpu_failed or (_WHISPER_GPU_MARKER.exists() and not running_on_gpu),
    }


# Whisper transcription state
# Exists while a GPU transcription runs, and stays when the GPU gave an empty transcript that the
# CPU did not. If it is there when the model loads, a GPU transcription failed or took the whole
# server down (seen on one server: short clips came back empty and long ones crashed the
# process), so auto mode uses the CPU instead.
_WHISPER_GPU_MARKER = Path(WHISPER_MODEL_DIR) / 'gpu-transcription-failed'
# One AI model runs at a time, Whisper or UVR: the pipelines are not thread-safe, and a
# CPU-only server has the RAM for one of them, not both.
model_lock = threading.Lock()
_whisper_pipe = None
_whisper_key = None  # (model id, device) of the loaded pipeline
_whisper_on_gpu = False
_whisper_desc = ''  # Where the loaded model runs, for log lines and error messages
_whisper_gpu_failed = False  # The GPU failed (empty transcript or crash); use the CPU
_whisper_refcount = 0
_whisper_refcount_lock = threading.Lock()


def _whisper_device():
    """Pick the device for the next model load: 'cpu' or 'cuda:N'."""
    global _whisper_gpu_failed
    choice = setting('whisper_device')
    device = resolve_device(choice)
    if device == 'cpu' or choice != 'auto':
        return device  # a GPU chosen by name is used even after a failure
    if _whisper_gpu_failed:
        return 'cpu'
    if _WHISPER_GPU_MARKER.exists():
        _whisper_gpu_failed = True
        print(f"[Whisper] A GPU transcription failed before: it crashed the server or gave no text "
              f"({_WHISPER_GPU_MARKER}). Whisper uses the CPU. Pick a GPU on the Settings page to "
              "try it again.")
        return 'cpu'
    return device


def _set_gpu_marker(note):
    """Write note into the GPU marker file, or remove the file when note is None."""
    try:
        if note is None:
            _WHISPER_GPU_MARKER.unlink(missing_ok=True)
        else:
            _WHISPER_GPU_MARKER.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {note}\n")
    except OSError:
        pass


def _load_whisper(device=None, fallback=False):
    """Load the Whisper model on demand. Returns the ASR pipeline.

    fallback=True (or a GPU that failed before) loads WHISPER_FALLBACK_MODEL_ID on the CPU.
    """
    global _whisper_pipe, _whisper_key, _whisper_on_gpu, _whisper_desc
    if device is None:
        device = _whisper_device()
        fallback = device == "cpu" and _whisper_gpu_failed
    model_id = WHISPER_FALLBACK_MODEL_ID if fallback else setting('whisper_model')
    if _whisper_pipe is not None:
        if _whisper_key == (model_id, device):
            return _whisper_pipe
        _unload_whisper()  # the settings changed since it loaded

    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    on_gpu = device.startswith("cuda")
    if on_gpu:
        hardware = torch.cuda.get_device_name(torch.device(device).index or 0)
    else:
        hardware = f"{torch.get_num_threads()} threads, {torch.backends.cpu.get_cpu_capability()}"
    # GTX 16xx cards get float16 wrong (known from Stable Diffusion, and every Whisper run on a
    # GTX 1650 came back empty), so they run in float32 like the CPU.
    dtype = torch.float16 if on_gpu and 'GTX 16' not in hardware else torch.float32
    _whisper_desc = (f"{model_id} on {device} ({hardware}), {str(dtype).replace('torch.', '')}; "
                     f"torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"[Whisper] Loading model {_whisper_desc}...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=WHISPER_MODEL_DIR,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=WHISPER_MODEL_DIR)

    _whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )
    _whisper_key = (model_id, device)
    _whisper_on_gpu = on_gpu
    print("[Whisper] Model loaded successfully.")
    return _whisper_pipe


_whisper_unload_lock = threading.Lock()  # whisper_release and separate_media can both unload


def _unload_whisper():
    """Unload the Whisper model to free GPU/RAM."""
    global _whisper_pipe, _whisper_key, _whisper_on_gpu
    with _whisper_unload_lock:
        if _whisper_pipe is None:
            return
        print("[Whisper] Unloading model to free resources...")
        _whisper_pipe = None
        _whisper_key = None
        _whisper_on_gpu = False
    _free_gpu_memory()
    print("[Whisper] Model unloaded.")


def _is_gpu_error(e):
    """True for errors from the GPU itself (out of memory, CUDA, cuDNN, cuBLAS), as opposed to a
    failed model download, a bad model name or a file that cannot be decoded."""
    if type(e).__name__ in ('OutOfMemoryError', 'AcceleratorError'):
        return True
    text = str(e).lower()
    return isinstance(e, RuntimeError) and any(
        word in text for word in ('cuda', 'cudnn', 'cublas', 'out of memory'))


def _free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def whisper_acquire():
    """Increment ref count - call before starting a transcription task."""
    global _whisper_refcount
    with _whisper_refcount_lock:
        _whisper_refcount += 1


def whisper_release():
    """Decrement ref count and unload model if no more pending work."""
    global _whisper_refcount
    with _whisper_refcount_lock:
        _whisper_refcount = max(0, _whisper_refcount - 1)
        if _whisper_refcount == 0:
            _unload_whisper()


def _seconds_to_srt_time(s):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    if s is None:
        s = 0.0
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = int(s % 60)
    millis = int((s - int(s)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_audio_file(audio_path, timestamps=False, on_start=None):
    """Transcribe an audio file using Whisper. Queued via model_lock (one at a time).
    If timestamps=True, returns SRT-formatted string. Otherwise plain text.
    on_start() is called once this file's turn comes."""
    global _whisper_gpu_failed

    def has_text(result):
        return bool((result.get("text") or "").strip())

    def run(pipe):
        if not _whisper_on_gpu:
            return pipe(audio_path, return_timestamps=True)
        _set_gpu_marker(f'transcription running on {_whisper_desc}')
        try:
            return pipe(audio_path, return_timestamps=True)
        finally:
            _set_gpu_marker(None)

    with model_lock:
        if on_start is not None:
            on_start()
        device = _whisper_device()
        try:
            pipe = _load_whisper(device, fallback=device == "cpu" and _whisper_gpu_failed)
            result = run(pipe)
            problem = None if has_text(result) else 'empty transcript'
        except Exception as e:
            if not device.startswith("cuda") or setting('whisper_device') != 'auto' or not _is_gpu_error(e):
                raise
            result, problem = {}, f'error ({e})'
        if problem and device.startswith("cuda") and setting('whisper_device') == 'auto':
            # Seen on one server (GTX 1650): GPU runs finished without an error but with no
            # text, and long ones crashed the process. Out of memory on a small card lands here.
            gpu = _whisper_desc
            print(f"[Whisper] {problem[:1].upper() + problem[1:]} on {gpu}; retrying on the CPU")
            pipe = None
            _unload_whisper()
            _free_gpu_memory()  # also when the GPU model never finished loading
            pipe = _load_whisper(device="cpu", fallback=True)
            result = run(pipe)
            if has_text(result):
                _whisper_gpu_failed = True
                _set_gpu_marker(f'{problem} on {gpu}; the CPU worked')
                print("[Whisper] The CPU worked, so Whisper uses the CPU from now on. Pick a GPU on "
                      "the Settings page to try it again.")
        where = _whisper_desc

    # A job must not report success with a blank transcript file. The callers store this as
    # transcription_error, which the web page and MCP results show.
    if not has_text(result):
        raise RuntimeError(f"Whisper returned an empty transcript ({where})")

    if not timestamps:
        return result["text"]

    # Build SRT format from chunks
    chunks = result.get("chunks", [])
    if not chunks:
        return result["text"]

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        start, end = chunk["timestamp"]
        if end is None:
            end = start + 5.0
        lines.append(str(i))
        lines.append(f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}")
        lines.append(chunk["text"].strip())
        lines.append("")
    return "\n".join(lines)


# --- UVR vocal separation (MDX-Net) ------------------------------------------------------
UVR_MODEL_URL = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
UVR_SAMPLE_RATE = 44100  # every MDX-Net model takes 44.1 kHz stereo
UVR_HOP = 1024           # STFT hop, the same for every MDX-Net model
# Settings from UVR's model_data.json, which is keyed by the md5 of the last 10,000 KiB of the
# .onnx file (md5_tail, checked after each download). dim_t 8 means 2**8 = 256 STFT frames per
# window (about 6 s). The network outputs the 'primary' stem; the other stem is the mix minus it.
UVR_MODELS = {
    'inst_hq_3': {'file': 'UVR-MDX-NET-Inst_HQ_3.onnx', 'label': 'UVR-MDX-NET Inst HQ 3',
                  'size': 66759214, 'md5_tail': '55657dd70583b0fedfba5f67df11d711',
                  'compensate': 1.022, 'dim_f': 3072, 'dim_t': 8, 'n_fft': 6144, 'primary': 'background'},
    'inst_hq_4': {'file': 'UVR-MDX-NET-Inst_HQ_4.onnx', 'label': 'UVR-MDX-NET Inst HQ 4',
                  'size': 59074342, 'md5_tail': '0f2a6bc5b49d87d64728ee40e23bceb1',
                  'compensate': 1.019, 'dim_f': 2560, 'dim_t': 8, 'n_fft': 5120, 'primary': 'background'},
    'kim_vocal_2': {'file': 'Kim_Vocal_2.onnx', 'label': 'Kim Vocal 2',
                    'size': 66759214, 'md5_tail': '970b3f9492014d18fefeedfe4773cb42',
                    'compensate': 1.009, 'dim_f': 3072, 'dim_t': 8, 'n_fft': 7680, 'primary': 'vocals'},
    'voc_ft': {'file': 'UVR-MDX-NET-Voc_FT.onnx', 'label': 'UVR-MDX-NET Voc FT',
               'size': 66762490, 'md5_tail': '77d07b2667ddf05b9e3175941b4454a0',
               'compensate': 1.021, 'dim_f': 3072, 'dim_t': 8, 'n_fft': 7680, 'primary': 'vocals'},
}
for _uvr_model in UVR_MODELS.values():
    _uvr_model['url'] = UVR_MODEL_URL + _uvr_model['file']
del _uvr_model
_uvr_download_lock = threading.Lock()


def uvr_cpu_threads():
    """CPU threads to use: OMP_NUM_THREADS if set, else the container's CPU quota, else all cores.

    onnxruntime's default (one thread per host core) was 1.7x slower under a 2-CPU quota."""
    try:
        threads = int(os.environ.get('OMP_NUM_THREADS', '0'))
        if threads > 0:
            return threads
    except ValueError:
        pass
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cores = os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
        if quota != 'max':
            cores = min(cores, max(1, -(-int(quota) // int(period))))  # round up: 1.5 CPUs -> 2
    except (OSError, ValueError):
        pass
    return cores


def _uvr_md5_tail(path):
    """md5 of the last 10,000 KiB of a file, the key UVR uses for model settings."""
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - 10000 * 1024))
        return hashlib.md5(f.read()).hexdigest()


def uvr_model_path(key, model_dir):
    """Path of the model's .onnx file in model_dir. Downloads it on first use."""
    model = UVR_MODELS[key]
    path = Path(model_dir) / model['file']
    if path.exists():
        return path
    with _uvr_download_lock:  # one download at a time; another job may have just finished it
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + '.part')
        print(f"[UVR] Downloading {model['file']} ({model['size'] / 1e6:.0f} MB)...")
        started = time.monotonic()
        try:
            request = urllib.request.Request(model['url'], headers={'User-Agent': 'ytdl-web'})
            with urllib.request.urlopen(request, timeout=60) as response, open(part, 'wb') as f:
                shutil.copyfileobj(response, f, 1 << 20)
            if _uvr_md5_tail(part) != model['md5_tail']:
                raise RuntimeError(f"Download of {model['file']} is damaged (checksum mismatch)")
            os.replace(part, path)  # atomic: the model file is complete or absent
        finally:
            part.unlink(missing_ok=True)
        print(f"[UVR] Downloaded {model['file']} in {time.monotonic() - started:.0f} s")
    return path


def _uvr_session(path, device):
    """onnxruntime session on 'cpu' or 'cuda:N'. Raises rather than quietly using the CPU."""
    on_gpu = device.startswith('cuda')
    if on_gpu:
        # Import torch first: it loads the CUDA and cuDNN libraries that onnxruntime-gpu needs.
        import torch
        index = int(device.partition(':')[2] or 0)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Vocal separation on {device}: PyTorch sees "
                               f"{torch.cuda.device_count()} GPU(s)")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only
    options.intra_op_num_threads = uvr_cpu_threads()
    options.inter_op_num_threads = 1
    # A window needs about 1.2 GB of scratch memory. With onnxruntime's arena and memory pattern
    # the process held 3.1 GB from the second window on; without them it peaks at 1.9 GB and
    # gives the memory back between windows, at the same speed.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    if not on_gpu:
        return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in available:
        raise RuntimeError(f"Vocal separation on {device} needs onnxruntime-gpu; onnxruntime "
                           f"{ort.__version__} only has {', '.join(available)}")
    cuda = {'device_id': index,
            # Quick algorithm choice and no arena over-allocation: the GPU may be shared.
            'cudnn_conv_algo_search': 'HEURISTIC', 'arena_extend_strategy': 'kSameAsRequested'}
    session = ort.InferenceSession(str(path), options,
                                   providers=[('CUDAExecutionProvider', cuda), 'CPUExecutionProvider'])
    if 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError(f"onnxruntime {ort.__version__} could not use {device} (see its error "
                           "above; usually CUDA or cuDNN 9 for CUDA 12 did not load)")
    return session


def _uvr_run_windows(session, model, waves, hann):
    """Run the network on float32 [b, 2, window] waves. Returns its stem (before compensate),
    same shape, as a numpy array."""
    import numpy as np
    import torch

    n_fft, dim_f, dim_t = model['n_fft'], model['dim_f'], 2 ** model['dim_t']
    n_bins = n_fft // 2 + 1
    window = waves.shape[-1]
    x = torch.from_numpy(waves).to(hann.device).reshape(-1, window)
    spec = torch.view_as_real(torch.stft(x, n_fft, UVR_HOP, window=hann, center=True, return_complex=True))
    # [b*2, bins, frames, re/im] -> [b, 4, dim_f, frames], channels L.re, L.im, R.re, R.im
    spec = spec.permute(0, 3, 1, 2).reshape(-1, 4, n_bins, dim_t)[:, :, :dim_f].contiguous()
    spec[:, :, :3] = 0  # UVR silences the 3 lowest bins before the network
    out = session.run(None, {session.get_inputs()[0].name: spec.cpu().numpy()})[0]
    out = torch.from_numpy(out).to(hann.device)
    out = torch.nn.functional.pad(out, (0, 0, 0, n_bins - dim_f))  # the bins above dim_f stay 0
    out = out.reshape(-1, 2, n_bins, dim_t).permute(0, 2, 3, 1).contiguous()
    wav = torch.istft(torch.view_as_complex(out), n_fft, UVR_HOP, window=hann, center=True, length=window)
    wav = wav.reshape(-1, 2, window).cpu().numpy()
    if not np.isfinite(wav).all():
        raise RuntimeError(f"{model['label']} returned invalid values (NaN) on {hann.device}")
    return wav


def uvr_separate(mix, key, model_dir, device='cpu', progress=None, batch_size=1):
    """Split a float32 [2, n] mix at 44.1 kHz into (vocals, background) with an MDX-Net model.
    The mix array itself becomes one of the two stems, to save memory.

    device is 'cpu' or 'cuda:N'. progress(fraction) is called as windows finish. The network
    runs on batch_size windows (about 6 s of audio each) at a time, so its memory use does not
    grow with the length. Each window needs about 1.2 GB of scratch memory (RAM or VRAM), and a
    bigger batch was no faster on the CPU."""
    import numpy as np
    import torch

    model = UVR_MODELS[key]
    mix = np.asarray(mix, dtype=np.float32)
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError(f"mix must be a [2, n] array, got shape {mix.shape}")
    n = mix.shape[1]
    trim = model['n_fft'] // 2                  # samples at each window edge spoiled by STFT padding
    window = UVR_HOP * (2 ** model['dim_t'] - 1)  # samples per window: exactly dim_t STFT frames
    step = window - 2 * trim                    # samples each window keeps: its middle
    count = max(1, -(-n // step))
    started = time.monotonic()
    session = _uvr_session(uvr_model_path(key, model_dir), device)
    if device.startswith('cuda'):
        where = f"{device} ({torch.cuda.get_device_name(torch.device(device))})"
    else:
        where = f"cpu ({uvr_cpu_threads()} threads)"
    print(f"[UVR] Separating {n / UVR_SAMPLE_RATE:.0f} s of audio with {model['label']} on {where}...")

    primary = np.empty_like(mix)
    try:
        hann = torch.hann_window(model['n_fft'], periodic=True, device=torch.device(device))
        with torch.inference_mode():
            for first in range(0, count, batch_size):
                ks = range(first, min(first + batch_size, count))
                # Window k covers mix[k*step - trim : k*step - trim + window], zero-padded past the ends.
                waves = np.zeros((len(ks), 2, window), dtype=np.float32)
                for i, k in enumerate(ks):
                    start = k * step - trim
                    lo, hi = max(start, 0), min(start + window, n)
                    if hi > lo:
                        waves[i, :, lo - start:hi - start] = mix[:, lo:hi]
                wav = _uvr_run_windows(session, model, waves, hann)
                for i, k in enumerate(ks):
                    lo, hi = k * step, min(k * step + step, n)
                    primary[:, lo:hi] = wav[i, :, trim:trim + hi - lo] * model['compensate']
                if progress is not None:
                    progress((ks[-1] + 1) / count)
    finally:
        del session  # frees the model's RAM or VRAM now, also after an error
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

    # In place: the mix becomes the second stem, so a long file needs two copies, not three.
    # The caller's array is overwritten.
    np.subtract(mix, primary, out=mix)
    secondary = mix
    print(f"[UVR] Done in {time.monotonic() - started:.0f} s")
    if model['primary'] == 'vocals':
        return primary, secondary
    return secondary, primary


def _uvr_ffmpeg(args, feed=None, read=None):
    """Run ffmpeg. feed(stdin) writes its input, read(stdout) consumes its output and gives the
    result. Raises RuntimeError with ffmpeg's last error lines if it fails."""
    cmd = [FFMPEG_PATH or 'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', *args]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if feed else subprocess.DEVNULL,
                            stdout=subprocess.PIPE if read else subprocess.DEVNULL, stderr=subprocess.PIPE)
    errors = []
    # Collect stderr on the side, so a chatty ffmpeg cannot stall on a full pipe.
    drain = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
    drain.start()
    result = None
    try:
        if feed:
            try:
                feed(proc.stdin)
                proc.stdin.close()
            except BrokenPipeError:  # ffmpeg quit early; its exit code and message say why
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
        if read:
            result = read(proc.stdout)
    except BaseException:
        proc.kill()
        raise
    finally:
        proc.wait()
        drain.join()
    if proc.returncode:
        detail = ' / '.join((errors[0] if errors else b'').decode(errors='replace').strip().splitlines()[-3:])
        raise RuntimeError(f"ffmpeg failed (exit code {proc.returncode}): {detail or 'no message'}")
    return result


def _uvr_channels(path):
    """Channel count of the first audio stream: ffmpeg writes an empty WAV, we read its header."""
    head = _uvr_ffmpeg(['-i', str(path), '-map', '0:a:0', '-t', '0', '-c:a', 'pcm_s16le', '-f', 'wav',
                        'pipe:1'], read=lambda pipe: pipe.read())
    fmt = head.find(b'fmt ')
    if head[:4] != b'RIFF' or fmt < 0:
        raise RuntimeError(f"Could not read the audio of {Path(path).name}")
    return int.from_bytes(head[fmt + 10:fmt + 12], 'little')  # the field after the format tag


def uvr_load_audio(path):
    """Decode the first audio stream of any media file to a float32 [2, n] array at 44.1 kHz.

    Mono goes to both channels at full level (ffmpeg's own upmix is 3 dB quieter); more than
    two channels are downmixed by ffmpeg."""
    import numpy as np

    channels = 1 if _uvr_channels(path) == 1 else 2
    parts = _uvr_ffmpeg(['-i', str(path), '-map', '0:a:0', '-ac', str(channels),
                         '-ar', str(UVR_SAMPLE_RATE), '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1'],
                        read=lambda pipe: list(iter(lambda: pipe.read(1 << 22), b'')))
    frame = 4 * channels
    audio = np.empty((2, sum(len(p) for p in parts) // frame), dtype=np.float32)
    pos = 0
    for i, part in enumerate(parts):
        samples = np.frombuffer(part, dtype=np.float32, count=len(part) // 4).reshape(-1, channels)
        audio[:, pos:pos + len(samples)] = samples.T  # a mono column fills both rows
        pos += len(samples)
        parts[i] = None  # free as we go: peak memory stays near two copies, not three
    return audio


def _uvr_raw_input():
    """ffmpeg input options for the float32 stereo stream that _uvr_feed writes to stdin."""
    return ['-f', 'f32le', '-ar', str(UVR_SAMPLE_RATE), '-ac', '2', '-i', 'pipe:0']


def _uvr_feed(stem):
    """A feed for _uvr_ffmpeg that writes a [2, n] stem as interleaved float32, in slices.

    A stem that peaks above full scale is turned down to 0.99 so the encoder does not clip it."""
    import numpy as np

    slice_len = 1 << 18
    peak = max((float(np.abs(stem[:, i:i + slice_len]).max()) for i in range(0, stem.shape[1], slice_len)),
               default=0.0)
    gain = 0.99 / peak if peak > 1.0 else 1.0

    def feed(pipe):
        for i in range(0, stem.shape[1], slice_len):
            pipe.write(np.ascontiguousarray(stem[:, i:i + slice_len].T * gain, dtype=np.float32).tobytes())
    return feed


def uvr_save_mp3(stem, out_path, quality=None, bitrate=None):
    """Encode a [2, n] stem to MP3: LAME VBR quality 0 (best) to 9, or a bitrate in kbit/s."""
    rate = ['-b:a', f'{int(bitrate)}k'] if bitrate else ['-q:a', str(2 if quality is None else int(quality))]
    _uvr_ffmpeg([*_uvr_raw_input(), '-c:a', 'libmp3lame', *rate, '-y', str(out_path)], feed=_uvr_feed(stem))


def uvr_save_video(stem, video_path, out_path, audio_bitrate=192):
    """Write an MP4 with the first video stream of video_path (copied) and the stem as AAC audio."""
    _uvr_ffmpeg(['-i', str(video_path), *_uvr_raw_input(), '-map', '0:v:0', '-map', '1:a:0',
                 '-c:v', 'copy', '-c:a', 'aac', '-b:a', f'{int(audio_bitrate)}k',
                 '-y', str(out_path)], feed=_uvr_feed(stem))


STEMS = ('vocals', 'background')
STEM_LABELS = {'vocals': 'Vocals', 'background': 'Background'}


def uvr_gpu_supported():
    """True when onnxruntime can run UVR on an NVIDIA GPU (the onnxruntime-gpu package)."""
    try:
        import onnxruntime
        return 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    except ImportError:
        return False


# The decoded audio and both stems stay in RAM (about 21 MB per minute each), so a very long
# file could take the server down. Longer audio gets a separation_error instead.
try:
    UVR_MAX_MINUTES = max(1, int(os.environ.get('UVR_MAX_MINUTES', '60')))
except ValueError:
    UVR_MAX_MINUTES = 60


def _stem_path(media_file, stem, video):
    """'<name> (Vocals).mp3' next to media_file, shortened to fit the 255-byte file name limit."""
    suffix = f' ({STEM_LABELS[stem]}){".mp4" if video else ".mp3"}'
    name = media_file.stem
    while len((name + suffix).encode('utf-8')) > 255:
        name = name[:-1]
    return media_file.with_name(name + suffix)


def separate_media(task_id, media_file, keep, mp3_quality=None, on_saved=None):
    """Split a file's audio into vocals and background with UVR.

    Saves '<name> (Vocals).<ext>' and/or '<name> (Background).<ext>' next to media_file for each
    stem in keep: MP3s for an MP3, MP4s with the original video for a video. on_saved(stem,
    path) is called as each file is saved, so a failure on the second keeps the first.
    """
    video = media_file.suffix.lower() != '.mp3'

    def progress(fraction):
        _set_task_message(task_id, f'Separating vocals and background... {fraction * 100:.0f}%', 96)

    _set_task_message(task_id, 'Waiting for another AI job to finish...' if model_lock.locked()
                      else 'Separating vocals and background...', 96)
    with model_lock:
        progress(0)
        key = setting('uvr_model')
        if key not in UVR_MODELS:  # a typo in UVR_MODEL
            print(f"[UVR] Unknown model {key!r}; using inst_hq_3")
            key = 'inst_hq_3'
        choice = setting('uvr_device')
        device = resolve_device(choice)
        uvr_model_path(key, UVR_MODEL_DIR)  # a failed download is not a GPU problem: no retry
        # Whisper stays loaded while more transcriptions are waiting. Free its RAM or VRAM for
        # UVR (on a 4 GB GPU they do not fit together); the next transcription reloads it.
        _unload_whisper()
        mix = uvr_load_audio(media_file)
        minutes = mix.shape[1] / UVR_SAMPLE_RATE / 60
        if minutes > UVR_MAX_MINUTES:
            raise RuntimeError(f'The audio is {minutes:.0f} minutes long; vocal separation is limited to '
                               f'{UVR_MAX_MINUTES} minutes (UVR_MAX_MINUTES)')
        try:
            vocals, background = uvr_separate(mix, key, UVR_MODEL_DIR, device, progress)
        except Exception as e:  # onnxruntime's GPU errors are not RuntimeErrors
            if device == 'cpu' or choice != 'auto':
                raise
            print(f"[UVR] {device} failed ({str(e) or type(e).__name__}); retrying on the CPU")
            vocals, background = uvr_separate(mix, key, UVR_MODEL_DIR, 'cpu', progress)
        del mix
        # Encode inside the lock: the next job's separation must not start while both stems
        # are still in memory.
        _set_task_message(task_id, 'Saving the separated files...', 97)
        stems = {}
        for stem, audio in (('vocals', vocals), ('background', background)):
            if stem not in keep:
                continue
            path = _stem_path(media_file, stem, video)
            try:
                if video:
                    uvr_save_video(audio, media_file, path)
                else:
                    uvr_save_mp3(audio, path, quality=mp3_quality)
            except Exception:
                path.unlink(missing_ok=True)  # no half-written file left for the listings
                raise
            stems[stem] = path
            if on_saved is not None:
                on_saved(stem, path)
    return stems


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
    <meta name="description" content="Download YouTube videos as MP3 or MP4, and transcribe them.">
    <meta name="theme-color" content="#0f0f0f">
    <link rel="icon" href="/favicon.ico" sizes="48x48">
    <link rel="icon" href="/static/icon-192.png" type="image/png" sizes="192x192">
    <link rel="apple-touch-icon" href="/static/apple-touch-icon.png">
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
            flex-shrink: 0;
        }

        .logo img { display: block; width: 100%; height: 100%; }

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

        .badge-processing {
            background: #ff9800;
            color: white;
            animation: pulse-badge 1.5s ease-in-out infinite;
        }

        @keyframes pulse-badge {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
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

        /* AI options under the quality picker: transcription and vocal separation */
        .ai-option { margin: 12px 0; padding: 10px 12px; background: #1a1a1a; border-radius: 8px; border: 1px solid #272727; }
        .ai-check { display: flex; align-items: center; gap: 10px; cursor: pointer; user-select: none; }
        .ai-check input { width: 18px; height: 18px; cursor: pointer; accent-color: #ff0000; flex-shrink: 0; }
        .ai-title { font-size: 14px; color: #f1f1f1; }
        .ai-hint { font-size: 11px; color: #9a9a9a; margin-top: 2px; }
        .stem-choices { display: none; gap: 8px; margin-top: 10px; padding-left: 28px; }
        .stem-choices.show { display: flex; }
        .stem-choice { flex: 1; cursor: pointer; }
        .stem-choice input { position: absolute; opacity: 0; pointer-events: none; }
        .stem-choice span { display: block; height: 100%; padding: 10px 8px; border: 1px solid #3f3f3f; border-radius: 8px; background: #121212; text-align: center; transition: all 0.2s; }
        .stem-choice b { display: block; font-size: 13px; font-weight: 500; color: #f1f1f1; }
        .stem-choice small { display: block; font-size: 11px; color: #9a9a9a; margin-top: 3px; }
        .stem-choice span:hover { border-color: #717171; }
        .stem-choice input:checked + span { border-color: #ff0000; background: rgba(255, 0, 0, 0.1); }
        .stem-choice input:focus-visible + span { outline: 2px solid #3ea6ff; }

        /* Extra results under the main download button */
        .extra-downloads { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 8px; }
        .extra-downloads .download-btn { background: #272727; padding: 10px 18px; font-size: 13px; }
        .extra-downloads .download-btn:hover { background: #3f3f3f; }
        .step-error { margin-top: 8px; font-size: 13px; color: #ff6b6b; }

        /* Labeled extra downloads in the history, as a row under the badges */
        .history-extras { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
        .history-extras:empty { display: none; }
        .history-extra-btn { height: 26px; padding: 0 10px; background: #272727; border: none; border-radius: 6px; color: #f1f1f1; font-size: 11px; cursor: pointer; white-space: nowrap; transition: all 0.2s; }
        .history-extra-btn:hover { background: #3f3f3f; }
        .history-extra-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Settings */
        .gpu-list { font-size: 13px; color: #aaa; margin-bottom: 16px; padding: 12px 16px; background: #1a1a1a; border-radius: 12px; border: 1px solid #272727; line-height: 1.6; }
        .gpu-list b { color: #f1f1f1; font-weight: 500; }
        .settings-card { background: #1a1a1a; border: 1px solid #272727; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
        .settings-card h3 { font-size: 15px; font-weight: 500; color: #f1f1f1; }
        .settings-card p { font-size: 12px; color: #9a9a9a; margin-top: 4px; }
        .settings-row { display: grid; grid-template-columns: 90px 1fr; align-items: center; gap: 10px; margin-top: 12px; }
        .settings-row label { font-size: 13px; color: #aaa; }
        .settings-note { display: none; font-size: 12px; color: #f0b429; margin-top: 10px; }
        .settings-note.show { display: block; }
        .settings-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        .settings-status { font-size: 13px; color: #2ba640; }
        .settings-status.bad { color: #ff6b6b; }

        @media (max-width: 600px) {
            .stem-choices { flex-direction: column; padding-left: 0; }
            .settings-row { grid-template-columns: 1fr; gap: 4px; }
            .settings-row .quality-select { font-size: 13px; padding: 10px 12px; }
            .history-actions { flex-wrap: wrap; }
            .history-download-btn, .history-delete-btn { flex-shrink: 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <img src="/static/icon-192.png" alt="">
            </div>
            <h1>YouTube Downloader</h1>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('audio')" id="tabAudio">Audio (MP3)</button>
            <button class="tab" onclick="switchTab('video')" id="tabVideo">Video (MP4)</button>
            <button class="tab" onclick="switchTab('history')" id="tabHistory">History</button>
            <button class="tab" onclick="switchTab('settings')" id="tabSettings">Settings</button>
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

                    <div class="ai-option">
                        <label class="ai-check">
                            <input type="checkbox" id="enableTranscription" onchange="document.getElementById('timestampOption').style.display = this.checked ? 'flex' : 'none'">
                            <div>
                                <div class="ai-title">Extract transcription</div>
                                <div class="ai-hint">Uses Whisper AI to generate a text file of the spoken content</div>
                            </div>
                        </label>
                        <label id="timestampOption" style="display: none; align-items: center; gap: 10px; cursor: pointer; user-select: none; margin-top: 8px; padding-left: 28px;">
                            <input type="checkbox" id="enableTimestamps" style="width: 16px; height: 16px; cursor: pointer; accent-color: #ff0000;">
                            <div>
                                <div style="font-size: 13px; color: #aaa;">Include timestamps (SRT format)</div>
                            </div>
                        </label>
                    </div>

                    <div class="ai-option">
                        <label class="ai-check">
                            <input type="checkbox" id="enableSeparation" onchange="document.getElementById('stemChoices').classList.toggle('show', this.checked)">
                            <div>
                                <div class="ai-title">Separate vocals and background</div>
                                <div class="ai-hint">Uses UVR AI to split the voice from the music. You always get the original too.</div>
                            </div>
                        </label>
                        <div class="stem-choices" id="stemChoices" role="radiogroup" aria-label="Files to make">
                            <label class="stem-choice">
                                <input type="radio" name="stems" value="vocals">
                                <span><b>Vocals only</b><small>Removes the music and background</small></span>
                            </label>
                            <label class="stem-choice">
                                <input type="radio" name="stems" value="background">
                                <span><b>Background only</b><small>Removes the vocals, like karaoke</small></span>
                            </label>
                            <label class="stem-choice">
                                <input type="radio" name="stems" value="both" checked>
                                <span><b>Both</b><small>One file of each</small></span>
                            </label>
                        </div>
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
                <div class="extra-downloads" id="extraDownloads"></div>
                <div id="stepErrors"></div>
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

        <!-- Settings Content Section -->
        <div class="content-section" id="settingsContent">
            <div class="gpu-list" id="gpuList">Looking for GPUs...</div>

            <div class="settings-card">
                <h3>Transcription (Whisper)</h3>
                <p>Turns speech into text. Larger models are more accurate but slower.</p>
                <div class="settings-row">
                    <label for="setWhisperModel">Model</label>
                    <select class="quality-select" id="setWhisperModel"></select>
                </div>
                <div class="settings-row">
                    <label for="setWhisperDevice">Run on</label>
                    <select class="quality-select" id="setWhisperDevice"></select>
                </div>
                <div class="settings-note" id="whisperNote"></div>
            </div>

            <div class="settings-card">
                <h3>Vocal separation (UVR MDX-Net)</h3>
                <p>Splits the voice from the music. Every model makes both files; each is best at the part in its name.</p>
                <div class="settings-row">
                    <label for="setUvrModel">Model</label>
                    <select class="quality-select" id="setUvrModel"></select>
                </div>
                <div class="settings-row">
                    <label for="setUvrDevice">Run on</label>
                    <select class="quality-select" id="setUvrDevice"></select>
                </div>
                <div class="settings-note" id="uvrNote"></div>
            </div>

            <div class="settings-actions">
                <button class="btn" id="saveSettingsBtn" onclick="saveSettings()">Save settings</button>
                <span class="settings-status" id="settingsStatus"></span>
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
        let polling = false;  // a job started on this page is still running

        // Track background polling for resumed tasks
        const backgroundPolls = {};

        // Initialize
        setInterval(updateActiveCount, 3000);
        updateActiveCount();
        loadHistory();
        resumeProcessingTasks();

        function resumeProcessingTasks() {
            const history = getHistory();
            history.filter(item => item.status === 'processing').forEach(item => {
                if (backgroundPolls[item.task_id]) return;
                backgroundPolls[item.task_id] = setInterval(() => {
                    fetch('/api/check/' + item.task_id)
                        .then(r => r.json())
                        .then(data => {
                            if (data.status === 'completed') {
                                clearInterval(backgroundPolls[item.task_id]);
                                delete backgroundPolls[item.task_id];
                                addToHistory({...item, ...completedFields(data)});
                                if (currentView === 'history') loadHistory();
                            } else if (data.status === 'error' || data.status === 'not_found') {
                                clearInterval(backgroundPolls[item.task_id]);
                                delete backgroundPolls[item.task_id];
                                removeFromHistory(item.task_id);
                                if (currentView === 'history') loadHistory();
                            }
                        })
                        .catch(() => {});
                }, 2000);
            });
        }

        // The files and messages of a finished job, as kept in the history.
        function completedFields(data) {
            return {
                status: 'completed',
                filename: data.filename,
                download_url: data.download_url,
                vocals_url: data.vocals_url || null,
                vocals_filename: data.vocals_filename || null,
                background_url: data.background_url || null,
                background_filename: data.background_filename || null,
                transcription_url: data.transcription_url || null,
                transcription_filename: data.transcription_filename || null
            };
        }

        function switchTab(mode) {
            const view = (mode === 'history' || mode === 'settings') ? mode : 'download';
            currentView = view;
            document.getElementById('tabAudio').classList.toggle('active', mode === 'audio');
            document.getElementById('tabVideo').classList.toggle('active', mode === 'video');
            document.getElementById('tabHistory').classList.toggle('active', view === 'history');
            document.getElementById('tabSettings').classList.toggle('active', view === 'settings');
            document.getElementById('downloadContent').classList.toggle('active', view === 'download');
            document.getElementById('historyContent').classList.toggle('active', view === 'history');
            document.getElementById('settingsContent').classList.toggle('active', view === 'settings');
            if (view === 'history') { loadHistory(); return; }
            if (view === 'settings') { loadSettings(); return; }

            // Audio/Video tabs. A finished result belongs to the tab it came from.
            if (mode !== currentMode && !polling) { hideDownload(); hideStatus(); }
            currentMode = mode;
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
                const isProcessing = item.status === 'processing';

                const youtubeUrl = item.youtube_url || '';
                html += `
                    <div class="history-item ${!isProcessing && !exists ? 'deleted' : ''}">
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
                                ${isProcessing ? '<span class="history-badge badge-processing">Processing...</span>' : ''}
                                <span class="history-date">${formatDate(item.date)}</span>
                            </div>
                            ${isProcessing ? '' : `<div class="history-extras">
                                ${extraButton(item.vocals_url, 'Vocals', 'Download the vocals only', exists)}
                                ${extraButton(item.background_url, 'Background', 'Download the background only (no vocals)', exists)}
                                ${extraButton(item.transcription_url, 'Transcript', 'Download the transcript', exists)}
                            </div>`}
                            ${!isProcessing && !exists ? '<div class="deleted-tooltip">File no longer available on server</div>' : ''}
                        </div>
                        <div class="history-actions">
                            ${isProcessing ? '' : `
                            <button class="history-download-btn" ${exists ? '' : 'disabled'}
                                    onclick="${exists ? `window.location.href='${item.download_url}'` : ''}"
                                    title="${exists ? 'Download' : 'File unavailable'}">
                                <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
                            </button>
                            <button class="history-delete-btn" onclick="deleteHistoryItem('${item.task_id}')" title="Delete">
                                <svg viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
                            </button>
                            `}
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

        // A labeled history button for an extra file (vocals, background, transcript).
        function extraButton(url, label, title, exists) {
            if (!url) return '';
            const go = exists ? `onclick="window.location.href='${escapeHtml(url)}'"` : 'disabled';
            return `<button class="history-extra-btn" ${go} title="${exists ? title : 'File unavailable'}">${label}</button>`;
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

            const transcribe = document.getElementById('enableTranscription').checked;
            const timestamps = document.getElementById('enableTimestamps').checked;
            // Vocals only = remove the background; background only = remove the vocals.
            const separate = document.getElementById('enableSeparation').checked;
            const stems = separate ? document.querySelector('input[name="stems"]:checked').value : '';

            fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: currentUrl, bitrate: quality, resolution: quality,
                                      transcribe: transcribe, timestamps: timestamps,
                                      remove_vocals: stems === 'background' || stems === 'both',
                                      remove_background: stems === 'vocals' || stems === 'both'})
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
                // Save to history immediately so it persists if user leaves
                window.currentConversion = {
                    task_id: data.task_id,
                    title: currentTitle,
                    thumbnail: currentThumbnail,
                    youtube_url: currentUrl,
                    type: currentMode,
                    quality: quality,
                    date: Date.now(),
                    status: 'processing'
                };
                addToHistory(window.currentConversion);
                showStatus('Downloading...');
                document.getElementById('convertBtn').disabled = false;
                polling = true;
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
                        polling = false;
                        setProgress(100);
                        showStatus('Complete!');
                        showDownload(data.filename, data.download_url);
                        showExtras(data);
                        updateActiveCount();

                        // Update history entry with completed info
                        if (window.currentConversion) {
                            addToHistory({...window.currentConversion, ...completedFields(data)});
                            window.currentConversion = null;
                        }
                        if (currentView === 'history') loadHistory();
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        polling = false;
                        showError(data.error || 'Download failed');
                        hideStatus();
                        updateActiveCount();
                        // Remove failed task from history
                        if (window.currentConversion) {
                            removeFromHistory(window.currentConversion.task_id);
                            window.currentConversion = null;
                        }
                        if (currentView === 'history') loadHistory();
                    } else if (data.status === 'processing') {
                        const progress = data.overall_progress ?? data.progress ?? 0;
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
        // The vocals, background and transcript buttons, and any step that failed.
        function showExtras(data) {
            const extras = document.getElementById('extraDownloads');
            const errors = document.getElementById('stepErrors');
            extras.innerHTML = '';
            errors.innerHTML = '';
            [['vocals', 'Vocals only'], ['background', 'Background only'], ['transcription', 'Transcript']]
                .forEach(([key, label]) => {
                    const url = data[key + '_url'];
                    if (!url) return;
                    const a = document.createElement('a');
                    a.className = 'download-btn';
                    a.href = url;
                    a.download = data[key + '_filename'] || '';
                    a.title = data[key + '_filename'] || label;
                    a.textContent = label;
                    extras.appendChild(a);
                });
            [['separation_error', 'Vocal separation failed: '], ['transcription_error', 'Transcription failed: ']]
                .forEach(([key, prefix]) => {
                    if (!data[key]) return;
                    const div = document.createElement('div');
                    div.className = 'step-error';
                    div.textContent = prefix + data[key];
                    errors.appendChild(div);
                });
        }
        function hideDownload() {
            document.getElementById('downloadSection').classList.remove('show');
            document.getElementById('extraDownloads').innerHTML = '';
            document.getElementById('stepErrors').innerHTML = '';
        }

        // Settings page
        function fillSelect(id, options, value) {
            const select = document.getElementById(id);
            select.innerHTML = '';
            options.forEach(o => select.add(new Option(o.label, o.id, false, o.id === value)));
        }

        function showSettings(data) {
            const opts = data.options;
            fillSelect('setWhisperModel', opts.whisper_models, data.settings.whisper_model);
            fillSelect('setWhisperDevice', opts.devices, data.settings.whisper_device);
            fillSelect('setUvrModel', opts.uvr_models, data.settings.uvr_model);
            fillSelect('setUvrDevice', opts.devices, data.settings.uvr_device);
            const gpus = opts.gpus || [];
            const list = document.getElementById('gpuList');
            list.innerHTML = gpus.length
                ? '<b>GPUs found:</b> ' + gpus.map(g => escapeHtml(`GPU ${g.id.slice(5)}: ${g.name} (${g.memory_gb} GB)`)).join(', ')
                : '<b>No NVIDIA GPU found.</b> Whisper and UVR run on the CPU, which is slower.';
            const whisperNote = document.getElementById('whisperNote');
            whisperNote.textContent = gpus.length && data.whisper_gpu_failed && data.settings.whisper_device === 'auto'
                ? 'The GPU failed a transcription before, so Automatic uses the CPU. Pick a GPU above to try it again.' : '';
            whisperNote.classList.toggle('show', !!whisperNote.textContent);
            const uvrNote = document.getElementById('uvrNote');
            uvrNote.textContent = gpus.length && !data.uvr_gpu_supported
                ? 'This install cannot run UVR on a GPU (onnxruntime-gpu is missing), so it uses the CPU.' : '';
            uvrNote.classList.toggle('show', !!uvrNote.textContent);
        }

        function settingsStatus(text, bad) {
            const status = document.getElementById('settingsStatus');
            status.textContent = text;
            status.classList.toggle('bad', !!bad);
        }

        function loadSettings() {
            settingsStatus('');
            fetch('/api/settings')
                .then(r => r.json())
                .then(showSettings)
                .catch(err => settingsStatus('Could not load the settings: ' + err.message, true));
        }

        function saveSettings() {
            const body = {
                whisper_model: document.getElementById('setWhisperModel').value,
                whisper_device: document.getElementById('setWhisperDevice').value,
                uvr_model: document.getElementById('setUvrModel').value,
                uvr_device: document.getElementById('setUvrDevice').value
            };
            document.getElementById('saveSettingsBtn').disabled = true;
            fetch('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
                .then(r => r.json())
                .then(data => {
                    if (data.error) { settingsStatus(data.error, true); return; }
                    showSettings(data);
                    settingsStatus('Saved. New jobs use these settings.');
                })
                .catch(err => settingsStatus('Could not save: ' + err.message, true))
                .finally(() => { document.getElementById('saveSettingsBtn').disabled = false; });
        }
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


def canonical_youtube_url(video_id):
    """Build the URL handed to yt-dlp from a validated 11-char video id.

    yt-dlp is never given the caller's raw string: extract_video_id() matches anywhere
    in its input, so a value like '--exec=... youtube.com/watch?v=<id>' would otherwise
    reach yt-dlp's argv and be parsed as an option.
    """
    return f'https://www.youtube.com/watch?v={video_id}'


TASK_ID_RE = re.compile(r'^[0-9a-f]{8}$')


def resolve_task_dir(task_id):
    """Return the directory for task_id, or None if the id is malformed.

    Task ids arrive straight from URL paths; anything other than the 8-hex-char id we
    mint is rejected so it can never name a path outside DOWNLOAD_DIR. ('..' used to
    resolve to /app, so DELETE /api/delete/.. would rmtree the whole app.)
    """
    if not isinstance(task_id, str) or not TASK_ID_RE.match(task_id):
        return None
    return DOWNLOAD_DIR / task_id


def resolve_task_file(task_id, filename):
    """Return the path of a file directly inside a task directory, or None."""
    task_dir = resolve_task_dir(task_id)
    if task_dir is None or not filename or filename in ('.', '..'):
        return None
    if any(c in filename for c in ('/', chr(92), chr(0))):  # slash, backslash, NUL
        return None
    path = task_dir / filename
    try:
        if path.resolve().parent != task_dir.resolve():
            return None
    except (OSError, ValueError):
        return None
    return path


def fetch_video_metadata(url):
    """Return yt-dlp's full info dict for a video."""
    cmd = [
        'yt-dlp',
        '--dump-json',
        '--no-playlist',
        '--js-runtimes', 'node',
        url
    ]
    for attempt in range(1, 2 + YTDLP_RETRIES):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        # A 403 cannot happen here (no media is fetched), but yt-dlp can still crash.
        if ytdlp_failure_kind(result.returncode, result.stderr) != 'crash' or attempt > YTDLP_RETRIES:
            raise Exception(f"Failed to get video info: {result.stderr}")
        print(f"[yt-dlp] Video info: yt-dlp crashed; retrying ({attempt + 1}/{1 + YTDLP_RETRIES})")


def get_video_info(url):
    """Get video title and duration using yt-dlp."""
    info = fetch_video_metadata(url)
    return {
        'title': info.get('title', 'Unknown'),
        'duration': info.get('duration', 0),  # Duration in seconds
        'thumbnail': info.get('thumbnail', '')
    }


def _set_task_message(task_id, message, progress=None):
    with downloads_lock:
        entry = active_downloads.get(task_id)
        if entry is not None:
            entry['message'] = message
            if progress is not None:
                entry['progress'] = progress


def _finish_task(task_id, record):
    """Publish a job's final state. The hourly cleanup's retention window starts now."""
    if record.get('status') == 'error':
        # The task record is deleted with the job, so keep the reason in the server log.
        print(f"[job] Task {task_id} failed: {' / '.join(str(record.get('error')).splitlines()[-3:])}")
    try:
        os.utime(DOWNLOAD_DIR / task_id)
    except OSError:
        pass
    with downloads_lock:
        active_downloads[task_id] = record


# Two yt-dlp failures are transient, so a fresh yt-dlp run usually succeeds:
# - YouTube intermittently refuses a freshly issued stream URL with HTTP 403 on the very first
#   request, more often after a burst of downloads (yt-dlp issue #17395, an external issue).
# - The yt-dlp process itself sometimes crashes while building YouTube's caption URLs: a
#   segfault, or a nonsensical "'<' not supported between instances of 'function' and 'str'".
#   Both point to memory corruption in native code; the rate swings with YouTube's responses.
try:
    YTDLP_RETRIES = max(0, int(os.environ.get('YTDLP_RETRIES', '3')))
except ValueError:
    YTDLP_RETRIES = 3

_YTDLP_CORRUPTION_SIGNS = ("not supported between instances of 'function' and 'str'",)


def ytdlp_failure_kind(returncode, output):
    """Classify a failed yt-dlp run: '403' or 'crash' are worth retrying, None is not."""
    if returncode < 0 or any(sign in output for sign in _YTDLP_CORRUPTION_SIGNS):
        return 'crash'
    if 'HTTP Error 403' in output:
        return '403'
    return None


def run_ytdlp(task_id, cmd, timeout, downloading_message, finishing_message):
    """Run yt-dlp, publishing its progress to the task. Raises if it fails."""
    attempts = 1 + YTDLP_RETRIES
    for attempt in range(1, attempts + 1):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

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
                        active_downloads[task_id]['message'] = f'{downloading_message} {progress:.1f}%'
                    else:
                        active_downloads[task_id]['message'] = finishing_message

        process.wait(timeout=timeout)
        if process.returncode == 0:
            return

        error_detail = '\n'.join(last_lines[-5:]) if last_lines else 'No output captured'
        kind = ytdlp_failure_kind(process.returncode, error_detail)
        if kind == 'crash' and process.returncode < 0:
            error_detail += f'\n(yt-dlp crashed with signal {-process.returncode})'
        if kind is None or attempt == attempts:
            if attempt > 1:
                error_detail += f'\n(failed on all {attempt} attempts)'
            raise Exception(f"yt-dlp download failed:\n{error_detail}")

        reason = 'YouTube refused the stream' if kind == '403' else 'yt-dlp crashed'
        print(f"[yt-dlp] Task {task_id}: {reason}; retrying ({attempt + 1}/{attempts})")
        with downloads_lock:
            active_downloads[task_id]['message'] = f'{reason}; retrying ({attempt + 1}/{attempts})...'
        time.sleep(2 * attempt)


def _ai_steps(keep, transcribe):
    """The AI steps a job runs after its download, for the progress scale."""
    return (('separate',) if keep else ()) + (('transcribe',) if transcribe else ())


def _finish_media(task_id, task_dir, media_file, result, transcribe, timestamps, keep, mp3_quality=None):
    """The optional AI steps after a download: UVR separation, then Whisper transcription.

    Both use the original audio. A failed step is reported in result (separation_error,
    transcription_error) and the download itself still succeeds.
    """
    if keep or transcribe:
        # AI steps run one at a time behind model_lock. Hand the download slot to the next job
        # now, so plain downloads do not wait behind them.
        release_job_slot(task_id)
    if keep:
        def saved(stem, path):
            result[f'{stem}_filename'] = path.name
            result[f'{stem}_url'] = f'/download/{task_id}/{quote(path.name)}'
        try:
            separate_media(task_id, media_file, keep, mp3_quality, on_saved=saved)
        except Exception as e:
            print(f"[UVR] Separation failed for task {task_id}: {e!r}")
            result['separation_error'] = str(e) or type(e).__name__
    if transcribe:
        try:
            _set_task_message(task_id, 'Waiting for another AI job to finish...'
                              if model_lock.locked() else 'Transcribing audio with Whisper AI...', 98)
            text = transcribe_audio_file(
                str(media_file), timestamps=timestamps,
                on_start=lambda: _set_task_message(task_id, 'Transcribing audio with Whisper AI...', 98))
            transcript_name = media_file.stem + ('.srt' if timestamps else '.txt')
            (task_dir / transcript_name).write_text(text, encoding='utf-8')
            result['transcription_url'] = f'/download/{task_id}/{quote(transcript_name)}'
            result['transcription_filename'] = transcript_name
        except Exception as e:
            print(f"[Whisper] Transcription failed for task {task_id}: {e!r}")
            result['transcription_error'] = str(e) or type(e).__name__


def download_and_convert(task_id, url, bitrate='320', transcribe=False, timestamps=False, keep=()):
    """Download YouTube video and convert to MP3, optionally separate and transcribe."""
    try:
        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'processing',
                'progress': 0,
                'steps': _ai_steps(keep, transcribe),
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
            '--js-runtimes', 'node',
        ]

        # Add FFmpeg location if available from imageio-ffmpeg
        if FFMPEG_PATH:
            cmd.extend(['--ffmpeg-location', str(Path(FFMPEG_PATH).parent)])

        cmd.append(url)

        with downloads_lock:
            active_downloads[task_id]['message'] = 'Downloading...'
            active_downloads[task_id]['progress'] = 0

        run_ytdlp(task_id, cmd, 300, 'Downloading...', 'Converting to MP3...')

        # Find the MP3 file
        mp3_files = list(task_dir.glob('*.mp3'))
        if not mp3_files:
            raise Exception("No MP3 file was created")

        mp3_file = mp3_files[0]
        filename = mp3_file.name

        result = {
            'status': 'completed',
            'filename': filename,
            'filepath': str(mp3_file),
            'download_url': f'/download/{task_id}/{quote(filename)}'
        }
        _finish_media(task_id, task_dir, mp3_file, result, transcribe, timestamps, keep, audio_quality)
        _finish_task(task_id, result)

    except Exception as e:
        _finish_task(task_id, {'status': 'error', 'error': str(e)})
    finally:
        if transcribe:
            whisper_release()


def download_video(task_id, url, resolution='1080', transcribe=False, timestamps=False, keep=()):
    """Download YouTube video as MP4, optionally separate and transcribe."""
    try:
        with downloads_lock:
            active_downloads[task_id] = {
                'status': 'processing',
                'progress': 0,
                'steps': _ai_steps(keep, transcribe),
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
            '--js-runtimes', 'node',
        ]

        # Add FFmpeg location if available
        if FFMPEG_PATH:
            cmd.extend(['--ffmpeg-location', str(Path(FFMPEG_PATH).parent)])

        cmd.append(url)

        with downloads_lock:
            active_downloads[task_id]['message'] = 'Downloading video...'
            active_downloads[task_id]['progress'] = 0

        run_ytdlp(task_id, cmd, 600, 'Downloading video...', 'Merging video and audio...')

        # Find the video file
        video_files = list(task_dir.glob('*.mp4')) + list(task_dir.glob('*.mkv')) + list(task_dir.glob('*.webm'))
        if not video_files:
            raise Exception("No video file was created")

        video_file = video_files[0]
        filename = video_file.name

        result = {
            'status': 'completed',
            'filename': filename,
            'filepath': str(video_file),
            'download_url': f'/download/{task_id}/{quote(filename)}'
        }
        # Whisper and UVR read the audio straight from the video file through ffmpeg.
        _finish_media(task_id, task_dir, video_file, result, transcribe, timestamps, keep)
        _finish_task(task_id, result)

    except Exception as e:
        _finish_task(task_id, {'status': 'error', 'error': str(e)})
    finally:
        if transcribe:
            whisper_release()


VALID_BITRATES = ['320', '256', '192', '128', '96', '64']
VALID_RESOLUTIONS = ['2160', '1440', '1080', '720', '480', '360']


def _env_count(name, default, minimum):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


# At most MAX_ACTIVE_JOBS jobs run at once, for the web page and MCP alike. More jobs wait in
# a first-in, first-out queue and start as running jobs finish. MAX_QUEUED_JOBS is only a
# backstop against a flood of requests.
MAX_ACTIVE_JOBS = _env_count('MAX_ACTIVE_JOBS', 4, 1)
MAX_QUEUED_JOBS = _env_count('MAX_QUEUED_JOBS', 100, 0)

# Lock order: _queue_lock before downloads_lock, never the reverse.
_queue_lock = threading.Lock()
_job_queue = collections.deque()  # (task_id, target, args, transcribe) waiting for a slot
_slot_holders = set()             # task ids of the jobs that hold a slot


class QueueFull(Exception):
    """MAX_QUEUED_JOBS jobs are already waiting."""


def _update_queue_positions_locked():
    """Tell each waiting job its place in line. Caller holds _queue_lock."""
    with downloads_lock:
        waiting = len(_job_queue)
        for position, job in enumerate(_job_queue, start=1):
            entry = active_downloads.get(job[0])
            if entry is not None:
                entry['message'] = f'Waiting in queue: {position} of {waiting}'
                entry['queue_position'] = position


def _dispatch_locked():
    """Start waiting jobs while slots are free. Caller holds _queue_lock."""
    while _job_queue and len(_slot_holders) < MAX_ACTIVE_JOBS:
        job = _job_queue.popleft()
        task_id, target, args, transcribe = job
        _slot_holders.add(task_id)
        try:
            threading.Thread(target=_run_job, args=(target, args), daemon=True).start()
        except RuntimeError as e:  # can't start new thread
            _slot_holders.discard(task_id)
            if _slot_holders:
                # A running job dispatches again when it finishes. Retry then, instead of
                # failing every waiting job on a momentary thread limit.
                _job_queue.appendleft(job)
                break
            with downloads_lock:
                active_downloads[task_id] = {'status': 'error', 'error': f'Could not start the job: {e}'}
            if transcribe:
                whisper_release()
    _update_queue_positions_locked()


def release_job_slot(task_id):
    """Give a job's slot to the next job in line. Calling it again does nothing."""
    with _queue_lock:
        if task_id in _slot_holders:
            _slot_holders.discard(task_id)
            _dispatch_locked()


def _run_job(target, args):
    """Run one job, then hand its slot to the next job in line."""
    try:
        target(*args)
    finally:
        release_job_slot(args[0])


class BadRequest(Exception):
    """A web API request with an invalid value (HTTP 400)."""


def _api_flag(data, name):
    """A true/false field of a web API request: a JSON boolean, 1/0, or "true"/"false"/"yes"/"no"."""
    value = data.get(name)
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ('true', 'false', 'yes', 'no', '1', '0'):
        return value.strip().lower() in ('true', 'yes', '1')
    raise BadRequest(f'{name} must be true or false (got {value!r})')


def stems_to_keep(remove_vocals=False, remove_background=False):
    """The UVR stems a job saves: removing the vocals keeps the background, and the reverse."""
    return tuple(stem for stem, wanted in (('vocals', remove_background), ('background', remove_vocals))
                 if wanted)


def _start_task(target, video_id, quality, transcribe, timestamps, keep=()):
    """Run a job now if a slot is free, otherwise queue it. Returns the task id."""
    task_id = str(uuid.uuid4())[:8]
    keep = tuple(stem for stem in STEMS if stem in keep)
    args = (task_id, canonical_youtube_url(video_id), quality, bool(transcribe), bool(timestamps), keep)
    with _queue_lock:
        if len(_slot_holders) >= MAX_ACTIVE_JOBS and len(_job_queue) >= MAX_QUEUED_JOBS:
            raise QueueFull(f'The queue is full: {len(_job_queue)} jobs are already waiting. Try again later.')
        # Registered before the job runs, so a status check made immediately after (the
        # MCP tools make one) can never see 'not_found'.
        with downloads_lock:
            active_downloads[task_id] = {'status': 'processing', 'progress': 0, 'message': 'Starting...'}
        if transcribe:
            whisper_acquire()
        _job_queue.append((task_id, target, args, bool(transcribe)))
        _dispatch_locked()
    return task_id


def cancel_queued_task(task_id):
    """Remove a job that has not started yet. Returns True if it was waiting in the queue."""
    with _queue_lock:
        job = next((j for j in _job_queue if j[0] == task_id), None)
        if job is None:
            return False
        _job_queue.remove(job)
        with downloads_lock:
            active_downloads.pop(task_id, None)
        _update_queue_positions_locked()
    if job[3]:
        whisper_release()
    return True


def start_audio_task(video_id, bitrate='320', transcribe=False, timestamps=False, keep=()):
    """Start an MP3 download in the background and return its task id.

    keep: the UVR stems to save as well ('vocals', 'background'); see stems_to_keep().
    """
    bitrate = str(bitrate)
    if bitrate not in VALID_BITRATES:
        bitrate = '320'
    return _start_task(download_and_convert, video_id, bitrate, transcribe, timestamps, keep)


def start_video_task(video_id, resolution='1080', transcribe=False, timestamps=False, keep=()):
    """Start an MP4 download in the background and return its task id."""
    resolution = str(resolution)
    if resolution not in VALID_RESOLUTIONS:
        resolution = '1080'
    return _start_task(download_video, video_id, resolution, transcribe, timestamps, keep)


# ===========================================================================
# MCP server - Model Context Protocol over Streamable HTTP (POST /mcp)
# ===========================================================================
#
# Lets AI assistants drive the downloader. Implements protocol revision 2026-07-28,
# which is stateless: every request carries its protocol version and client
# capabilities in params._meta, mirrored into the MCP-Protocol-Version, Mcp-Method and
# Mcp-Name headers, and there is no initialize handshake. Most deployed clients still
# speak the legacy era (2025-03-26 through 2025-11-25) and open with `initialize`, so
# that is answered too. In both eras the server keeps no protocol session: long-running
# work is referenced by an explicit task_id that the model passes to get_task_status.
#
# Authorization is optional and off by default: MCP_AUTH=off | token | oauth.

SERVER_VERSION = '1.3.0'
SERVER_INFO = {'name': 'ytdl-web', 'title': 'ytdl-web YouTube Downloader', 'version': SERVER_VERSION}

MCP_MODERN_VERSIONS = ['2026-07-28']
MCP_LEGACY_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26']
MCP_SUPPORTED_VERSIONS = MCP_MODERN_VERSIONS + MCP_LEGACY_VERSIONS

META_PROTOCOL_VERSION = 'io.modelcontextprotocol/protocolVersion'
META_CLIENT_CAPABILITIES = 'io.modelcontextprotocol/clientCapabilities'
META_SERVER_INFO = 'io.modelcontextprotocol/serverInfo'

MCP_PATHS = ('/mcp', '/mcp/')
MCP_PRM_PATHS = ('/.well-known/oauth-protected-resource', '/.well-known/oauth-protected-resource/mcp')
MCP_MAX_BODY_BYTES = 1024 * 1024
MCP_DEFAULT_WAIT = 45
MCP_MAX_WAIT = 600
# Transcript text returned inline with a job result; get_transcript returns more. A result
# carries it twice (text content and structuredContent), so keep it well under the ~25k-token
# tool-output limits some clients apply.
MCP_INLINE_TRANSCRIPT_CHARS = 20000


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MCP_ENABLED = _env_flag('MCP_ENABLED', True)
MCP_AUTH = os.environ.get('MCP_AUTH', 'off').strip().lower() or 'off'
MCP_AUTH_TOKEN = os.environ.get('MCP_AUTH_TOKEN', '').strip()
MCP_OAUTH_ISSUER = os.environ.get('MCP_OAUTH_ISSUER', '').strip()
MCP_OAUTH_INTROSPECTION_URL = os.environ.get('MCP_OAUTH_INTROSPECTION_URL', '').strip()
MCP_OAUTH_CLIENT_ID = os.environ.get('MCP_OAUTH_CLIENT_ID', '')
MCP_OAUTH_CLIENT_SECRET = os.environ.get('MCP_OAUTH_CLIENT_SECRET', '')
MCP_OAUTH_SCOPES = os.environ.get('MCP_OAUTH_SCOPES', '').split()
PUBLIC_BASE_URL = os.environ.get('PUBLIC_BASE_URL', '').strip().rstrip('/')
MCP_ALLOWED_ORIGINS = {o.strip().rstrip('/').lower()
                       for o in os.environ.get('MCP_ALLOWED_ORIGINS', '').split(',') if o.strip()}


def _mcp_config_error():
    """Return why the MCP settings are unusable, or None. /mcp fails closed on error."""
    if MCP_AUTH not in ('off', 'token', 'oauth'):
        return f'MCP_AUTH must be off, token or oauth (got {MCP_AUTH!r})'
    if MCP_AUTH == 'token' and not MCP_AUTH_TOKEN:
        return 'MCP_AUTH=token requires MCP_AUTH_TOKEN'
    if MCP_AUTH == 'oauth':
        missing = [name for name, value in (('PUBLIC_BASE_URL', PUBLIC_BASE_URL),
                                            ('MCP_OAUTH_ISSUER', MCP_OAUTH_ISSUER),
                                            ('MCP_OAUTH_INTROSPECTION_URL', MCP_OAUTH_INTROSPECTION_URL))
                   if not value]
        if missing:
            return 'MCP_AUTH=oauth requires ' + ', '.join(missing)
    return None


MCP_CONFIG_ERROR = _mcp_config_error()

MCP_CAPABILITIES = {'tools': {'listChanged': False}}
# 2026-07-28 requires caching hints on server/discover and tools/list results. The tool
# list is the same for every caller and only changes with a new server version.
MCP_CACHE_HINTS = {'ttlMs': 3600000, 'cacheScope': 'public'}

MCP_INSTRUCTIONS = (
    'ytdl-web downloads YouTube videos as MP3 (download_audio) or MP4 (download_video), '
    'transcribes them with Whisper (transcribe_video, or transcribe=true on a download) as SRT or '
    'plain text, and splits vocals from background with UVR (remove_vocals / remove_background). '
    'get_video_info looks up a video without starting a job.\n'
    'Jobs: download_audio, download_video and transcribe_video start a job and return task_id and '
    f'a status ("processing", "completed" or "error") after waiting up to wait_seconds (default '
    f'{MCP_DEFAULT_WAIT}). While "processing", call get_task_status with the task_id until it ends; '
    'do not start the same job again. Busy servers queue jobs (see queue_position). '
    'progress is percent 0-100. Read next_step when present. On "error", give the user the error '
    'text (403s and yt-dlp crashes were already retried).\n'
    'Results: file (the original), vocals_file, background_file and transcript_file each have name, '
    'url (download link) and mime_type. The files stay on the server: give the user the '
    f'url. transcript holds the text, up to {MCP_INLINE_TRANSCRIPT_CHARS:,} characters '
    '(transcript_truncated=true if cut); get_transcript returns more (max_chars). '
    'separation_error / transcription_error: the download worked but that step failed.\n'
    'Lifetime: files are deleted 1-2 hours after the job finishes. list_downloads lists started '
    "and finished jobs. delete_download deletes a finished job's files or cancels a queued job; "
    'a running job cannot be deleted or stopped.\n'
    'AI steps on a long video can take several minutes.'
)


class McpError(Exception):
    """A JSON-RPC error to return for the current request."""

    def __init__(self, code, message, http_status=200, data=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.data = data


class ToolError(Exception):
    """A tool execution error, reported to the model as a result with isError set."""


class ClientGone(Exception):
    """The client closed the response stream, which cancels the request."""


class AuthFailure(Exception):
    def __init__(self, status, challenge, error, description):
        super().__init__(description)
        self.status = status
        self.challenge = challenge
        self.error = error
        self.description = description


# --- Origin and authorization -------------------------------------------------------

def _origin_of(url):
    parsed = urlparse(url)
    return f'{parsed.scheme}://{parsed.netloc}'.lower()


def mcp_origin_allowed(origin):
    """DNS-rebinding protection for /mcp.

    Native clients send no Origin. A browser-based client is allowed only from a loopback
    page, PUBLIC_BASE_URL or MCP_ALLOWED_ORIGINS - never merely because Origin matches the
    Host header, since a DNS-rebinding page controls both of those.
    """
    if origin is None:
        return True
    origin = origin.strip().rstrip('/').lower()
    if origin in MCP_ALLOWED_ORIGINS:
        return True
    if PUBLIC_BASE_URL and origin == _origin_of(PUBLIC_BASE_URL):
        return True
    parsed = urlparse(origin)
    return parsed.scheme in ('http', 'https') and parsed.hostname in ('localhost', '127.0.0.1', '::1')


def _www_authenticate(error=None, description=None):
    if MCP_AUTH == 'oauth':
        params = [f'resource_metadata="{PUBLIC_BASE_URL}/.well-known/oauth-protected-resource/mcp"']
        if MCP_OAUTH_SCOPES:
            params.append('scope="' + ' '.join(MCP_OAUTH_SCOPES) + '"')
    else:
        params = ['realm="ytdl-web"']
    if error:
        params.append(f'error="{error}"')
    if description:
        params.append(f'error_description="{description}"')
    return 'Bearer ' + ', '.join(params)


def _norm_resource(uri):
    parsed = urlparse(uri.strip())
    return f'{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip("/")}'


# Active and inactive results are cached separately, so a flood of junk tokens can only
# churn the negative cache, never evict the tokens of legitimate clients.
_active_tokens = {}
_inactive_tokens = {}
_introspection_lock = threading.Lock()
_introspection_slots = threading.BoundedSemaphore(8)
_TOKEN_CACHE_MAX = 1000


def _cache_put(cache, key, entry):
    """Insert (expiry, claims), evicting expired entries and then the oldest."""
    if len(cache) >= _TOKEN_CACHE_MAX:
        now = time.time()
        for stale in [k for k, (expiry, _) in cache.items() if expiry <= now]:
            del cache[stale]
        while len(cache) >= _TOKEN_CACHE_MAX:
            del cache[next(iter(cache))]
    cache[key] = entry


def _introspect_token(token):
    """Ask the authorization server about a token (RFC 7662). Results are cached briefly."""
    key = hashlib.sha256(token.encode('utf-8')).hexdigest()
    now = time.time()
    with _introspection_lock:
        for cache in (_active_tokens, _inactive_tokens):
            hit = cache.get(key)
            if hit and hit[0] > now:
                return hit[1]

    # Cap concurrent calls so unauthenticated traffic cannot pin threads on the
    # authorization server or exhaust its rate limit for this client.
    if not _introspection_slots.acquire(blocking=False):
        raise AuthFailure(503, None, 'temporarily_unavailable', 'Too many authorization checks in flight')
    try:
        claims = _call_introspection_endpoint(token)
    finally:
        _introspection_slots.release()

    ttl = 60.0 if claims.get('active') else 10.0
    exp = claims.get('exp')
    if isinstance(exp, (int, float)):
        ttl = max(0.0, min(ttl, exp - now))
    with _introspection_lock:
        _cache_put(_active_tokens if claims.get('active') else _inactive_tokens, key, (now + ttl, claims))
    return claims


def _call_introspection_endpoint(token):

    request = urllib.request.Request(
        MCP_OAUTH_INTROSPECTION_URL,
        data=urlencode({'token': token, 'token_type_hint': 'access_token'}).encode('ascii'),
        method='POST',
        headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
    )
    if MCP_OAUTH_CLIENT_ID:
        # RFC 6749 2.3.1: id and secret are form-encoded before Basic encoding.
        pair = f'{quote_plus(MCP_OAUTH_CLIENT_ID)}:{quote_plus(MCP_OAUTH_CLIENT_SECRET)}'
        request.add_header('Authorization', 'Basic ' + base64.b64encode(pair.encode('utf-8')).decode('ascii'))
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            claims = json.loads(response.read().decode('utf-8'))
        if not isinstance(claims, dict):
            raise ValueError('introspection response is not a JSON object')
    except Exception as e:
        print(f'[MCP] Token introspection failed: {e}')
        raise AuthFailure(503, None, 'temporarily_unavailable', 'Authorization server unavailable')
    return claims


def mcp_authorize(headers):
    """Raise AuthFailure unless the request is allowed to use /mcp."""
    if MCP_AUTH == 'off':
        return
    scheme, _, token = (headers.get('Authorization') or '').partition(' ')
    token = token.strip()
    if scheme.lower() != 'bearer' or not token:
        raise AuthFailure(401, _www_authenticate(), 'invalid_request', 'Authorization required')

    if MCP_AUTH == 'token':
        if not hmac.compare_digest(token.encode('utf-8'), MCP_AUTH_TOKEN.encode('utf-8')):
            raise AuthFailure(401, _www_authenticate('invalid_token'), 'invalid_token', 'Invalid token')
        return

    claims = _introspect_token(token)
    if not claims.get('active'):
        raise AuthFailure(401, _www_authenticate('invalid_token', 'Token is not active'),
                          'invalid_token', 'Token is not active')
    exp = claims.get('exp')
    if isinstance(exp, (int, float)) and exp < time.time():
        raise AuthFailure(401, _www_authenticate('invalid_token', 'Token has expired'),
                          'invalid_token', 'Token has expired')
    # RFC 8707: only accept tokens issued for this server.
    aud = claims.get('aud')
    audiences = [aud] if isinstance(aud, str) else (aud if isinstance(aud, list) else [])
    accepted = {_norm_resource(PUBLIC_BASE_URL + '/mcp'), _norm_resource(PUBLIC_BASE_URL)}
    if not any(isinstance(a, str) and _norm_resource(a) in accepted for a in audiences):
        raise AuthFailure(401, _www_authenticate('invalid_token', 'Token was not issued for this server'),
                          'invalid_token', 'Token was not issued for this server')
    granted = set(str(claims.get('scope') or '').split())
    if MCP_OAUTH_SCOPES and not set(MCP_OAUTH_SCOPES) <= granted:
        raise AuthFailure(403, _www_authenticate('insufficient_scope', 'Token lacks a required scope'),
                          'insufficient_scope', 'Token lacks a required scope')


def protected_resource_metadata():
    """OAuth 2.0 Protected Resource Metadata (RFC 9728) for MCP_AUTH=oauth."""
    doc = {
        'resource': f'{PUBLIC_BASE_URL}/mcp',
        'authorization_servers': [MCP_OAUTH_ISSUER],
        'bearer_methods_supported': ['header'],
        'resource_name': 'ytdl-web',
    }
    if MCP_OAUTH_SCOPES:
        doc['scopes_supported'] = MCP_OAUTH_SCOPES
    return doc


# --- Tasks as seen by MCP clients ---------------------------------------------------

_HOST_RE = re.compile(r'^([A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(:\d{1,5})?$')
MEDIA_EXTS = ('.mp3', '.mp4', '.mkv', '.webm')
TRANSCRIPT_EXTS = {'.srt': 'srt', '.txt': 'text'}
# yt-dlp leftovers: per-format streams (name.f398.mp4), merge temp files, partials.
_INTERMEDIATE_RE = re.compile(r'\.(f\d+(-\d+)?|temp)\.[^.]+$|\.(part|ytdl)$')
MIME_TYPES = {
    '.mp3': 'audio/mpeg',
    '.mp4': 'video/mp4',
    '.mkv': 'video/x-matroska',
    '.webm': 'video/webm',
    '.srt': 'application/x-subrip',
    '.txt': 'text/plain',
}
# Each waiting tools/call holds a thread; past this many, calls return without waiting.
_wait_slots = threading.BoundedSemaphore(32)


def client_connected(handler):
    """False once the client has closed its end of the connection."""
    try:
        readable, _, _ = select.select([handler.connection], [], [], 0)
        return not readable or handler.connection.recv(1, socket.MSG_PEEK) != b''
    except (OSError, ValueError):
        return False


def request_base_url(headers):
    """Absolute base URL for the download links handed back to MCP clients."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    proto = (headers.get('X-Forwarded-Proto') or 'http').split(',')[0].strip().lower()
    host = (headers.get('X-Forwarded-Host') or headers.get('Host') or '').split(',')[0].strip()
    if proto not in ('http', 'https'):
        proto = 'http'
    if not _HOST_RE.match(host):
        host = f'localhost:{EXTERNAL_PORT}'
    return f'{proto}://{host}'


def _file_entry(task_id, path, base_url):
    entry = {
        'name': path.name,
        'url': f'{base_url}/download/{task_id}/{quote(path.name)}',
        'mime_type': MIME_TYPES.get(path.suffix.lower(), 'application/octet-stream'),
    }
    try:
        entry['size_bytes'] = path.stat().st_size
    except OSError:
        pass
    return entry


def _finished_files(task_dir):
    """Files in a task directory that are deliverables rather than yt-dlp leftovers."""
    try:
        files = [p for p in task_dir.iterdir() if p.is_file() and not _INTERMEDIATE_RE.search(p.name)]
    except OSError:
        return []
    return sorted(files, key=lambda p: p.name)


def _read_transcript(path, max_chars):
    """Return (text, truncated, total_chars)."""
    text = path.read_text(encoding='utf-8', errors='replace')
    if len(text) > max_chars:
        return text[:max_chars], True, len(text)
    return text, False, len(text)


def _reported_progress_locked(entry):
    """A running task's progress on one 0-100 scale that never goes back between polls.

    The raw value restarts for each stream and phase (see _overall_progress), so a poll could
    otherwise see 100 while the job still runs, or 90 followed by 10. Hold downloads_lock.
    """
    value = max(_overall_progress(entry.get('progress'), entry.get('message'), entry.get('steps', ())),
                entry.get('reported_progress', 0.0))
    entry['reported_progress'] = value
    return round(value, 1)


def task_snapshot(task_id, base_url, transcript_chars=MCP_INLINE_TRANSCRIPT_CHARS):
    """Describe a task for an MCP client. Raises ToolError for an unknown or expired id."""
    task_dir = resolve_task_dir(task_id)
    if task_dir is None:
        raise ToolError(f'Invalid task_id {task_id!r}: expected the 8-character id returned when the job started.')
    with downloads_lock:
        live = active_downloads.get(task_id)
        entry = dict(live or {})
        if entry.get('status') == 'processing':
            entry['progress'] = _reported_progress_locked(live)

    status = entry.get('status')
    if status == 'processing':
        out = {
            'task_id': task_id,
            'status': 'processing',
            'progress': entry['progress'],
            'message': entry.get('message') or 'Processing...',
        }
        if entry.get('queue_position'):
            out['queue_position'] = entry['queue_position']
        return out
    if status == 'error':
        return {'task_id': task_id, 'status': 'error', 'error': entry.get('error') or 'Unknown error'}

    # Completed - or finished before a server restart emptied active_downloads, in which
    # case the files on disk are the record.
    if not task_dir.is_dir():
        raise ToolError(f'Unknown or expired task_id {task_id}. Files are deleted 1-2 hours after '
                        'the job finishes; start a new download.')
    found = {}  # role -> path: file (the original), vocals, background, transcript
    if entry.get('filename'):
        # The job's own record names every file it made (a title can end in "(Vocals)").
        for role, key, ext_ok in (('file', 'filename', MEDIA_EXTS), ('vocals', 'vocals_filename', MEDIA_EXTS),
                                  ('background', 'background_filename', MEDIA_EXTS),
                                  ('transcript', 'transcription_filename', tuple(TRANSCRIPT_EXTS))):
            named = resolve_task_file(task_id, entry.get(key) or '')
            if named is not None and named.is_file() and named.suffix.lower() in ext_ok:
                found[role] = named
    else:
        # No record (a server restart emptied active_downloads): guess from the file names.
        for path in _finished_files(task_dir):
            ext = path.suffix.lower()
            if ext in TRANSCRIPT_EXTS:
                found.setdefault('transcript', path)
            elif ext in MEDIA_EXTS:
                stem = next((s for s in STEMS if path.stem.endswith(f' ({STEM_LABELS[s]})')), None)
                found.setdefault(stem or 'file', path)
        if 'file' not in found and len([r for r in found if r in STEMS]) == 1:
            # A lone "(Vocals)" or "(Background)" file is more likely the original's own title.
            stem = next(r for r in found if r in STEMS)
            found['file'] = found.pop(stem)
    if 'file' not in found and 'transcript' not in found:
        raise ToolError(f'Task {task_id} has no finished files (it may have been interrupted). '
                        'Start a new download.')

    out = {'task_id': task_id, 'status': 'completed', 'progress': 100}
    for role, field in (('file', 'file'), ('vocals', 'vocals_file'), ('background', 'background_file')):
        if role in found:
            out[field] = _file_entry(task_id, found[role], base_url)
    transcript = found.get('transcript')
    if transcript is not None:
        out['transcript_file'] = _file_entry(task_id, transcript, base_url)
        out['transcript_format'] = TRANSCRIPT_EXTS[transcript.suffix.lower()]
        if transcript_chars:
            text, truncated, _ = _read_transcript(transcript, transcript_chars)
            out['transcript'] = text
            out['transcript_truncated'] = truncated
    for key in ('separation_error', 'transcription_error'):
        if entry.get(key):
            out[key] = entry[key]
    return out


def _overall_progress(raw, message, steps=()):
    """Map the per-phase progress a job reports onto one 0-100 scale.

    The raw value restarts per phase (yt-dlp download reaches 100, then extraction reports
    95, separation 96 and transcription 98), so the phase is taken from the status message.
    steps ('separate', 'transcribe') are the AI steps the job will run: they take most of the
    time, so they get most of the scale (download 0-30, then an equal share each, up to 98).
    """
    message = (message or '').lower()
    ai = [s for s in ('separate', 'transcribe') if s in steps]
    download_end = 30.0 if ai else 90.0
    spans, start = {}, download_end + 2
    for step in ai:
        spans[step] = (start, start + (98.0 - download_end - 2) / len(ai))
        start = spans[step][1]
    separate = spans.get('separate', (93.0, 94.0))
    transcribe = spans.get('transcribe', (95.0, 95.0))
    if 'transcri' in message:
        return transcribe[0]
    if 'saving the separated' in message:
        return separate[1]
    if 'separating' in message:  # "Separating vocals and background... 45%"
        match = re.search(r'(\d+)%', message)
        return separate[0] + (int(match.group(1)) / 100 if match else 0.0) * (separate[1] - separate[0])
    if 'waiting for another ai job' in message:  # raw 96 before separation, 98 before transcription
        return transcribe[0] if (raw or 0) >= 98 else separate[0]
    if 'convert' in message or 'merg' in message:
        return download_end + 1
    try:
        return min(download_end, float(raw or 0) * download_end / 100)
    except (TypeError, ValueError):
        return 0.0


def _wait_for_task(task_id, wait_seconds, progress, alive):
    """Block until the task stops processing or wait_seconds pass, reporting progress.

    progress(value, message) is called at most once a second on change, and at least every
    15 seconds regardless, so a client's request timeout keeps resetting during a long
    transcription. alive() is polled every half second; a hung-up client raises ClientGone.
    """
    deadline = time.monotonic() + wait_seconds
    last_key = None
    last_sent = 0.0
    while True:
        with downloads_lock:
            entry = active_downloads.get(task_id)
            if not entry or entry.get('status') != 'processing':
                return
            value, message = _reported_progress_locked(entry), entry.get('message')
        if alive is not None and not alive():
            raise ClientGone()
        now = time.monotonic()
        if progress is not None:
            key = (value, message)
            if (key != last_key and now - last_sent >= 1.0) or now - last_sent >= 15.0:
                progress(value, message or 'Processing...')
                last_key, last_sent = key, now
        if now >= deadline:
            return
        time.sleep(min(0.5, deadline - now))


def _wait_and_snapshot(task_id, wait_seconds, ctx, transcript_chars=MCP_INLINE_TRANSCRIPT_CHARS):
    if wait_seconds and _wait_slots.acquire(blocking=False):
        try:
            _wait_for_task(task_id, wait_seconds, ctx['progress'], ctx['alive'])
        finally:
            _wait_slots.release()
    snapshot = task_snapshot(task_id, ctx['base_url'], transcript_chars)
    if snapshot['status'] == 'processing':
        snapshot['next_step'] = (f'Still running. Call get_task_status with task_id "{task_id}" '
                                 f'(wait_seconds up to {MCP_MAX_WAIT}) to keep waiting.')
    return snapshot


def _start_job(start):
    """Start a job, or queue it when MAX_ACTIVE_JOBS are already running."""
    try:
        return start()
    except QueueFull as e:
        raise ToolError(str(e))


# --- Tool arguments -----------------------------------------------------------------

def _arg_video_id(args):
    url = args.get('url')
    if not isinstance(url, str) or not url.strip():
        raise ToolError('url is required: a YouTube video URL or an 11-character video id.')
    url = url.strip()
    video_id = url if re.fullmatch(r'[A-Za-z0-9_-]{11}', url) else extract_video_id(url)
    if not video_id:
        raise ToolError(f'Not a YouTube video URL: {url!r}. Use a youtube.com/watch?v=, youtu.be/, '
                        '/shorts/ or /embed/ link, or the 11-character video id.')
    return video_id


def _arg_choice(args, name, choices, default):
    """An integer option that also accepts '192', '192k' or '1080p'."""
    value = args.get(name)
    if value is None:
        return default
    match = None if isinstance(value, bool) else re.fullmatch(r'\s*(\d+)\s*(k|kbps|p)?\s*', str(value), re.I)
    if not match or int(match.group(1)) not in choices:
        raise ToolError(f'{name} must be one of {", ".join(map(str, choices))} (got {value!r}).')
    return int(match.group(1))


def _arg_bool(args, name, default=False):
    value = args.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ('true', 'false', 'yes', 'no', '1', '0'):
        return value.strip().lower() in ('true', 'yes', '1')
    raise ToolError(f'{name} must be true or false (got {value!r}).')


def _arg_int(args, name, default, lo, hi):
    value = args.get(name)
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            raise ValueError(value)
        number = int(float(value))
    except (TypeError, ValueError):
        raise ToolError(f'{name} must be a number from {lo} to {hi} (got {value!r}).')
    return max(lo, min(hi, number))


def _arg_enum(args, name, choices, default):
    value = args.get(name)
    if value is None:
        return default
    if not isinstance(value, str) or value.strip().lower() not in choices:
        raise ToolError(f'{name} must be one of {", ".join(choices)} (got {value!r}).')
    return value.strip().lower()


def _arg_stems(args):
    """remove_vocals / remove_background (also accepted without the underscore)."""
    def flag(name):
        alias = name.replace('_', '')
        return _arg_bool(args, name if args.get(name) is not None else alias)
    return stems_to_keep(remove_vocals=flag('remove_vocals'), remove_background=flag('remove_background'))


def _arg_task_id(args):
    task_id = args.get('task_id')
    task_id = task_id.strip().lower() if isinstance(task_id, str) else task_id
    if resolve_task_dir(task_id) is None:
        raise ToolError(f'task_id must be the 8-character id returned when the job started (got {task_id!r}).')
    return task_id


# --- Tools --------------------------------------------------------------------------

AUDIO_BITRATES = sorted(int(b) for b in VALID_BITRATES)
VIDEO_RESOLUTIONS = sorted(int(r) for r in VALID_RESOLUTIONS)


def _fmt_duration(seconds):
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}' if hours else f'{minutes}:{secs:02d}'


def _resolution_caps(max_height):
    """Resolution caps that give distinct results for a video whose tallest stream is max_height.

    download_video takes the best stream at or below the cap, so the smallest cap at or above
    max_height is the one that gets the best stream - e.g. 2160 for a 1080x1920 Short, or 360
    for a 240p-only upload.
    """
    if not max_height:
        return []
    caps = [r for r in VIDEO_RESOLUTIONS if r < max_height]
    top = next((r for r in VIDEO_RESOLUTIONS if r >= max_height), None)
    return caps + [top] if top else caps


def tool_get_video_info(args, ctx):
    video_id = _arg_video_id(args)
    url = canonical_youtube_url(video_id)
    info = fetch_video_metadata(url)
    duration = int(info.get('duration') or 0)
    heights = [f['height'] for f in info.get('formats') or []
               if isinstance(f.get('height'), int) and f.get('vcodec') != 'none']
    max_height = max(heights, default=0)
    out = {
        'video_id': video_id,
        'url': url,
        'title': info.get('title') or 'Unknown',
        'channel': info.get('channel') or info.get('uploader') or '',
        'duration_seconds': duration,
        'duration': _fmt_duration(duration),
        'thumbnail': info.get('thumbnail') or '',
        'available_resolutions': _resolution_caps(max_height),
        'audio_bitrates': AUDIO_BITRATES,
    }
    upload_date = info.get('upload_date')
    if isinstance(upload_date, str) and re.fullmatch(r'\d{8}', upload_date):
        out['upload_date'] = f'{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}'
    if isinstance(info.get('view_count'), int):
        out['view_count'] = info['view_count']
    if info.get('is_live'):
        out['is_live'] = True
    return out


def tool_download_audio(args, ctx):
    video_id = _arg_video_id(args)
    bitrate = _arg_choice(args, 'bitrate', AUDIO_BITRATES, 320)
    transcribe = _arg_bool(args, 'transcribe')
    srt = _arg_enum(args, 'transcript_format', ('srt', 'text'), 'srt') == 'srt'
    keep = _arg_stems(args)
    wait = _arg_int(args, 'wait_seconds', MCP_DEFAULT_WAIT, 0, MCP_MAX_WAIT)
    task_id = _start_job(lambda: start_audio_task(video_id, str(bitrate), transcribe, srt, keep))
    print(f'[MCP] download_audio {video_id} {bitrate}k transcribe={transcribe} stems={keep} -> task {task_id}')
    return _wait_and_snapshot(task_id, wait, ctx)


def tool_download_video(args, ctx):
    video_id = _arg_video_id(args)
    resolution = _arg_choice(args, 'resolution', VIDEO_RESOLUTIONS, 1080)
    transcribe = _arg_bool(args, 'transcribe')
    srt = _arg_enum(args, 'transcript_format', ('srt', 'text'), 'srt') == 'srt'
    keep = _arg_stems(args)
    wait = _arg_int(args, 'wait_seconds', MCP_DEFAULT_WAIT, 0, MCP_MAX_WAIT)
    task_id = _start_job(lambda: start_video_task(video_id, str(resolution), transcribe, srt, keep))
    print(f'[MCP] download_video {video_id} {resolution}p transcribe={transcribe} stems={keep} -> task {task_id}')
    return _wait_and_snapshot(task_id, wait, ctx)


def tool_transcribe_video(args, ctx):
    video_id = _arg_video_id(args)
    # Before 1.2.0 this argument was called "format"; older clients still send that name.
    key = 'format' if args.get('transcript_format') is None and args.get('format') is not None else 'transcript_format'
    srt = _arg_enum(args, key, ('srt', 'text'), 'srt') == 'srt'
    wait = _arg_int(args, 'wait_seconds', MCP_DEFAULT_WAIT, 0, MCP_MAX_WAIT)
    # Whisper resamples to 16 kHz mono, so the lowest bitrate loses nothing.
    task_id = _start_job(lambda: start_audio_task(video_id, '64', True, srt))
    print(f'[MCP] transcribe_video {video_id} format={"srt" if srt else "text"} -> task {task_id}')
    return _wait_and_snapshot(task_id, wait, ctx)


def tool_get_task_status(args, ctx):
    task_id = _arg_task_id(args)
    wait = _arg_int(args, 'wait_seconds', MCP_DEFAULT_WAIT, 0, MCP_MAX_WAIT)
    task_snapshot(task_id, ctx['base_url'], 0)  # unknown or expired id: fail before waiting
    return _wait_and_snapshot(task_id, wait, ctx)


def tool_get_transcript(args, ctx):
    task_id = _arg_task_id(args)
    max_chars = _arg_int(args, 'max_chars', 100000, 1000, 2000000)
    snapshot = task_snapshot(task_id, ctx['base_url'], 0)
    if snapshot['status'] == 'processing':
        raise ToolError(f'Task {task_id} is still running ({snapshot["progress"]}%: {snapshot["message"]}). '
                        'Call get_task_status with wait_seconds to wait for it.')
    if snapshot['status'] == 'error':
        raise ToolError(f'Task {task_id} failed: {snapshot["error"]}')
    if 'transcript_file' not in snapshot:
        reason = snapshot.get('transcription_error')
        raise ToolError(f'Task {task_id} has no transcript. ' + (
            f'Transcription failed: {reason}' if reason else
            'Use transcribe_video, or download_audio / download_video with transcribe=true.'))
    path = resolve_task_file(task_id, snapshot['transcript_file']['name'])
    text, truncated, total = _read_transcript(path, max_chars)
    return {
        'task_id': task_id,
        'format': snapshot['transcript_format'],
        'transcript': text,
        'truncated': truncated,
        'total_chars': total,
        'file': snapshot['transcript_file'],
    }


def tool_list_downloads(args, ctx):
    dirs = []
    try:
        for d in DOWNLOAD_DIR.iterdir():
            if TASK_ID_RE.match(d.name):
                try:
                    if d.is_dir():
                        dirs.append((d.stat().st_mtime, d))
                except OSError:  # deleted while listing
                    pass
    except OSError:
        pass
    dirs.sort(key=lambda item: item[0], reverse=True)
    rows = []
    for mtime, task_dir in dirs[:200]:
        task_id = task_dir.name
        with downloads_lock:
            live = active_downloads.get(task_id)
            entry = dict(live or {})
            if entry.get('status') == 'processing':
                entry['progress'] = _reported_progress_locked(live)
        modified = datetime.fromtimestamp(mtime, timezone.utc)
        row = {'task_id': task_id, 'modified': modified.isoformat(timespec='seconds'), 'files': []}
        if entry.get('status') == 'processing':
            row.update(status='processing', progress=entry['progress'],
                       message=entry.get('message') or 'Processing...')
        elif entry.get('status') == 'error':
            row.update(status='error', error=entry.get('error') or 'Unknown error')
        else:
            files = _finished_files(task_dir)
            # The names a job recorded beat the leftover-file heuristic (a title can
            # legitimately end in '.f1').
            for key in ('filename', 'vocals_filename', 'background_filename', 'transcription_filename'):
                named = resolve_task_file(task_id, entry.get(key) or '')
                if named is not None and named.is_file() and named not in files:
                    files.append(named)
            files.sort(key=lambda p: p.name)
            row['status'] = 'completed' if files else 'incomplete'
            row['files'] = [_file_entry(task_id, p, ctx['base_url']) for p in files]
        rows.append(row)
    return {'downloads': rows, 'count': len(rows)}


def tool_delete_download(args, ctx):
    task_id = _arg_task_id(args)
    if cancel_queued_task(task_id):
        print(f'[MCP] delete_download {task_id}: cancelled while waiting in the queue')
        return {'task_id': task_id, 'deleted': True}
    with downloads_lock:
        entry = active_downloads.get(task_id)
        if entry and entry.get('status') == 'processing':
            raise ToolError(f'Task {task_id} is still running. Wait for it with get_task_status, then delete it.')
        active_downloads.pop(task_id, None)
    task_dir = resolve_task_dir(task_id)
    existed = task_dir.is_dir()
    if existed:
        shutil.rmtree(task_dir)
    print(f'[MCP] delete_download {task_id} deleted={existed}')
    return {'task_id': task_id, 'deleted': existed}


# --- Tool definitions ---------------------------------------------------------------

_URL_ARG = {'type': 'string', 'description': 'YouTube video URL (youtube.com/watch?v=..., youtu.be/..., '
            '/shorts/... or /embed/...) or a bare 11-character video id.'}
_WAIT_ARG = {'type': 'integer', 'minimum': 0, 'maximum': MCP_MAX_WAIT, 'default': MCP_DEFAULT_WAIT,
             'description': 'Seconds to wait for the job to finish. If it is still running after that, the '
             'result has status "processing" and a task_id to pass to get_task_status.'}
_TASK_ID_ARG = {'type': 'string', 'pattern': '^[0-9a-f]{8}$',
                'description': 'The 8-character task_id returned by download_audio, download_video or transcribe_video.'}
_TRANSCRIBE_ARG = {'type': 'boolean', 'default': False,
                   'description': 'Also transcribe the audio with Whisper. Allow several minutes '
                   'for a long video.'}
_TRANSCRIPT_FORMAT_ARG = {'type': 'string', 'enum': ['srt', 'text'], 'default': 'srt',
                          'description': '"srt" for subtitles with timestamps, "text" for plain text.'}
_DOWNLOAD_TRANSCRIPT_FORMAT_ARG = dict(_TRANSCRIPT_FORMAT_ARG, description=_TRANSCRIPT_FORMAT_ARG['description']
                                       + ' Only used when transcribe is true.')
_JOB_STEPS = (f'Starts a job: waits up to wait_seconds (default {MCP_DEFAULT_WAIT}), then returns task_id and '
              'status. If status is "processing", call get_task_status with the task_id.')
_REMOVE_VOCALS_ARG = {'type': 'boolean', 'default': False,
                      'description': 'Also save a copy without the vocals (background or instrumental only), '
                      'separated with UVR MDX-Net. It is returned as background_file.'}
_REMOVE_BACKGROUND_ARG = {'type': 'boolean', 'default': False,
                          'description': 'Also save a copy with only the vocals (the background removed), '
                          'separated with UVR MDX-Net. It is returned as vocals_file.'}
_SEPARATION_STEPS = (' Set remove_vocals=true for a copy without the vocals (background_file) and/or '
                     'remove_background=true for a vocals-only copy (vocals_file); this can take about as '
                     'long as the audio itself.')

_FILE_OUT = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'url': {'type': 'string', 'description': 'Direct HTTP download link.'},
        'mime_type': {'type': 'string'},
        'size_bytes': {'type': 'integer'},
    },
    'required': ['name', 'url', 'mime_type'],
}

_TASK_OUT = {
    'type': 'object',
    'properties': {
        'task_id': {'type': 'string'},
        'status': {'type': 'string', 'enum': ['processing', 'completed', 'error']},
        'progress': {'type': 'number', 'description': 'Percent complete, 0-100.'},
        'message': {'type': 'string'},
        'error': {'type': 'string'},
        'file': dict(_FILE_OUT, description='The downloaded MP3 or MP4, unchanged.'),
        'vocals_file': dict(_FILE_OUT, description='Vocals only (remove_background=true).'),
        'background_file': dict(_FILE_OUT, description='Background only, without the vocals '
                                '(remove_vocals=true).'),
        'separation_error': {'type': 'string', 'description': 'Set when the download succeeded but '
                             'the vocal separation failed.'},
        'transcript_file': _FILE_OUT,
        'transcript_format': {'type': 'string', 'enum': ['srt', 'text']},
        'transcript': {'type': 'string', 'description': f'Transcript text, up to '
                       f'{MCP_INLINE_TRANSCRIPT_CHARS:,} characters - see transcript_truncated. '
                       'get_transcript returns more.'},
        'transcript_truncated': {'type': 'boolean'},
        'transcription_error': {'type': 'string', 'description': 'Set when the download succeeded but '
                                'transcription failed.'},
        'queue_position': {'type': 'integer', 'description': 'Place in line while the job waits for a '
                           'free slot. It starts automatically.'},
        'next_step': {'type': 'string'},
    },
    'required': ['task_id', 'status'],
}

_STARTS_JOB = {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': False, 'openWorldHint': True}
_READS_TASKS = {'readOnlyHint': True, 'openWorldHint': False}

MCP_TOOL_DEFS = [
    {
        'name': 'get_video_info',
        'title': 'Get YouTube video info',
        'description': 'Look up a YouTube video without downloading it: title, channel, duration, '
                       'thumbnail, and which MP4 resolutions and MP3 bitrates can be requested.',
        'inputSchema': {'type': 'object', 'properties': {'url': _URL_ARG}, 'required': ['url']},
        'outputSchema': {
            'type': 'object',
            'properties': {
                'video_id': {'type': 'string'},
                'url': {'type': 'string'},
                'title': {'type': 'string'},
                'channel': {'type': 'string'},
                'duration_seconds': {'type': 'integer'},
                'duration': {'type': 'string', 'description': 'H:MM:SS or M:SS'},
                'thumbnail': {'type': 'string'},
                'upload_date': {'type': 'string', 'description': 'YYYY-MM-DD'},
                'view_count': {'type': 'integer'},
                'is_live': {'type': 'boolean'},
                'available_resolutions': {'type': 'array', 'items': {'type': 'integer'}},
                'audio_bitrates': {'type': 'array', 'items': {'type': 'integer'}},
            },
            'required': ['video_id', 'url', 'title', 'duration_seconds', 'available_resolutions',
                         'audio_bitrates'],
        },
        'annotations': {'title': 'Get YouTube video info', 'readOnlyHint': True, 'openWorldHint': True},
    },
    {
        'name': 'download_audio',
        'title': 'Download YouTube audio as MP3',
        'description': "Download a YouTube video's audio as an MP3 (bitrate in kbps, default 320). Set "
                       'transcribe=true to also get a Whisper transcript (transcript_format "srt", the default, '
                       'or "text"); if you only need the words, use transcribe_video.' + _SEPARATION_STEPS +
                       ' ' + _JOB_STEPS + ' When completed, file.url is the MP3 link, plus vocals_file, '
                       'background_file, transcript_file and transcript as requested. Files are deleted 1-2 '
                       'hours after the job finishes.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'url': _URL_ARG,
                'bitrate': {'type': 'integer', 'enum': AUDIO_BITRATES, 'default': 320,
                            'description': 'MP3 bitrate in kbps.'},
                'transcribe': _TRANSCRIBE_ARG,
                'transcript_format': _DOWNLOAD_TRANSCRIPT_FORMAT_ARG,
                'remove_vocals': _REMOVE_VOCALS_ARG,
                'remove_background': _REMOVE_BACKGROUND_ARG,
                'wait_seconds': _WAIT_ARG,
            },
            'required': ['url'],
        },
        'outputSchema': _TASK_OUT,
        'annotations': dict(_STARTS_JOB, title='Download YouTube audio as MP3'),
    },
    {
        'name': 'download_video',
        'title': 'Download YouTube video as MP4',
        'description': 'Download a YouTube video as an MP4 at up to resolution (maximum height in pixels, '
                       'default 1080); if the video has no stream that high, a lower one is used '
                       '(get_video_info lists available_resolutions). Set transcribe=true to also get a '
                       'Whisper transcript (transcript_format "srt", the default, or "text").' +
                       _SEPARATION_STEPS + ' The separated copies are MP4s with the same video. ' + _JOB_STEPS +
                       ' When completed, file.url is the MP4 link, plus vocals_file, background_file, '
                       'transcript_file and transcript as requested. Files are deleted 1-2 hours after the job '
                       'finishes.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'url': _URL_ARG,
                'resolution': {'type': 'integer', 'enum': VIDEO_RESOLUTIONS, 'default': 1080,
                               'description': 'Maximum video height in pixels.'},
                'transcribe': _TRANSCRIBE_ARG,
                'transcript_format': _DOWNLOAD_TRANSCRIPT_FORMAT_ARG,
                'remove_vocals': _REMOVE_VOCALS_ARG,
                'remove_background': _REMOVE_BACKGROUND_ARG,
                'wait_seconds': _WAIT_ARG,
            },
            'required': ['url'],
        },
        'outputSchema': _TASK_OUT,
        'annotations': dict(_STARTS_JOB, title='Download YouTube video as MP4'),
    },
    {
        'name': 'transcribe_video',
        'title': 'Transcribe a YouTube video',
        'description': 'Transcribe a YouTube video with Whisper. Use it when you need the words, not the '
                       'media file. transcript_format: "srt" (subtitles with timestamps, default) or "text" '
                       '(plain text). ' + _JOB_STEPS + ' A long video can take several minutes; progress '
                       'stays at 95 while Whisper runs (read message). When completed, transcript holds the '
                       f'text (up to {MCP_INLINE_TRANSCRIPT_CHARS:,} characters, see transcript_truncated; '
                       'get_transcript returns more) and transcript_file links to the .srt/.txt file. file is the '
                       'low-bitrate MP3 that was transcribed. If transcription_error is set, transcription '
                       'failed.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'url': _URL_ARG,
                'transcript_format': _TRANSCRIPT_FORMAT_ARG,
                'wait_seconds': _WAIT_ARG,
            },
            'required': ['url'],
        },
        'outputSchema': _TASK_OUT,
        'annotations': dict(_STARTS_JOB, title='Transcribe a YouTube video'),
    },
    {
        'name': 'get_task_status',
        'title': 'Get job status',
        'description': 'Wait for, or check, a job started by download_audio, download_video or '
                       f'transcribe_video. Waits up to wait_seconds (default {MCP_DEFAULT_WAIT}, max '
                       f'{MCP_MAX_WAIT}; 0 = return at once) and returns the same fields as the tool that '
                       'started the job: status "processing" (with progress 0-100, and queue_position while '
                       'queued), "completed" (file, vocals_file, background_file and transcript_file links, '
                       'and transcript, as requested), or "error" (see error). Call it again while status is "processing"; it '
                       'only reads the job and never restarts it.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'task_id': _TASK_ID_ARG,
                'wait_seconds': dict(_WAIT_ARG, description='Seconds to wait for the job to finish before '
                                     'returning its current status. 0 returns immediately.'),
            },
            'required': ['task_id'],
        },
        'outputSchema': _TASK_OUT,
        'annotations': dict(_READS_TASKS, title='Get job status'),
    },
    {
        'name': 'get_transcript',
        'title': 'Get transcript text',
        'description': 'Return the transcript text (SRT or plain text) of a completed job that was '
                       'transcribed (transcribe_video, or a download with transcribe=true). Use it when a '
                       'result had transcript_truncated=true, or to re-read a transcript later. max_chars '
                       '(default 100000) caps the text returned; truncated and total_chars show whether more '
                       'exists. Also returns file, a direct download link to the transcript file.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'task_id': _TASK_ID_ARG,
                'max_chars': {'type': 'integer', 'minimum': 1000, 'maximum': 2000000, 'default': 100000,
                              'description': 'Truncate the transcript to this many characters.'},
            },
            'required': ['task_id'],
        },
        'outputSchema': {
            'type': 'object',
            'properties': {
                'task_id': {'type': 'string'},
                'format': {'type': 'string', 'enum': ['srt', 'text']},
                'transcript': {'type': 'string'},
                'truncated': {'type': 'boolean'},
                'total_chars': {'type': 'integer'},
                'file': _FILE_OUT,
            },
            'required': ['task_id', 'format', 'transcript', 'truncated', 'total_chars', 'file'],
        },
        'annotations': dict(_READS_TASKS, title='Get transcript text'),
    },
    {
        'name': 'list_downloads',
        'title': 'List downloads',
        'description': 'List the jobs whose files are still on the server, newest first, with their status '
                       'and download links. Files are deleted 1-2 hours after the job finishes.',
        'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
        'outputSchema': {
            'type': 'object',
            'properties': {
                'downloads': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'task_id': {'type': 'string'},
                            'status': {'type': 'string',
                                       'enum': ['processing', 'completed', 'error', 'incomplete']},
                            'modified': {'type': 'string', 'description': 'ISO 8601 UTC timestamp.'},
                            'files': {'type': 'array', 'items': _FILE_OUT},
                            'progress': {'type': 'number'},
                            'message': {'type': 'string'},
                            'error': {'type': 'string'},
                        },
                        'required': ['task_id', 'status', 'modified', 'files'],
                    },
                },
                'count': {'type': 'integer'},
            },
            'required': ['downloads', 'count'],
        },
        'annotations': dict(_READS_TASKS, title='List downloads'),
    },
    {
        'name': 'delete_download',
        'title': 'Delete a download',
        'description': 'Delete a finished job\'s files from the server, or cancel a job that is still waiting '
                       'in the queue. A job that is already running cannot be deleted.',
        'inputSchema': {'type': 'object', 'properties': {'task_id': _TASK_ID_ARG}, 'required': ['task_id']},
        'outputSchema': {
            'type': 'object',
            'properties': {
                'task_id': {'type': 'string'},
                'deleted': {'type': 'boolean', 'description': 'False if there was nothing to delete.'},
            },
            'required': ['task_id', 'deleted'],
        },
        'annotations': {'title': 'Delete a download', 'readOnlyHint': False, 'destructiveHint': True,
                        'idempotentHint': True, 'openWorldHint': False},
    },
]

# name -> (implementation, long_running). Long-running tools answer over SSE when the
# client accepts it, so progress notifications can flow while they wait.
MCP_TOOLS = {
    'get_video_info': (tool_get_video_info, False),
    'download_audio': (tool_download_audio, True),
    'download_video': (tool_download_video, True),
    'transcribe_video': (tool_transcribe_video, True),
    'get_task_status': (tool_get_task_status, True),
    'get_transcript': (tool_get_transcript, False),
    'list_downloads': (tool_list_downloads, False),
    'delete_download': (tool_delete_download, False),
}
assert [t['name'] for t in MCP_TOOL_DEFS] == list(MCP_TOOLS)


def _run_tool(name, args, ctx):
    """Run a tool and build its CallToolResult."""
    implementation, _ = MCP_TOOLS[name]
    try:
        structured = implementation(args, ctx)
    except ClientGone:
        raise
    except ToolError as e:
        return {'content': [{'type': 'text', 'text': str(e)}], 'isError': True}
    except Exception as e:
        print(f'[MCP] {name} failed: {e!r}')
        detail = str(e).strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        return {'content': [{'type': 'text', 'text': f'{name} failed: {detail}'}], 'isError': True}
    return {
        'content': [{'type': 'text', 'text': json.dumps(structured, ensure_ascii=False, indent=2)}],
        'structuredContent': structured,
        'isError': structured.get('status') == 'error',
    }


# --- JSON-RPC over HTTP -------------------------------------------------------------

def _rpc_error(req_id, code, message, data=None):
    error = {'code': code, 'message': message}
    if data is not None:
        error['data'] = data
    response = {'jsonrpc': '2.0', 'error': error}
    if req_id is not None:
        response['id'] = req_id
    return response


def _send_cors(handler, origin):
    if origin:
        handler.send_header('Access-Control-Allow-Origin', origin)
        handler.send_header('Vary', 'Origin')
        handler.send_header('Access-Control-Expose-Headers', 'WWW-Authenticate')


def _mcp_send_json(handler, status, payload, origin, extra_headers=None):
    body = b'' if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    if payload is not None:
        handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(body)))
    for name, value in (extra_headers or {}).items():
        handler.send_header(name, value)
    _send_cors(handler, origin)
    handler.end_headers()
    if body:
        handler.wfile.write(body)


class _SSEStream:
    """A text/event-stream response scoped to one request."""

    def __init__(self, handler, progress_token, origin):
        self.handler = handler
        self.token = progress_token
        self.last_progress = -1.0
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream')
        handler.send_header('Cache-Control', 'no-cache')
        handler.send_header('X-Accel-Buffering', 'no')
        handler.send_header('Connection', 'close')
        _send_cors(handler, origin)
        handler.end_headers()
        handler.close_connection = True

    def _write(self, chunk):
        try:
            self.handler.wfile.write(chunk)
            self.handler.wfile.flush()
        except OSError as e:  # BrokenPipe, ConnectionReset, ConnectionAborted
            raise ClientGone() from e

    def send(self, message):
        # ASCII-only: line splitters such as httpx's also break on U+2028, U+2029 and
        # U+0085, which ensure_ascii=False would leave raw inside a video title.
        data = json.dumps(message).encode('ascii')
        self._write(b'event: message\ndata: ' + data + b'\n\n')

    def progress(self, value, message):
        if self.token is None:
            # No progressToken: send an SSE comment, which keeps proxies from idling the
            # connection out and surfaces a client disconnect.
            self._write(b': keepalive\n\n')
            return
        # Progress must strictly increase, even when a phase reports no movement.
        value = round(max(value, self.last_progress + 0.01), 2)
        self.last_progress = value
        self.send({'jsonrpc': '2.0', 'method': 'notifications/progress',
                   'params': {'progressToken': self.token, 'progress': value, 'total': 100,
                              'message': message}})


def _decode_header_value(value):
    """Undo the =?base64?...?= sentinel encoding used for non-ASCII Mcp-Name values."""
    if value.startswith('=?base64?') and value.endswith('?=') and len(value) >= 11:
        try:
            return base64.b64decode(value[9:-2], validate=True).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return None
    return value


def _check_mirrored_header(headers, name, expected, decode=False):
    value = headers.get(name)
    if value is None:
        raise McpError(-32020, f'Header mismatch: required header {name} is missing', 400)
    if decode:
        value = _decode_header_value(value)
        if value is None:
            raise McpError(-32020, f'Header mismatch: {name} header is not valid base64', 400)
    if value != expected:
        raise McpError(-32020, f'Header mismatch: {name} header value {value!r} does not match '
                       f'body value {expected!r}', 400)


def _check_request_version(headers, method, params):
    """Validate protocol metadata and return True for a modern (2026-07-28) request."""
    meta = params.get('_meta')
    if meta is not None and not isinstance(meta, dict):
        raise McpError(-32602, '_meta must be an object', 400)
    meta = meta or {}
    header_version = headers.get('MCP-Protocol-Version')
    body_version = meta.get(META_PROTOCOL_VERSION)

    if body_version is None and header_version not in MCP_MODERN_VERSIONS:
        # Legacy era: the version is the negotiated one in the header (absent means
        # 2025-03-26) and requests carry no _meta to validate.
        if header_version is not None and header_version not in MCP_LEGACY_VERSIONS:
            raise McpError(-32022, 'Unsupported protocol version', 400,
                           {'supported': MCP_SUPPORTED_VERSIONS, 'requested': header_version})
        return False

    if not isinstance(body_version, str):
        raise McpError(-32602, f'Missing required _meta field {META_PROTOCOL_VERSION}', 400)
    if not isinstance(meta.get(META_CLIENT_CAPABILITIES), dict):
        raise McpError(-32602, f'Missing required _meta field {META_CLIENT_CAPABILITIES}', 400)
    _check_mirrored_header(headers, 'MCP-Protocol-Version', body_version)
    if body_version not in MCP_SUPPORTED_VERSIONS:
        raise McpError(-32022, 'Unsupported protocol version', 400,
                       {'supported': MCP_SUPPORTED_VERSIONS, 'requested': body_version})
    _check_mirrored_header(headers, 'Mcp-Method', method)
    if method in ('tools/call', 'prompts/get'):
        _check_mirrored_header(headers, 'Mcp-Name', params.get('name'), decode=True)
    elif method == 'resources/read':
        _check_mirrored_header(headers, 'Mcp-Name', params.get('uri'), decode=True)
    return body_version in MCP_MODERN_VERSIONS


def _shape_result(result, modern):
    """Modern results carry resultType and identify the server in _meta."""
    if not modern:
        return result
    shaped = {'resultType': 'complete'}
    shaped.update(result)
    meta = dict(shaped.get('_meta') or {})
    meta[META_SERVER_INFO] = SERVER_INFO
    shaped['_meta'] = meta
    return shaped


def _mcp_tools_call(handler, req_id, params, modern, origin):
    name = params.get('name')
    if not isinstance(name, str) or name not in MCP_TOOLS:
        raise McpError(-32602, f'Unknown tool: {name}')
    args = params.get('arguments')
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise McpError(-32602, 'arguments must be an object')
    token = (params.get('_meta') or {}).get('progressToken')
    if isinstance(token, bool) or not isinstance(token, (str, int)):
        token = None

    stream = None
    if MCP_TOOLS[name][1] and 'text/event-stream' in (handler.headers.get('Accept') or ''):
        stream = _SSEStream(handler, token, origin)
    ctx = {
        'base_url': request_base_url(handler.headers),
        'progress': stream.progress if stream else None,
        'alive': lambda: client_connected(handler),
    }
    response = {'jsonrpc': '2.0', 'id': req_id, 'result': _shape_result(_run_tool(name, args, ctx), modern)}
    if stream:
        stream.send(response)
    else:
        _mcp_send_json(handler, 200, response, origin)


def handle_mcp_post(handler):
    """POST /mcp: one JSON-RPC request or notification per HTTP request."""
    origin = handler.headers.get('Origin')
    cors = origin if mcp_origin_allowed(origin) else None
    # Read the body before any early rejection: closing with unread request bytes makes
    # the kernel send a TCP RST, and the client never sees the 401/403.
    try:
        length = int(handler.headers.get('Content-Length', ''))
    except ValueError:
        _mcp_send_json(handler, 411, _rpc_error(None, -32600, 'Content-Length required'), cors)
        return
    if length < 0 or length > MCP_MAX_BODY_BYTES:
        _mcp_send_json(handler, 413, _rpc_error(None, -32600, 'Request body too large'), cors)
        return
    body = handler.rfile.read(length)

    if cors is None and origin is not None:
        _mcp_send_json(handler, 403, _rpc_error(None, -32600, 'Origin not allowed'), None)
        return
    if MCP_CONFIG_ERROR:
        _mcp_send_json(handler, 503, _rpc_error(None, -32603, 'MCP server is misconfigured; see server log'),
                       origin)
        return
    try:
        mcp_authorize(handler.headers)
    except AuthFailure as e:
        _mcp_send_json(handler, e.status, {'error': e.error, 'error_description': e.description}, origin,
                       {'WWW-Authenticate': e.challenge} if e.challenge else None)
        return

    try:
        message = json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        _mcp_send_json(handler, 400, _rpc_error(None, -32700, 'Parse error'), origin)
        return

    if isinstance(message, list):
        _mcp_send_json(handler, 400, _rpc_error(None, -32600, 'Batch requests are not supported'), origin)
        return
    if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
        _mcp_send_json(handler, 400, _rpc_error(None, -32600, 'Invalid Request'), origin)
        return
    method = message.get('method')
    if method is None or 'id' not in message:
        # A notification (e.g. legacy notifications/initialized), or a response to a
        # server request - this server never sends any. Either way there is nothing to answer.
        _mcp_send_json(handler, 202, None, origin)
        return
    req_id = message['id']
    if not isinstance(method, str) or isinstance(req_id, bool) or not isinstance(req_id, (str, int)):
        _mcp_send_json(handler, 400, _rpc_error(None, -32600, 'Invalid Request'), origin)
        return
    params = message.get('params')
    if params is None:
        params = {}

    try:
        if not isinstance(params, dict):
            raise McpError(-32602, 'params must be an object', 400)
        modern = _check_request_version(handler.headers, method, params)
        if method == 'tools/call':
            _mcp_tools_call(handler, req_id, params, modern, origin)
            return
        if method == 'initialize' and not modern:
            requested = params.get('protocolVersion')
            result = {
                'protocolVersion': requested if requested in MCP_LEGACY_VERSIONS else MCP_LEGACY_VERSIONS[0],
                'capabilities': MCP_CAPABILITIES,
                'serverInfo': SERVER_INFO,
                'instructions': MCP_INSTRUCTIONS,
            }
        elif method == 'server/discover':
            result = {'supportedVersions': MCP_SUPPORTED_VERSIONS, 'capabilities': MCP_CAPABILITIES,
                      'instructions': MCP_INSTRUCTIONS, **MCP_CACHE_HINTS}
        elif method == 'ping':
            result = {}
        elif method == 'tools/list':
            result = {'tools': MCP_TOOL_DEFS}
            if modern:
                result.update(MCP_CACHE_HINTS)
        else:
            # The modern transport signals an unknown method with HTTP 404; legacy clients
            # expect the JSON-RPC error on a 200.
            raise McpError(-32601, f'Method not found: {method}', 404 if modern else 200)
        _mcp_send_json(handler, 200, {'jsonrpc': '2.0', 'id': req_id, 'result': _shape_result(result, modern)},
                       origin)
    except McpError as e:
        _mcp_send_json(handler, e.http_status, _rpc_error(req_id, e.code, e.message, e.data), origin)
    except ClientGone:
        print(f'[MCP] Client closed the stream for request {req_id!r}; any started job keeps running.')


def handle_mcp_preflight(handler):
    """CORS preflight for browser-based MCP clients (e.g. MCP Inspector)."""
    origin = handler.headers.get('Origin')
    if not mcp_origin_allowed(origin):
        handler.send_response(403)
        handler.send_header('Content-Length', '0')
        handler.end_headers()
        return
    handler.send_response(204)
    if origin:
        handler.send_header('Access-Control-Allow-Origin', origin)
        handler.send_header('Vary', 'Origin')
        handler.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        handler.send_header('Access-Control-Allow-Headers',
                            handler.headers.get('Access-Control-Request-Headers')
                            or 'Content-Type, Authorization, MCP-Protocol-Version, Mcp-Method, Mcp-Name')
        handler.send_header('Access-Control-Max-Age', '600')
    handler.send_header('Content-Length', '0')
    handler.end_headers()


def send_mcp_method_not_allowed(handler):
    """GET and DELETE on /mcp: no standalone SSE stream and no sessions to terminate."""
    handler.send_response(405)
    handler.send_header('Allow', 'POST, OPTIONS')
    handler.send_header('Content-Length', '0')
    handler.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


# Site icons. Public even with WEB_AUTH_PASSWORD set: none of them hold anything private.
STATIC_DIR = Path(__file__).parent / 'static'
APP_ASSETS = {
    'icon-192.png': 'image/png',
    'apple-touch-icon.png': 'image/png',
    'favicon.ico': 'image/x-icon',
}
APP_ASSET_ALIASES = {'/favicon.ico': 'favicon.ico', '/apple-touch-icon.png': 'apple-touch-icon.png'}


def web_auth_ok(headers):
    """True when web UI auth is off, or the request carries the configured Basic credentials."""
    if not WEB_AUTH_PASSWORD:
        return True
    scheme, _, value = (headers.get('Authorization') or '').partition(' ')
    if scheme.lower() != 'basic':
        return False
    try:
        user, sep, password = base64.b64decode(value.strip(), validate=True).decode('utf-8').partition(':')
    except (ValueError, UnicodeDecodeError):
        return False
    user_ok = hmac.compare_digest(user.encode('utf-8'), WEB_AUTH_USER.encode('utf-8'))
    password_ok = hmac.compare_digest(password.encode('utf-8'), WEB_AUTH_PASSWORD.encode('utf-8'))
    return bool(sep) and user_ok and password_ok


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
    
    def serve_app_asset(self, path):
        """Serve a site icon. Returns False if path is not one of them."""
        name = APP_ASSET_ALIASES.get(path) or (path[len('/static/'):] if path.startswith('/static/') else None)
        if name not in APP_ASSETS:  # a fixed list, so no path ever reaches the filesystem
            return False
        try:
            body = (STATIC_DIR / name).read_bytes()
        except OSError:
            self.send_error(404, 'Not found')
            return True
        content_type = APP_ASSETS[name]
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'public, max-age=86400')
        self.end_headers()
        self.wfile.write(body)
        return True

    def require_web_auth(self, path):
        """Gate the web UI and /api behind WEB_AUTH_PASSWORD. Sends the 401 and returns False on failure."""
        if path.startswith('/download/') or web_auth_ok(self.headers):
            return True
        body = b'Authentication required'
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="ytdl-web", charset="UTF-8"')
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if MCP_ENABLED and path in MCP_PATHS:
            send_mcp_method_not_allowed(self)
            return

        # OAuth 2.0 Protected Resource Metadata (RFC 9728), only when MCP_AUTH=oauth
        if MCP_ENABLED and MCP_AUTH == 'oauth' and not MCP_CONFIG_ERROR and path in MCP_PRM_PATHS:
            body = json.dumps(protected_resource_metadata()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
            return
        
        if self.serve_app_asset(path):
            return

        if not self.require_web_auth(path):
            return

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
        
        # API: What the Settings page shows
        if path == '/api/settings':
            self.send_json(settings_payload())
            return

        # API: Check task status
        if path.startswith('/api/check/'):
            task_id = path.split('/')[-1]
            with downloads_lock:
                live = active_downloads.get(task_id)
                task = dict(live) if live else {'status': 'not_found'}
                if task.get('status') == 'processing':
                    # One 0-100 scale for the page's progress bar that never goes back; the raw
                    # 'progress' restarts for each stream and step.
                    task['overall_progress'] = _reported_progress_locked(live)
            self.send_json(task)
            return

        # API: Check if file exists for a task
        if path.startswith('/api/file-exists/'):
            task_dir = resolve_task_dir(path.split('/')[-1])
            exists = task_dir is not None and task_dir.is_dir() and any(task_dir.iterdir())
            self.send_json({'exists': exists})
            return

        # Serve download files
        if path.startswith('/download/'):
            parts = path.split('/')
            if len(parts) >= 4:
                filename = unquote('/'.join(parts[3:]))
                filepath = resolve_task_file(parts[2], filename)

                if filepath is not None and filepath.is_file():
                    self.send_response(200)
                    # Determine content type based on extension
                    ext = filepath.suffix.lower()
                    content_types = {
                        '.mp3': 'audio/mpeg',
                        '.mp4': 'video/mp4',
                        '.mkv': 'video/x-matroska',
                        '.webm': 'video/webm',
                        '.txt': 'text/plain; charset=utf-8',
                        '.srt': 'text/plain; charset=utf-8',
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
        if MCP_ENABLED and urlparse(self.path).path in MCP_PATHS:
            handle_mcp_post(self)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
        except ValueError:
            content_length = -1
        if content_length < 0 or content_length > MCP_MAX_BODY_BYTES:
            self.send_json({'error': 'Request body too large or missing Content-Length'}, 413)
            return
        # Read the body before any auth rejection: closing with unread request bytes makes
        # the kernel reset the connection, and the client never sees the 401.
        body = self.rfile.read(content_length)
        if not self.require_web_auth(urlparse(self.path).path):
            return

        # API: Save settings (any of whisper_model, whisper_device, uvr_model, uvr_device)
        if self.path == '/api/settings':
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_json({'error': 'Invalid JSON'}, 400)
                return
            error = save_settings(data) if isinstance(data, dict) else 'Expected a JSON object'
            if error:
                self.send_json({'error': error}, 400)
            else:
                self.send_json(settings_payload())
            return

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
                info = get_video_info(canonical_youtube_url(video_id))
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
                transcribe = data.get('transcribe', False)
                timestamps = data.get('timestamps', False)
                keep = stems_to_keep(_api_flag(data, 'remove_vocals'), _api_flag(data, 'remove_background'))

                # Validate URL
                video_id = extract_video_id(url)
                if not video_id:
                    self.send_json({'error': 'Invalid YouTube URL'}, 400)
                    return

                task_id = start_audio_task(video_id, bitrate, transcribe, timestamps, keep)
                self.send_json({'task_id': task_id, 'status': 'started'})

            except QueueFull as e:
                self.send_json({'error': str(e)}, 503)
            except BadRequest as e:
                self.send_json({'error': str(e)}, 400)
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
                transcribe = data.get('transcribe', False)
                timestamps = data.get('timestamps', False)
                keep = stems_to_keep(_api_flag(data, 'remove_vocals'), _api_flag(data, 'remove_background'))

                # Validate URL
                video_id = extract_video_id(url)
                if not video_id:
                    self.send_json({'error': 'Invalid YouTube URL'}, 400)
                    return

                task_id = start_video_task(video_id, resolution, transcribe, timestamps, keep)
                self.send_json({'task_id': task_id, 'status': 'started'})

            except QueueFull as e:
                self.send_json({'error': str(e)}, 503)
            except BadRequest as e:
                self.send_json({'error': str(e)}, 400)
            except json.JSONDecodeError:
                self.send_json({'error': 'Invalid JSON'}, 400)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
            return

        self.send_error(404, 'Not found')
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        if MCP_ENABLED and urlparse(self.path).path in MCP_PATHS:
            handle_mcp_preflight(self)
            return
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_DELETE(self):
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if MCP_ENABLED and path in MCP_PATHS:
            send_mcp_method_not_allowed(self)
            return

        if not self.require_web_auth(path):
            return

        # API: Delete a task's files
        if path.startswith('/api/delete/'):
            task_id = path.split('/')[-1]
            task_dir = resolve_task_dir(task_id)
            if task_dir is None:
                self.send_json({'success': False, 'error': 'Invalid task id'}, 400)
                return
            cancel_queued_task(task_id)

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
    """Delete downloads that finished more than an hour ago."""
    while True:
        time.sleep(3600)  # Check every hour
        cutoff = time.time() - 3600
        try:
            task_dirs = [d for d in DOWNLOAD_DIR.iterdir() if d.is_dir()]
        except OSError:
            continue
        for task_dir in task_dirs:
            try:
                with downloads_lock:
                    # A long CPU transcription can outlive the window; never pull the files
                    # out from under a job that is still running.
                    if active_downloads.get(task_dir.name, {}).get('status') == 'processing':
                        continue
                if task_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(task_dir)
                    with downloads_lock:
                        active_downloads.pop(task_dir.name, None)
            except OSError:
                pass


def _network_url():
    """The address other devices should use, if this process can tell. None if it cannot."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    if os.path.exists('/.dockerenv'):
        # In a container the only address visible here is Docker's internal one, which other
        # devices cannot reach. Docker publishes the port on the host's addresses instead.
        return None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1', 80))  # TEST-NET-1: selects the outgoing interface, sends nothing
            return f'http://{s.getsockname()[0]}:{EXTERNAL_PORT}'
    except OSError:
        return None


def main():
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_downloads, daemon=True)
    cleanup_thread.start()
    
    # Start server
    server = ThreadedHTTPServer((HOST, PORT), RequestHandler)
    
    network_url = _network_url()
    other_devices = network_url or f"http://<this server's IP>:{EXTERNAL_PORT}"

    print("=" * 50)
    print("ytdl-web - YouTube Downloader")
    print("=" * 50)
    print(f"\nServer running on port {EXTERNAL_PORT}, on all network interfaces (0.0.0.0).")
    print("\nOpen it at (plain http, not https):")
    print(f"   This machine:   http://localhost:{EXTERNAL_PORT}")
    print(f"   Other devices:  {other_devices}")
    if not network_url:
        print("   (Set PUBLIC_BASE_URL to show the exact address here.)")
    print(f"\nJobs: up to {MAX_ACTIVE_JOBS} at once; more wait in a queue (max {MAX_QUEUED_JOBS})")
    if WEB_AUTH_PASSWORD:
        print(f"\nWeb UI auth: on (HTTP Basic, user {WEB_AUTH_USER!r})")
        if len(WEB_AUTH_PASSWORD) < 12:
            print("   [WARN] WEB_AUTH_PASSWORD is short; use at least 12 characters")
    else:
        print("\nWeb UI auth: off (set WEB_AUTH_PASSWORD to require a login)")
    if MCP_ENABLED:
        print(f"\nMCP server (Streamable HTTP, protocol {MCP_MODERN_VERSIONS[0]} + legacy {MCP_LEGACY_VERSIONS[-1]}..{MCP_LEGACY_VERSIONS[0]}):")
        print(f"   Endpoint: {other_devices}/mcp")
        print(f"   Auth:     {MCP_AUTH}")
        if MCP_CONFIG_ERROR:
            print(f"   [ERROR] {MCP_CONFIG_ERROR} - /mcp refuses every request until this is fixed")
        elif MCP_AUTH == 'token' and len(MCP_AUTH_TOKEN) < 32:
            print("   [WARN] MCP_AUTH_TOKEN is short; use at least 32 random characters")
        if MCP_AUTH != 'off' and not WEB_AUTH_PASSWORD:
            print("   [WARN] MCP_AUTH protects /mcp only; the web UI and /api/* are still open.")
            print("          Set WEB_AUTH_PASSWORD to protect them too.")
    print(f"\nPress Ctrl+C to stop the server\n")
    print("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        server.shutdown()


if __name__ == '__main__':
    main()
