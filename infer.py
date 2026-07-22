import base64
import os
import tempfile
import urllib.request
from pathlib import Path

import runpod
import torch
from faster_whisper import WhisperModel

MODEL_DEFAULT = "ivrit-ai/whisper-large-v3-turbo-ct2"

_transcription_model = None
_transcription_model_name = None


def _audio_path(transcribe_args):
    """Materialize blob, URL, or local path for faster-whisper."""
    blob = transcribe_args.get("blob")
    url = transcribe_args.get("url")
    path = transcribe_args.get("path")
    sources = [value for value in (blob, url, path) if value]
    if len(sources) != 1:
        raise ValueError("transcribe_args must contain exactly one of blob, url, or path")

    if path:
        audio_path = Path(path).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(str(audio_path))
        return str(audio_path), None

    if blob:
        data = base64.b64decode(blob, validate=True)
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "runpod-stt/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()

    if not data:
        raise ValueError("audio input is empty")
    handle = tempfile.NamedTemporaryFile(prefix="runpod-audio-", suffix=".audio", delete=False)
    handle.write(data)
    handle.close()
    return handle.name, handle.name


def _load_model(model_name):
    global _transcription_model, _transcription_model_name
    if _transcription_model is None or _transcription_model_name != model_name:
        print(f"Loading faster-whisper model: {model_name}", flush=True)
        _transcription_model = WhisperModel(model_name, local_files_only=True)
        _transcription_model_name = model_name
    return _transcription_model


def _transcribe(model_name, transcribe_args):
    audio_path, cleanup_path = _audio_path(transcribe_args)
    try:
        model = _load_model(model_name)
        language = transcribe_args.get("language") or None
        multilingual = bool(transcribe_args.get("multilingual", True))
        threshold = float(transcribe_args.get("language_detection_threshold", 0.7))
        detection_segments = int(transcribe_args.get("language_detection_segments", 2))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("language_detection_threshold must be between 0 and 1")
        if detection_segments < 1:
            raise ValueError("language_detection_segments must be at least 1")

        segments, _info = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            multilingual=multilingual,
            language_detection_threshold=threshold,
            language_detection_segments=detection_segments,
            vad_filter=True,
            word_timestamps=False,
        )
        yield [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text,
            }
            for segment in segments
        ]
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except FileNotFoundError:
                pass


def handler(job):
    payload = job.get("input", {})
    model_name = payload.get("model") or MODEL_DEFAULT
    transcribe_args = payload.get("transcribe_args")
    if not isinstance(transcribe_args, dict):
        yield {"error": "transcribe_args field not provided."}
        return

    try:
        yield {"result": list(_transcribe(model_name, transcribe_args))}
    except Exception as exc:
        print(f"transcription failed: {exc!r}", flush=True)
        yield {"error": str(exc)}


if not torch.cuda.is_available():
    raise RuntimeError("GPU health check failed: CUDA not available")

runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
