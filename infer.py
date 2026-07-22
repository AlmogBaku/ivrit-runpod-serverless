import base64
import os
import tempfile
import urllib.request
from pathlib import Path

import runpod
import torch
from faster_whisper import WhisperModel

MODEL_DEFAULT = "ivrit-ai/whisper-large-v3-ct2"
HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

_transcription_model = None
_transcription_model_name = None


def _resolve_cached_model(model_id):
    """Resolve a RunPod-managed Hugging Face snapshot to a local path."""
    candidate = Path(model_id)
    if candidate.is_dir():
        return str(candidate)
    if "/" not in model_id:
        raise ValueError(f"model_id must be an org/name identifier: {model_id}")

    org, name = model_id.split("/", 1)
    model_root = HF_CACHE_ROOT / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        snapshot_hash = refs_main.read_text().strip()
        snapshot = snapshots_dir / snapshot_hash
        if snapshot.is_dir():
            return str(snapshot)

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir()) if snapshots_dir.is_dir() else []
    if snapshots:
        return str(snapshots[0])

    raise RuntimeError(
        f"RunPod model cache missing: {model_id}. "
        "Set the endpoint Model field to this Hugging Face ID and redeploy."
    )


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
        model_path = _resolve_cached_model(model_name)
        print(f"Loading cached faster-whisper model: {model_name} ({model_path})", flush=True)
        _transcription_model = WhisperModel(model_path, local_files_only=True)
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
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text}
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
