# Include Python
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

RUN apt update && apt install -y ffmpeg

# Direct faster-whisper for transcription and language detection.
RUN pip3 install \
    faster-whisper==1.2.1 \
    ctranslate2==4.8.1 \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    torchvision==0.22.1 \
    huggingface-hub==1.24.0 \
    runpod==1.11.0

# Preload both ivrit checkpoints; the full model is active, turbo remains available for fallback/A-B tests.
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-ct2")'
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2")'
ENV HF_HUB_OFFLINE=1

ADD infer.py .

CMD ["python", "-u", "/infer.py"]
