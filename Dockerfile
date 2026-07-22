# Include Python
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

RUN apt update && apt install -y ffmpeg

# Runtime dependencies only. Model weights are supplied through RunPod's
# managed Hugging Face cache, not downloaded during the Docker build.
RUN pip3 install \
    faster-whisper==1.2.1 \
    ctranslate2==4.8.1 \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    torchvision==0.22.1 \
    huggingface-hub==1.24.0 \
    runpod==1.11.0

# The worker resolves /runpod-volume/huggingface-cache/hub at startup.
ENV HF_HUB_OFFLINE=1

ADD infer.py .

CMD ["python", "-u", "/infer.py"]
