# Include Python
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Define your working directory
WORKDIR /

# Configure LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# Install relevant packages
RUN apt update
RUN apt install -y ffmpeg

# Install Python packages. Keep the PyTorch pins matched to the CUDA base image;
# upgrade the transcription/runtime packages independently.
RUN pip3 install ivrit[all]==0.2.6 \
    faster-whisper==1.2.1 \
    ctranslate2==4.8.1 \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    torchvision==0.22.1 \
    huggingface-hub==1.24.0 \
    runpod==1.11.0

# ivrit 0.2.6 drops newer faster-whisper kwargs. Forward the language-routing
# controls before the worker starts.
COPY patch_ivrit_faster_whisper.py /tmp/patch_ivrit_faster_whisper.py
RUN python3 /tmp/patch_ivrit_faster_whisper.py && rm /tmp/patch_ivrit_faster_whisper.py

# Preload the upstream model set.
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2")'
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/yi-whisper-large-v3-turbo-ct2")'
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("large-v3-turbo")'
RUN python3 -c 'import pyannote.audio; import torch; from pyannote.audio.core.task import Problem, Resolution, Specifications; torch.serialization.add_safe_globals([Problem, Resolution, Specifications, torch.torch_version.TorchVersion]); p = pyannote.audio.Pipeline.from_pretrained("ivrit-ai/pyannote-speaker-diarization-3.1")'
RUN python3 -c 'from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")'

# Add your file
ADD infer.py .

# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
