# Include Python
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Define your working directory
WORKDIR /

# Configure LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# Install relevant packages 
RUN apt update
RUN apt install -y ffmpeg

# Install only the runtime dependencies needed for non-diarized ivrit STT.
RUN pip3 install ivrit==0.2.6 torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 huggingface-hub==0.36.0 runpod

# Preload only the model used by this endpoint. Runtime uses local_files_only=True.
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-ct2")'

# Add your file
ADD infer.py .

# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]

