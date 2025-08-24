FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv \
                       build-essential cmake git vim && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
