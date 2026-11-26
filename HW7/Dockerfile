FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        "transformers==4.55.2" \
        "datasets==4.1.1" \
        "accelerate==1.10.1" \
        "llmcompressor==0.8.1" \
        "compressed-tensors==0.12.2" \
        "numpy==2.2.6" \
        "pandas==2.2.3" \
        "safetensors>=0.5" \
        "huggingface-hub<1" \
        tqdm \
        ipykernel

CMD ["bash"]
