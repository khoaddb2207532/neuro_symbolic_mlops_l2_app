FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ví dụ chạy toàn bộ pipeline bằng DVC (MLOps L2: môi trường đóng gói, tái lập được)
CMD ["dvc", "repro"]
