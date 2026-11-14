FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Chạy pipeline Baseline + CBM + GFlowNet + rule regularization.
CMD ["python", "-m", "pipelines.run_cbm_neuro_symbolic_pipeline", "--config", "params.yaml"]
