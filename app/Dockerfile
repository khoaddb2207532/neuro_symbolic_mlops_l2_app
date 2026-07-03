# Container riêng cho SERVING — nhẹ hơn nhiều so với Dockerfile huấn luyện
# (không cần torchgfn, dvc, sklearn training deps nặng).
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY app/requirements-app.txt app/requirements-app.txt
RUN pip install --no-cache-dir -r requirements.txt -r app/requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY params.yaml params.yaml
# Copy checkpoint + luật đã huấn luyện — build image SAU KHI đã có outputs/
# (hoặc mount outputs/ như volume khi chạy container, xem README-serving.md)
COPY outputs/05_rules_model/final_model_weights.pth outputs/05_rules_model/final_model_weights.pth
COPY outputs/04_filtered_rules/selected_rules_improved.pkl outputs/04_filtered_rules/selected_rules_improved.pkl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
