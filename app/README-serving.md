# Triển khai Inference Service

## 1. Chạy local (không Docker) — để test nhanh
```bash
pip install -r requirements.txt -r app/requirements-app.txt
# Cần đã có checkpoint: outputs/05_rules_model/final_model_weights.pth
#                    và: outputs/04_filtered_rules/selected_rules_improved.pkl
# (tức đã chạy `dvc repro` xong ít nhất 1 lần)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Test:
```bash
curl -X POST http://localhost:8000/predict -F "file=@sample.jpg"
curl http://localhost:8000/health
```
Docs tương tác (Swagger UI): http://localhost:8000/docs

## 2. Demo UI (Gradio)
```bash
python -m app.demo_gradio
# mở http://localhost:7860
```

## 3. Đóng gói Docker (production)
```bash
# Build SAU KHI đã có outputs/ (checkpoint + rules) từ `dvc repro`
docker build -f app/Dockerfile -t vn-cultural-classifier:latest .
docker run -p 8000:8000 vn-cultural-classifier:latest
```

Nếu không muốn bake checkpoint vào image (image sẽ nặng, phải build lại mỗi
khi retrain), mount `outputs/` như volume thay vì `COPY` trong Dockerfile:
```bash
docker run -p 8000:8000 -v $(pwd)/outputs:/workspace/outputs vn-cultural-classifier:latest
```
→ sửa `app/Dockerfile`, xoá 2 dòng `COPY outputs/...`, để checkpoint luôn được
nạp từ volume ngoài — cho phép cập nhật model mới mà không cần rebuild image
(gần với "model registry" pattern của MLOps thật).

## 4. Gợi ý nơi host (miễn phí / chi phí thấp)
| Nơi host | Phù hợp khi |
|---|---|
| **Hugging Face Spaces** (Docker SDK) | Muốn public demo Gradio nhanh, miễn phí CPU, có GPU trả phí nếu cần. Chỉ cần push repo có `app/Dockerfile` + `app.py` (đổi tên `demo_gradio.py`). |
| **Render / Railway / Fly.io** | Muốn FastAPI service ổn định, có custom domain, tự động deploy khi push GitHub. |
| **Self-host qua GitHub Actions + Docker registry** | Đã có sẵn `.github/workflows/ci.yml` — thêm 1 job build & push image tới GitHub Container Registry (`ghcr.io`) sau khi test pass, rồi `docker pull` trên server riêng. |

## 5. Kết nối vào CI/CD hiện có
Thêm job sau vào `.github/workflows/ci.yml` (sau job `lint-and-test`) để hoàn
thiện vòng CI/CD → CD (Continuous Delivery) cho MLOps L2:
```yaml
  build-and-push-image:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          file: app/Dockerfile
          push: true
          tags: ghcr.io/<user>/<repo>:latest
```
(Cần checkpoint model đã có sẵn trong repo hoặc tải về từ DVC remote trong bước build — ví dụ thêm `dvc pull` trước bước build image.)
