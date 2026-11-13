# Neuro-Symbolic Vietnamese Cultural Classifier — CBM

Nhánh `cbm` tập trung vào pipeline **Baseline + Concept Bottleneck Model
(CBM) + trích xuất luật + GFlowNet + rule regularization**. Notebook Kaggle
dual-GPU dùng template
[cbm-neuro-symbolic-kaggle-template.ipynb](./cbm-neuro-symbolic-kaggle-template.ipynb).
API inference CBM nằm trong [app_cbm/](./app_cbm/).

## Cài đặt

Yêu cầu Python 3.10 trở lên. Cài dependencies trong môi trường phù hợp với
phiên bản PyTorch/CUDA của máy:

```powershell
python -m pip install -r requirements.txt
```

Trước khi chạy, cập nhật `data_dir` và `cbm.concept_annotations_path` trong
[params.yaml](./params.yaml). Dataset cần có các thư mục `train`, `val`,
`test` và `val2`, mỗi thư mục chứa thư mục con theo lớp.

## Huấn luyện

Chạy một lượt pipeline trên thiết bị khả dụng:

```powershell
python -m pipelines.run_cbm_neuro_symbolic_pipeline --config params.yaml
```

Chạy song song hai seed trên hai GPU (ví dụ Kaggle GPU T4 x2):

```powershell
python -m pipelines.run_dual_gpu_cbm_experiment --config params.yaml --seeds 42 44
```

Mỗi seed ghi kết quả vào `outputs_seed_<seed>/`. Có thể bỏ qua baseline đã
được huấn luyện bằng `--skip-baseline`. Cấu hình và đường dẫn đầu ra có thể
điều chỉnh trong `params.yaml`; kết quả training history được lưu cùng các
checkpoint và báo cáo trong thư mục output.

## Chạy notebook trên Kaggle

Mở notebook template ở trên, bật GPU (khuyến nghị GPU T4 x2), cập nhật
`GIT_BRANCH`, dataset và file concept annotations nếu cần, rồi chạy các cell
theo thứ tự. Notebook cài dependencies cần thiết và chạy cùng pipeline CBM
trên nhánh này.

## Chạy API inference

Sau khi có artifacts, cài dependencies của ứng dụng và khởi động API:

```powershell
python -m pip install -r requirements.txt -r app_cbm/requirements-app.txt
uvicorn app_cbm.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở `http://localhost:8000` để dùng giao diện hoặc `/docs` để xem API. Hướng
dẫn chuẩn bị artifacts chi tiết nằm trong [app_cbm/README.md](./app_cbm/README.md).

## Kiểm thử

```powershell
pytest tests/ -v
```
