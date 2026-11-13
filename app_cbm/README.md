# CBM inference API

Ứng dụng này phục vụ ba loại model được huấn luyện bởi notebook trong
`generated_cbm_notebooks`:

- Baseline: `01_baseline/baseline_best.pth`
- CBM: `cbm/cbm_best.pth`
- CBM regularized: `05_rules_model/rule_regularized_best.pth`

Artifacts dual-GPU được notebook tạo trong `outputs_seed_<seed>/`. Ứng dụng
chỉ tự quét các thư mục `outputs_seed_*` nằm trực tiếp trong `app_cbm/`;
không quét `outputs/` cũ, thư mục con lồng sâu hoặc đường dẫn bên ngoài. Hãy
giải nén `dual_gpu_cbm_outputs.zip` của notebook vào `app_cbm/` để tạo các thư
mục này. Mỗi run cần có `params_seed_<seed>.yaml`; metadata concept được đọc
từ `cbm/cbm_metadata.json`.

```powershell
pip install -r requirements.txt -r app_cbm/requirements-app.txt
uvicorn app_cbm.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở `http://localhost:8000` để chọn run/model và tải ảnh lên. Chọn nhiều model
trong cùng run để so sánh cả ba loại. Có hai chế độ:

- **Bình thường**: phân loại nhanh, hiển thị nhãn, độ tin cậy và concept kích
  hoạt; không chạy occlusion mapping.
- **Vừa chỉnh sửa**: chạy thêm occlusion mapping trên lưới `7×7`, rồi hiển thị
  luồng ảnh đầu vào → tối đa 5 crop box → ảnh kết quả có box, nhãn và độ tin
  cậy. Tọa độ box theo ảnh gốc; đây là vùng đóng góp ước lượng, không phải
  localization ground truth. Chế độ này chậm hơn do cần các lượt che ảnh.

- `GET /models`: quét lại artifacts và liệt kê mô hình tải được.
- `GET /health`: trạng thái service và số mô hình khả dụng.
- `POST /predict`: multipart fields `file`, `run_id` (tuỳ chọn), `model_ids`
  (tuỳ chọn, danh sách ID phân tách bằng dấu phẩy), `mode` (`normal` hoặc
  `edited`, mặc định `normal`).
- `GET /docs`: OpenAPI/Swagger UI.
