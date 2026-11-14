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
Các luật giải thích được đọc từ
`04_filtered_rules/selected_rules.pkl` nếu artifact này có trong run. Ứng dụng
đánh giá các điều kiện luật trên concept probabilities của ảnh hiện tại và
hiển thị các luật khớp với lớp dự đoán cho CBM/CBM regularized; Baseline không
có concept để áp dụng các luật này. Độ tin cậy luật là confidence lưu trong
artifact, không phải độ tin cậy dự đoán của ảnh.

```powershell
pip install -r requirements.txt -r app_cbm/requirements-app.txt
uvicorn app_cbm.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở `http://localhost:8000` để chọn run/model và tải ảnh lên. Chọn nhiều model
trong cùng run để so sánh cả ba loại. Có hai chế độ:

- **Bình thường**: phân loại nhanh, hiển thị nhãn, độ tin cậy, tối đa 5 concept
  có xác suất cao nhất và các luật khớp; không chạy occlusion mapping.
- **Vừa chỉnh sửa**: chạy thêm occlusion mapping trên lưới `7×7`, rồi hiển thị
  tối đa 5 concept có xác suất cao nhất, luồng ảnh đầu vào → tối đa 5 crop box
  → ảnh kết quả có box, nhãn, độ tin cậy và luật khớp. Tọa độ box theo ảnh
  gốc; đây là vùng đóng góp ước lượng, không phải localization ground truth.
  Chế độ này chậm hơn do cần các lượt che ảnh.

- `GET /models`: quét lại artifacts và liệt kê mô hình tải được.
- `GET /health`: trạng thái service và số mô hình khả dụng.
- `POST /predict`: multipart fields `file`, `run_id` (tuỳ chọn), `model_ids`
  (tuỳ chọn, danh sách ID phân tách bằng dấu phẩy), `mode` (`normal` hoặc
  `edited`, mặc định `normal`).
- `GET /docs`: OpenAPI/Swagger UI.

## Giải thích CBM theo đóng góp concept

Với mô hình CBM, `label_predictor` là một lớp Linear nên logit của lớp dự đoán
`c` bằng `Σ_i W[c, i] · p_i + b_c`. Mỗi concept `i` có **đóng góp**
`p_i · W[c, i]` vào logit đó; `label_bias` là `b_c` và `predicted_logit` là
logit của lớp dự đoán. Các đóng góp cộng với bias đúng bằng `predicted_logit`.

Trong response `POST /predict`, mỗi prediction CBM có:

- `concept_contributions`: danh sách đóng góp của **tất cả** concept, xếp từ
  đẩy logit lên nhiều nhất đến kéo xuống nhiều nhất. Mỗi mục gồm `probability`,
  `threshold`, `activated`, `weight` (`W[c, i]`) và `contribution`.
- `label_bias`, `predicted_logit`: để kiểm tra `Σ contribution + label_bias ≈ predicted_logit`.
- `concept_threshold_source`: `per_concept` nếu dùng ngưỡng riêng, `global` nếu dùng
  ngưỡng mặc định.

Lưu ý khi đọc: đây là phân rã của **logit**, không phải của xác suất softmax.
Concept có đóng góp dương lớn chỉ cho biết nó đẩy logit của lớp đó lên; không đảm
bảo lớp đó thắng các lớp khác. Muốn biết vì sao lớp A thắng lớp B, so sánh
`W[A, i] − W[B, i]` thay vì chỉ nhìn một hàng.

### Ngưỡng riêng cho từng concept

Ngưỡng kích hoạt mặc định là `0.5` cho mọi concept. Nếu `cbm_metadata.json` có khoá
`concept_thresholds` (danh sách đúng `n_concepts` số trong khoảng `(0, 1)`, theo
đúng thứ tự `concept_names`), mỗi concept dùng ngưỡng của chính nó. Ngưỡng này
áp dụng cho cả cờ `activated`, việc chọn vùng occlusion và trường `threshold` của
từng concept. Nếu khoá có mặt nhưng sai định dạng, mô hình đó không được tải và
lỗi được ghi trong `/models`, thay vì lặng lẽ dùng ngưỡng sai.
`concept_activation_threshold` vẫn trả về ngưỡng toàn cục để tương thích ngược.
