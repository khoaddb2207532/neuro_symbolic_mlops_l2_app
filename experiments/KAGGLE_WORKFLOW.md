# Quy trình Kaggle: 2 dataset × 5 seed × 6 phương pháp

Mỗi seed là **một run notebook** và bên trong run đó thực hiện đủ sáu phương pháp:
`baseline`, `gflownet_elite`, `random`, `topk_confidence`, `greedy_coverage`, `bayesian`.
Vì vậy ma trận này cần 10 run huấn luyện, không phải 60 notebook.

## 1. Chuẩn bị duy nhất một lần

1. Mở `experiments/experiment_registry.csv`.
2. Dataset B dùng `/kaggle/working`: notebook được sinh tự tạo symlink `train`, `test`, `val` từ Fast Food Classification V2 ngay sau khi clone repository.
3. Kiểm tra phân công tài khoản:
   - `account-1`: dataset A, seed 42/44/46.
   - `account-2`: dataset A, seed 48/50.
   - `account-3`: dataset B, seed 42/44/46.
   - `account-4`: dataset B, seed 48/50.
4. Commit và push toàn bộ mã nguồn. Ghi lại SHA cố định bằng `git rev-parse HEAD`.
5. Sinh 10 notebook từ cùng một template:

   ```bash
   python scripts/generate_kaggle_notebooks.py --git-commit <SHA>
   ```

Notebook được tạo trong `generated_kaggle_notebooks/`. Không sửa logic riêng từng bản; tham số phải đến từ registry.
Riêng năm notebook `culture-b` có thêm hai cell tự động: tắt warning/log rác và tạo ba symlink dataset. Notebook `culture-a` không có hai cell này.

Sinh luôn bốn notebook đóng gói theo tài khoản:

   ```bash
   python scripts/generate_kaggle_bundle_notebooks.py --git-commit <SHA>
   ```

## 2. Chạy 10 notebook

Trên mỗi tài khoản, upload đúng các notebook được phân công, bật GPU và Add Input dataset tương ứng.
Mỗi run phải dùng đúng `RUN_ID`, `DATASET_ID`, `SEED`, `BACKBONE`, `ACCOUNT_ID` và `GIT_COMMIT` đã sinh.

Runner tạo:

- `run_manifest.json`: danh tính run, dataset fingerprint, commit, trạng thái;
- `results.csv`: đúng sáu phương pháp;
- fairness matched-budget, exact metrics, rule quality, ranking, checkpoint posterior và runtime;
- `/kaggle/working/experiment_exports/<RUN_ID>/`: export nhẹ để tổng hợp.

Chỉ coi run là hợp lệ khi manifest có `status=complete`. Sau khi chạy, dùng **Save Version** để output trở thành input cho bước sau.

### Resume khi Kaggle ngắt

Add Input output của **chính run có cùng RUN_ID**, sau đó chạy lại notebook. Runner tự dò artefact hoàn tất và tiếp tục từ stage thiếu. Không dùng output của dataset/seed khác. Manifest và fingerprint được dùng để phát hiện nhầm lẫn ở bước đóng gói/tổng hợp.

## 3. Đóng gói theo bốn tài khoản

Mỗi tài khoản dùng notebook `bundle_account-<n>.ipynb` vừa sinh:

1. Add Input output của 2 hoặc 3 run thuộc tài khoản đó.
2. Xác nhận `ACCOUNT_ID`, `EXPECTED_RUN_IDS` và `GIT_COMMIT` đã được điền tự động.
3. Run All, rồi Save Version.

Notebook chỉ thành công nếu tìm thấy chính xác danh sách run mong đợi. Output gồm thư mục `bundle_<ACCOUNT_ID>`, index CSV, bundle manifest và file `.tar.gz`.

## 4. Tổng hợp chính thức

Tạo notebook từ `managed-multi-dataset-aggregate.ipynb`:

1. Add Input cả bốn bundle đã Save Version.
2. Đặt đúng `GIT_COMMIT` dùng để chạy thí nghiệm.
3. Run All.

Notebook sẽ dừng nếu không đủ:

- 2 dataset;
- 5 seed riêng biệt cho mỗi dataset;
- 10 run;
- đúng 6 phương pháp mỗi run;
- tổng cộng 60 hàng kết quả.

Các file chính:

- `multi_dataset_run_audit.csv`: kiểm toán danh tính 10 run;
- `multi_dataset_all_results.csv`: 60 kết quả đầy đủ;
- `dataset_method_summary.csv`: mean và sample standard deviation theo từng dataset;
- `dataset_paired_deltas.csv`: paired delta so với baseline và Bayesian so với GFlowNet elite;
- `multi_dataset_aggregate_manifest.json`: xác nhận tổng hợp hoàn tất.

Không lấy trung bình chung hai dataset. Báo cáo từng dataset riêng, sau đó mô tả tính nhất quán giữa hai dataset.

## 5. Quy tắc đặt tên và theo dõi

Run ID có dạng:

```text
<dataset-id>__<backbone>__db__seed_<seed>
```

Không dùng tên như `final`, `new`, `copy`. Cập nhật các cột `status`, `notebook_slug`, `kaggle_version`, `git_commit`, `output_reference` trong registry sau mỗi Save Version. Các trạng thái nên dùng: `pending`, `running`, `complete`, `failed`, `bundled`.

Chỉ phân phối công việc giữa các tài khoản mà bạn được phép sử dụng và tuân thủ điều khoản/hạn mức của Kaggle; không dùng nhiều tài khoản để né giới hạn nền tảng.
