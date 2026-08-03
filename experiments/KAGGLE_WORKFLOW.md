# Quy trình Kaggle bundle: 2 dataset × 3 seed × 6 phương pháp

Mỗi seed là **một run notebook** và bên trong run đó thực hiện đủ sáu phương pháp:
`baseline`, `gflownet_elite`, `random`, `topk_confidence`, `greedy_coverage`, `bayesian`.
Bundle của mỗi mô hình chỉ lấy 6 run thuộc ba seed `46`, `48`, `50`; các seed
khác trong registry không được đưa vào bundle.

## 1. Chuẩn bị duy nhất một lần

1. Mở `experiments/experiment_registry.csv`.
2. Dataset B dùng `/kaggle/working`: notebook được sinh tự tạo symlink `train`, `test`, `val` từ Fast Food Classification V2 ngay sau khi clone repository.
3. Kiểm tra phân công tài khoản:
   - `account-1`: dataset A, seed 42/44/46.
   - `account-2`: dataset A, seed 48/50.
   - `account-3`: dataset B, seed 42/44/46.
   - `account-4`: dataset B, seed 48/50.
4. Commit và push toàn bộ mã nguồn. Ghi lại SHA cố định bằng `git rev-parse HEAD`.
5. Sinh các run notebook trong registry từ cùng một template:

   ```bash
   python scripts/generate_kaggle_notebooks.py --git-commit <SHA>
   ```

Notebook được tạo trong `generated_kaggle_notebooks/<backbone>/`, ví dụ `generated_kaggle_notebooks/resnet50/`. Không sửa logic riêng từng bản; tham số phải đến từ registry. Nếu registry có nhiều model, script tự tách mỗi model thành một thư mục để không ghi đè hoặc trộn notebook.
Các notebook `culture-b` có thêm hai cell tự động: tắt warning/log rác và tạo ba symlink dataset. Notebook `culture-a` không có hai cell này.

Sinh một notebook đóng gói cho mỗi mô hình. Mặc định bundle chỉ lấy ba seed
`46`, `48`, `50` (ở cả hai dataset):

   ```bash
   python scripts/generate_kaggle_bundle_notebooks.py --git-commit <SHA>
   ```

Nếu registry hiện chỉ chứa một backbone nhưng cần tái sử dụng cùng ma trận cho nhiều
mô hình, truyền danh sách tường minh, ví dụ
`--backbones mobilenetv3_small alexnet resnet50 swin_t vit_b_32`.

## 2. Chạy các run notebook

Trên mỗi tài khoản, upload đúng các notebook được phân công, bật GPU và Add Input dataset tương ứng.
Mỗi run phải dùng đúng `RUN_ID`, `DATASET_ID`, `SEED`, `BACKBONE`, `ACCOUNT_ID` và `GIT_COMMIT` đã sinh.

Runner tạo:

- `run_manifest.json`: danh tính run, dataset fingerprint, commit, trạng thái;
- `results.csv`: đúng sáu phương pháp;
- fairness matched-budget, exact metrics, rule quality, ranking, checkpoint posterior và runtime;
- `/kaggle/working/experiment_exports/<RUN_ID>/`: export nhẹ để tổng hợp.

Chỉ coi run là hợp lệ khi manifest có `status=complete`. Sau khi chạy, dùng **Save Version** để output trở thành input cho bước sau.

### Resume khi Kaggle ngắt

Add Input output của **chính run có cùng RUN_ID**, sau đó chạy lại notebook. Runner ưu tiên tìm `run_manifest.json` khớp đồng thời `run_id + dataset_id + backbone + seed`, kể cả manifest có trạng thái `running` hoặc `failed`, rồi khôi phục cây stage trước khi chạy tiếp. Nếu không có manifest mới, runner mới thử output/archive notebook cũ có metadata phù hợp.

Các completion marker được kiểm tra riêng cho từng stage, nên stage đã đủ artefact sẽ được `SKIP`, còn stage lỗi hoặc thiếu output sẽ chạy lại. Không dùng output của dataset/seed/model khác. Với dataset B, fingerprint chỉ tính nội dung `train/val/test`; thay đổi repository hoặc file khác trong `/kaggle/working` không làm sai fingerprint.

## 3. Đóng gói theo mô hình

Mỗi mô hình dùng notebook `bundle_<backbone>.ipynb` vừa sinh:

1. Add Input output của các run thuộc mô hình đó với seed 46, 48 và 50.
2. Xác nhận `BACKBONE`, `SEEDS`, `EXPECTED_RUN_IDS` và `GIT_COMMIT` đã được điền tự động.
3. Run All, rồi Save Version.

Notebook chỉ thành công nếu tìm thấy chính xác danh sách run mong đợi và mọi manifest
khớp backbone/seed. Output gồm thư mục `bundle_<BACKBONE>`, index CSV, bundle
manifest và file `.tar.gz`.

Ngay sau khi bundle hoàn tất, notebook hiển thị và lưu hai bảng:

- `model_all_results.csv`: kết quả chi tiết của từng seed;
- `model_three_seed_summary.csv`: mean, sample standard deviation và `n_seeds`
  theo từng dataset/phương pháp.
- `official_experiment_comparison.csv`: bảng chính thức từng seed kèm thống kê nhóm;
- `paired_delta_vs_cnn.csv` và `bayesian_vs_core_methods.csv`: paired delta;
- `matched_budget_audit.csv`: hậu kiểm fairness;
- `rule_set_quality_mean_std.csv` và `rule_ranking_metrics_mean_std.csv`:
  tổng hợp chất lượng/ranking của tập luật.

## 4. Tổng hợp chính thức

Tạo notebook từ `managed-multi-dataset-aggregate.ipynb`:

1. Add Input model bundle đã Save Version.
2. Đặt đúng `GIT_COMMIT` dùng để chạy thí nghiệm.
3. Run All.

Notebook sẽ dừng nếu không đủ:

- 2 dataset;
- 3 seed (46, 48, 50) riêng biệt cho mỗi dataset;
- 6 run;
- đúng 6 phương pháp mỗi run;
- tổng cộng 36 hàng kết quả.

Các file chính:

- `multi_dataset_run_audit.csv`: kiểm toán danh tính 6 run;
- `multi_dataset_all_results.csv`: 36 kết quả đầy đủ;
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
