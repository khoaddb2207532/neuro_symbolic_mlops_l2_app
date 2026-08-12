# Kaggle dual-GPU Bayesian Elite theo seed

## 1. Commit và push phiên bản pipeline

Notebook pin commit tuyệt đối, vì vậy commit phải chứa ít nhất:

- `gflownet_best_elite.pth` trong Stage 4;
- `stage5_train_rule_bayesian_elite`;
- resume checkpoint trong `trainer.py`;
- `run_dual_gpu_elite_seed_experiment.py`.

Sau khi commit/push, lấy full SHA:

```powershell
git rev-parse HEAD
```

## 2. Sinh notebook riêng cho từng seed

```powershell
python scripts/generate_dual_gpu_elite_notebooks.py `
  --git-commit <FULL_40_CHARACTER_SHA>
```

Notebook được ghi vào:

```text
generated_dual_gpu_elite_notebooks/<dataset_id>/<backbone>/dual_elite_seed_<seed>.ipynb
```

Để sinh 50 notebook cho 2 dataset × 5 backbone × 5 seed:

```powershell
python scripts/generate_dual_gpu_elite_notebooks.py `
  --git-commit <FULL_40_CHARACTER_SHA> `
  --datasets culture-a culture-b `
  --backbones mobilenetv3_small alexnet resnet50 densenet121 efficientnet_b0 `
  --seeds 42 44 46 48 50
```

Có thể thay danh sách `--backbones` bằng bất kỳ năm model nào trong:
`mobilenetv3_small`, `alexnet`, `resnet50`, `densenet121`, `efficientnet_b0`,
`swin_t`, `vit_b_16`, `vit_b_32`.

### Khuyến nghị mới: 5 notebook, mỗi model chạy tất cả seed

Thay vì 50 notebook riêng lẻ, sinh đúng một notebook cho mỗi backbone:

```powershell
python scripts/generate_dual_gpu_elite_model_notebooks.py `
  --git-commit <FULL_40_CHARACTER_SHA> `
  --datasets culture-a culture-b `
  --backbones mobilenetv3_small alexnet resnet50 densenet121 efficientnet_b0 `
  --seeds 42 44 46 48 50
```

Kết quả gồm đúng 5 file:

```text
generated_dual_gpu_elite_model_notebooks/
├── dual_elite_all_seeds_mobilenetv3_small.ipynb
├── dual_elite_all_seeds_alexnet.ipynb
├── dual_elite_all_seeds_resnet50.ipynb
├── dual_elite_all_seeds_densenet121.ipynb
└── dual_elite_all_seeds_efficientnet_b0.ipynb
```

Mỗi notebook tuần tự chạy mọi dataset/seed đã khai báo. Trong từng seed, hai
worker TB/DB vẫn chiếm hai GPU song song. Output cũ được tìm theo
`<dataset>__<backbone>__db__seed_<seed>`; DB Bayesian cũ được tái sử dụng nếu
checkpoint đã tồn tại. Module `aggregate_dual_gpu_elite_model` tổng hợp mean/std
qua các seed mà không cần cell import pandas/matplotlib.

Generator đọc `experiments/experiment_registry.csv`. `PRIOR_RUN_ID` là run DB
cùng dataset/backbone/seed đã chứa baseline, feature, raw rules và ba heuristic.

## 3. Cấu hình Kaggle

1. Upload notebook tương ứng seed.
2. Add Input output của `PRIOR_RUN_ID` cùng seed.
3. Chọn Accelerator **GPU T4 x2**.
4. Bật Internet để clone GitHub và cài dependency.
5. Nếu repo private, tạo Kaggle Secret `GITHUB_TOKEN`.
6. Run All.

Hai tiến trình chạy đồng thời:

```text
GPU 0: GFlowNet loss=tb -> fixed-prior TB -> stage5_train_rule_bayesian_elite
GPU 1: GFlowNet loss=db -> fixed-prior DB -> stage5_train_rule_bayesian_elite
```

Fixed-prior dùng `stage5_train_rule_regularized` với `selected_rules.pkl` riêng
của từng objective. Nhánh DB tái sử dụng `05_rules_model` từ core run nếu đã có;
nhánh TB tạo `tb/05_rules_model/` mới và resume theo `training_last.pth`.

Sau khi Bayesian Elite hoàn tất, mỗi GPU chạy thêm Bayesian Stage 5 chuẩn từ
frozen diverse sampler của chính objective đó:

```text
GFlowNet loss=tb -> stage5_train_rule_bayesian_tb
GFlowNet loss=db -> stage5_train_rule_bayesian
```

Hai stage dùng frozen sampler `gflownet_best_diverse.pth`. Wrapper TB xác minh
`gflownet_rule_order.pkl` thực sự có `loss_type=tb`. Kết quả được lưu riêng tại
`{tb,db}/05b_rules_model_bayesian/` nên không ghi đè kết quả Elite.

Mỗi tiến trình chỉ thấy một GPU thông qua `CUDA_VISIBLE_DEVICES`, dùng config và
output riêng dưới `/kaggle/working/dual_elite_seed_<seed>/{tb,db}`.

## 4. Resume

Stage 5 lưu nguyên tử sau mỗi epoch:

```text
tb/05b_rules_model_bayesian_elite/training_last.pth
db/05b_rules_model_bayesian_elite/training_last.pth
```

Nếu Kaggle bị ngắt, Save Version để giữ output, Add Input version đó vào notebook
mới rồi Run All. Runner tìm `dual_elite_manifest.json`, copy trạng thái cũ và:

- bỏ qua Stage 4 nếu đã có `gflownet_best_elite.pth` và rule order;
- resume Stage 5 từ epoch hoàn tất gần nhất;
- từ chối resume nếu seed/backbone/checkpoint/rule-order không khớp.

### Chính sách tiết kiệm dung lượng

Runner không còn tạo `seed_<seed>_dual_elite_artifacts.tar.gz` sau mỗi seed.
Khi bắt đầu/resume, runner xóa các file `*.tmp` do lần ghi checkpoint bị ngắt và
xóa archive đầy đủ cũ của đúng seed nếu còn tồn tại.

`training_last.pth` chỉ được giữ khi run chưa hoàn tất để có thể resume. Sau khi
hai worker chạy xong và bảng so sánh được tạo thành công, runner xóa
`training_last.pth` và `final_model_weights.pth` trong các thư mục Stage 5 cục bộ,
nhưng luôn giữ `rule_regularized_best.pth`. Thư mục symlink tới Kaggle Input không
bị sửa hoặc xóa. Chi tiết dung lượng đã dọn được ghi trong trường `disk_cleanup`
của `dual_elite_manifest.json`.

## 5. Kết quả so sánh

Module của repo xuất bảng so sánh từ:

```text
seed_<seed>_dual_elite_comparison.csv
```

Notebook không có cell cuối import `pandas`, `matplotlib` hoặc `IPython`.
Module `pipelines.run_dual_gpu_elite_seed_experiment` tự in bảng xếp hạng vào
log của cell chạy chính và lưu thêm báo cáo Markdown:

```text
seed_<seed>_dual_elite_comparison.md
```

Các method gồm:

- `cnn_baseline`;
- `random`;
- `topk_confidence`;
- `greedy_coverage`;
- `gflownet_fixed_prior` (tên tương thích cũ của DB; tái sử dụng checkpoint core
  nếu đã tồn tại);
- `gflownet_tb_fixed_prior` (tập luật cố định do GFlowNet-TB chọn);
- `gflownet_tb_bayesian_elite`;
- `gflownet_db_bayesian_elite`;
- `gflownet_tb_bayesian` (Bayesian chuẩn, frozen diverse sampler TB);
- `gflownet_db_bayesian` (Bayesian Stage 5 ban đầu, frozen diverse sampler DB).

CSV có accuracy, macro precision/recall/F1, weighted F1 và delta của accuracy/F1
so với baseline.
