# Notebook GFlowNet-TB nhiều seed trên 2 GPU

Bộ notebook trong `generated_dual_gpu_tb_multiseed_notebooks/` chạy ba phương
pháp cho mỗi seed:

1. baseline từ prior Stage 1–3;
2. `stage4_select_rules_gflownet` với `loss_type: tb`, sau đó
   `stage5_train_rule_regularized`;
3. cùng GFlowNet-TB frozen sampler, sau đó
   `stage5_train_rule_bayesian_tb`.

Hai tiến trình GPU dùng hàng đợi cố định theo round-robin. Với seed mặc định
`[42, 44, 46, 48, 50]`, GPU 0 nhận `[42, 46, 50]` và GPU 1 nhận `[44, 48]`.
Mỗi GPU chỉ chạy một seed tại một thời điểm nên không có hai tiến trình tranh
bộ nhớ trên cùng thiết bị.

## Sinh lại notebook

```powershell
python scripts/generate_dual_gpu_tb_multiseed_notebooks.py `
  --git-ref <branch-hoặc-40-char-commit> `
  --seeds 42 44 46 48 50 `
  --no-resume-seeds 46 50
```

Mặc định script sinh tám notebook: MobileNetV3 Small, ShuffleNetV2 x1.0
(`shufflenet_v2_x1_0`), AlexNet, ResNet-50, DenseNet-121, EfficientNet-B0,
ViT-B/32 (`vit_b_32`) và Swin-T (`swin_t`).
Có thể giới hạn bằng `--backbones` và `--datasets`.

## Chạy đúng 2 seed trên 2 GPU

Để tránh chạy quá nhiều seed trong cùng một phiên, dùng generator chuyên biệt:

```powershell
python scripts/generate_dual_gpu_tb_two_seed_notebooks.py `
  --git-ref <branch-hoặc-40-char-commit> `
  --seeds 46 50 `
  --backbones vit_b_32
```

Thứ tự seed cũng là thứ tự GPU: ví dụ trên gán seed 46 cho GPU 0 và seed 50
cho GPU 1. Nếu có nhiều dataset, mỗi GPU xử lý tuần tự các dataset của seed đó;
tại mọi thời điểm chỉ có tối đa hai worker huấn luyện, mỗi GPU một worker. Notebook
mẫu được ghi vào `generated_dual_gpu_tb_two_seed_notebooks/`. Tên file chứa cả
hai seed để các cặp thí nghiệm không ghi đè nhau.
Hai seed chạy mới hoàn toàn theo mặc định: không restore output cũ và Stage 5 đặt
`resume: false`. Thêm `--resume` nếu chủ động muốn tiếp tục checkpoint của lần dở.

Notebook cần Kaggle Accelerator `GPU T4 x2` và Add Input chứa prior output đúng
`dataset/backbone/seed`. `prior_run_id` có hậu tố `db` vì đó là tên run lõi
Stage 1–3 hiện có; Stage 4 và hai Stage 5 mới luôn được ép dùng TB.

## Bảng so sánh

Mỗi notebook tạo:

- `tb_all_seed_comparison.csv`: từng dataset/model/seed/method;
- `tb_all_seed_summary.csv`: mean và sample standard deviation qua seed;
- `tb_all_seed_summary.md`: bảng đọc nhanh Accuracy và Macro-F1 dạng mean±std.

Nếu phiên Kaggle bị ngắt, Save Version và Add Input output đó vào lần chạy sau.
Runner khôi phục run có identity khớp và bỏ qua checkpoint Stage 4/5 đã hoàn tất.
Mặc định seed 46 và 50 chạy mới: runner không copy run cũ từ Kaggle Input và
đặt `resume: false` cho hai seed này. Các seed 42, 44 và 48 vẫn resume bình thường.
