# Neuro-Symbolic Vietnamese Cultural Classifier — Refactored (MLOps Level 2)

Repo này là bản tổ chức lại của notebook `lv_gfn.ipynb`: CNN (MobileNetV3-Large)
→ trích đặc trưng → Random Forest → trích/lọc luật → GFlowNet chọn luật →
fine-tune CNN với rule-penalty.

## 1. Chiến lược Transfer Learning đã chọn

**Kết luận: "Freeze Backbone + Freeze BatchNorm" → "Differential Learning
Rate" → "Progressive Unfreezing" theo 3 giai đoạn (`head_only` → `last_block`
→ `full`).** Đây là sự kết hợp của mục 3, 6, 7 trong tài liệu `ĐÓNG BĂNG TRONG
CNN.pdf`, không phải chỉ một chiến lược đơn lẻ.

### Vì sao không chọn các chiến lược khác

| Chiến lược | Vì sao không phù hợp (hoàn toàn) ở đây |
|---|---|
| Fixed Feature Extractor / chỉ train head | Dataset ảnh văn hoá VN lệch domain khá xa ImageNet (kiến trúc, trang phục, hoa văn...) → nếu chỉ train head, backbone không bao giờ thích nghi, trần accuracy sẽ thấp. |
| Full Fine-Tuning ngay từ đầu | `NUM_EPOCHS` gốc trong notebook chỉ là 3, dataset có vẻ vừa/nhỏ (chia train/val/test theo thư mục, không thấy augmentation cực mạnh) → full fine-tuning từ epoch 0 dễ overfit và phá huỷ pretrained features nhanh, đặc biệt nguy hiểm vì **đặc trưng CNN còn được dùng làm input cho Random Forest + GFlowNet ở downstream** — nếu đặc trưng "trôi" quá nhanh/không ổn định, các luật trích ra ở bước 3 sẽ không còn khớp với model cuối cùng. |
| Chỉ Differential LR mà không freeze gì | Không giải quyết vấn đề BatchNorm bị lệch running-stats khi batch/dataset nhỏ — quan sát thấy code gốc đã tự ý thức điều này (`freeze_bn=True` trong config gốc), nên đây là tín hiệu đúng cần giữ lại và làm rõ ràng hơn. |

### Vì sao chọn tổ hợp này

1. **Freeze Backbone + Freeze BatchNorm (giai đoạn `head_only`)** — code gốc
   (`CNNBaseline.freeze_bn()`, `config["freeze_bn"]=True`) đã ngầm áp dụng một
   phần chiến lược này nhưng không nhất quán (backbone thực ra không bị freeze
   trong `trainbase()`). Bản refactor làm rõ và bắt buộc: epoch đầu luôn khởi
   động ổn định bằng cách chỉ train classifier, giữ nguyên toàn bộ đặc trưng
   pretrained + BN stats.
2. **Differential LR (giai đoạn `last_block`)** — mở 3 block cuối của backbone
   với `lr_backbone = 1e-5`, thấp hơn 10 lần `lr_head = 1e-4` (giữ đúng tỉ lệ
   đã có trong `config` gốc của `trainbase()`), để thích nghi domain mà không
   phá vỡ các đặc trưng tổng quát ở layer đầu.
3. **Progressive Unfreezing (giai đoạn `full`, tuỳ chọn)** — chỉ mở toàn bộ
   mạng ở epoch muộn hơn, và chỉ nên bật nếu `num_epochs`/dataset đủ lớn để
   không overfit. Toàn bộ lịch trình (`freeze_schedule`) khai báo trong
   `params.yaml`, không hard-code, để dễ thử nghiệm A/B mà không sửa code.

Việc chuyển giai đoạn được cài đặt trong `src/models/cnn.py::CNNBaseline.set_freeze_stage()`
và được `src/training/trainer.py::train_model()` tự động gọi + build lại
optimizer đúng epoch theo `config["freeze_schedule"]`.

## 2. Cấu trúc MLOps cấp độ 2 (so với notebook gốc)

MLOps Level 2 (theo mô hình trưởng thành phổ biến: L0 thủ công → L1 tự động
hoá pipeline huấn luyện → **L2 tự động hoá CI/CD cho chính pipeline**) yêu cầu:
pipeline được đóng gói thành các thành phần độc lập, có thể kiểm thử, tái lập,
và tự động trigger lại đúng phần bị ảnh hưởng khi code/tham số/dữ liệu đổi.

| Vấn đề trong notebook gốc | Điều chỉnh trong bản refactor |
|---|---|
| 41 cell tuần tự, biến toàn cục (`device`, `class_names`, `save_dir`...) chia sẻ ngầm giữa các cell | Tách thành package `src/` theo domain (`data`, `models`, `rules`, `gflownet`, `training`, `evaluation`, `utils`) — mỗi module có input/output rõ ràng, import được, test được. |
| Hằng số cấu hình rải rác (`NUM_EPOCHS`, `DATA_DIR`, path Kaggle hard-code `/kaggle/input/...`) | Gom về **1 file `params.yaml`** duy nhất — "single source of truth". Không còn path tuyệt đối hard-code trong code. |
| Không có cách nào biết stage nào cần chạy lại khi đổi 1 tham số | `dvc.yaml` khai báo `deps`/`outs`/`params` cho từng stage (baseline → features → rules → gflownet selection → rule-regularized). `dvc repro` tự tính lại đúng phần phụ thuộc bị ảnh hưởng — đây là "Continuous Training" đúng nghĩa L2. |
| `print()` rải rác, không log ra file, không log level | `src/utils/logging_utils.py`: logger có timestamp, level, ghi ra cả console + file `logs/pipeline.log`. |
| Chiến lược freeze/LR hard-code trong nhiều hàm (`trainbase`, `train_rule`) không nhất quán với nhau | Logic freeze/differential-LR tập trung 1 nơi (`CNNBaseline.set_freeze_stage`, `trainable_param_groups`) — sửa 1 chỗ, áp dụng cho cả 2 pha huấn luyện (baseline & rule-regularized). |
| Không có unit test | `tests/test_rules.py`, `tests/test_cnn_freeze_strategy.py` — kiểm tra logic đóng băng, penalty, rule set. Chạy trong CI trước khi merge. |
| Không có CI | `.github/workflows/ci.yml`: lint (flake8) + unit test tự động trên mỗi PR. Có ghi chú rõ bước `dvc repro` + so sánh metrics cần GPU runner riêng (production thật). |
| Không đóng gói môi trường | `Dockerfile` build từ base image PyTorch+CUDA chính thức, cài `requirements.txt`, chạy `dvc repro` — đảm bảo "chạy đâu cũng ra kết quả giống nhau". |
| DVCLive đã có nhưng metrics/artefact không được `dvc.yaml` track làm output chính thức | `dvc.yaml` khai báo `metrics:` trỏ tới `dvclive/metrics.json` của từng stage, cho phép `dvc metrics diff` so sánh giữa các lần chạy/branch — điều kiện cần cho gate CI "không cho merge nếu accuracy giảm". |
| Không tách biệt code library vs script thực thi | `src/` = logic tái sử dụng (import được), `pipelines/` = script CLI mỏng, chỉ orchestration (đọc config → gọi src → lưu output) — đúng nguyên tắc "thin entrypoint, thick library" của MLOps. |

## 3. Cấu trúc thư mục

```
project/
├── params.yaml              # cấu hình trung tâm (data, model, transfer learning, GFlowNet...)
├── dvc.yaml                 # định nghĩa 5 stage pipeline + deps/outs/params/metrics
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data/                # dataset, dataloader, feature extraction
│   ├── models/               # CNNBaseline (chiến lược transfer learning), ProxyRewardNet
│   ├── rules/                # Rule/RuleSet, extractor, GPU validator, penalty
│   ├── gflownet/              # env (DiscreteEnv) + pipeline huấn luyện GFlowNet
│   ├── training/              # trainer (progressive unfreezing), optimizer, early stopping
│   ├── evaluation/            # evaluate + plot
│   └── utils/                 # seed, logging, config loader
├── pipelines/                # 5 script CLI mỏng, mỗi script = 1 DVC stage
│   ├── stage1_train_baseline.py
│   ├── stage2_extract_features.py
│   ├── stage3_extract_rules.py
│   ├── stage4_select_rules_gflownet.py
│   └── stage5_train_rule_regularized.py
├── tests/                     # unit test cho rules + chiến lược freeze
└── .github/workflows/ci.yml   # lint + test tự động
```

## 4. Cách chạy

```bash
pip install -r requirements.txt

# Chạy toàn bộ pipeline, DVC tự xác định thứ tự & cache những gì chưa đổi
dvc repro

# Hoặc chạy từng stage thủ công khi debug
python -m pipelines.stage1_train_baseline --config params.yaml
python -m pipelines.stage2_extract_features --config params.yaml
python -m pipelines.stage3_extract_rules --config params.yaml
python -m pipelines.stage4_select_rules_gflownet --config params.yaml
python -m pipelines.stage5_train_rule_regularized --config params.yaml

# Test
pytest tests/ -v
```

## 5. Muốn thử nghiệm chiến lược freeze khác?

Chỉ cần sửa `transfer_learning.freeze_schedule` trong `params.yaml`, ví dụ để
dùng Full Fine-Tuning ngay từ đầu (nếu sau này có nhiều data hơn):

```yaml
transfer_learning:
  freeze_schedule:
    0: "full"
```

`dvc repro` sẽ tự phát hiện `params.yaml` đổi và chạy lại đúng `train_baseline`
(và các stage phụ thuộc), không cần sửa code.

## 6. Đổi metric cho EarlyStopping / chọn checkpoint tốt nhất

Mặc định theo dõi `val_acc` (càng cao càng tốt). Muốn đổi sang `val_loss`
(càng thấp càng tốt), chỉ cần sửa 1 dòng trong `params.yaml`:

```yaml
monitor_metric: "val_loss"
```

`EarlyStopping` (trong `src/training/callbacks.py`) tự chọn đúng chiều so
sánh (`mode="max"` hay `mode="min"`) theo bảng `MONITOR_MODES` trong
`src/training/trainer.py` — không cần sửa logic so sánh thủ công. Muốn theo
dõi metric khác (vd F1), chỉ cần đăng ký thêm 1 dòng vào `MONITOR_MODES`.
