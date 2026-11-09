# Two-phase dual-GPU workflow (DB only)

Workflow mới tách thí nghiệm thành hai loại notebook để tận dụng GPU T4 x2
mà không vi phạm dependency giữa các stage.

## Phase A — Stage 1 đến Stage 4

Mỗi notebook chứa hai run độc lập. GPU 0 và GPU 1 cùng chạy tuần tự:

```text
Stage 1 baseline → Stage 2 features → Stage 3 raw rules → Stage 4 GFN-DB
```

Với 2 backbone × 2 dataset × 5 seed có 20 run, generator ghép thành 10
notebook, không để GPU thứ hai rảnh ở seed cuối của từng dataset.

## Phase B — Stage 5 DB

Mỗi notebook xử lý một backbone/dataset/seed. Add Output của notebook Phase A
có run tương ứng làm Kaggle Input. Hai GPU lấy việc động từ cùng một hàng đợi:

1. GFN-DB Bayesian
2. GFN-DB Bayesian Elite
3. GFN-DB fixed-prior
4. Random
5. Top-K Confidence
6. Greedy Coverage

Không chạy TB fixed-prior, TB Bayesian hoặc TB Bayesian Elite.

## DVCLive trong chế độ dual-GPU

Các worker đặt `DISABLE_DVCLIVE=1`. DVCLive không được khởi tạo trong trainer
CNN hoặc GFlowNet, tránh nhiều process cùng tương tác với một DVC repository.
Checkpoint, history, exact metrics, CSV/JSON và log riêng của từng worker vẫn
được lưu. `PYTHONWARNINGS=ignore`, `DVCLIVE_LOGLEVEL=ERROR` và
`DVC_NO_ANALYTICS=1` cũng được truyền trực tiếp xuống subprocess.

## Sinh notebook

```powershell
python scripts/generate_two_phase_dual_gpu_notebooks.py `
  --output-dir generated_two_phase_dual_gpu_notebooks_v2 `
  --git-commit <FULL_40_CHARACTER_PUSHED_SHA> `
  --backbones efficientnet_b0 shufflenet_v2_x1_0 `
  --datasets culture-a culture-b `
  --seeds 42 44 46 48 50 `
  --seeds-per-prior-notebook 2
```

Kết quả gồm 10 notebook Phase A và 20 notebook Phase B.
