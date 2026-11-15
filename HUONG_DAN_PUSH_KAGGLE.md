# Hướng dẫn đẩy notebook lên nhiều account Kaggle (chạy local)

Mục tiêu: sau khi đã test notebook xong, chạy **một lệnh** để đẩy toàn bộ notebook trong `generated_cbm_notebooks/` lên **5 account Kaggle**. Không tự động đẩy khi commit; chỉ đẩy khi bạn gõ lệnh.

---

## 1. Cài Kaggle CLI

```
pip install kaggle
kaggle --version
```

## 2. Lấy token của 5 account

Với mỗi account: đăng nhập Kaggle → **Settings** → **API** → **Create New Token**. Ghi lại `username` và `key` của từng account.

## 3. Tạo `accounts.json` ở thư mục gốc repo

```json
[
  {"username": "acc1", "key": "KEY1"},
  {"username": "acc2", "key": "KEY2"},
  {"username": "acc3", "key": "KEY3"},
  {"username": "acc4", "key": "KEY4"},
  {"username": "acc5", "key": "KEY5"}
]
```

**Bắt buộc:** thêm vào `.gitignore` để không lỡ commit token:

```
accounts.json
```

## 4. Tạo `scripts/push_kaggle.py`

```python
"""Đẩy các notebook trong generated_cbm_notebooks/ lên nhiều account Kaggle.

  python scripts/push_kaggle.py                      # tất cả notebook, tất cả account
  python scripts/push_kaggle.py seeds_42_44          # chỉ notebook có tên chứa chuỗi này
  python scripts/push_kaggle.py --accounts acc1      # chỉ 1 account (để thử trước)
  python scripts/push_kaggle.py --dry-run            # xem sẽ đẩy gì, chưa đẩy thật
"""
import argparse, json, os, shutil, subprocess, tempfile
from pathlib import Path

NB_DIR = Path("generated_cbm_notebooks")
DATASET = "dangduykhoab2207532/mykhmer"

p = argparse.ArgumentParser()
p.add_argument("names", nargs="*", help="lọc notebook theo chuỗi con của tên file")
p.add_argument("--accounts", nargs="*", help="chỉ đẩy lên các username này")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

accounts = json.loads(Path("accounts.json").read_text(encoding="utf-8"))
if args.accounts:
    accounts = [a for a in accounts if a["username"] in args.accounts]

notebooks = [f for f in sorted(NB_DIR.glob("*.ipynb"))
             if not args.names or any(n in f.stem for n in args.names)]

print(f"{len(notebooks)} notebook x {len(accounts)} account = {len(notebooks) * len(accounts)} lần đẩy")
failed = []

for acc in accounts:
    env = {**os.environ, "KAGGLE_USERNAME": acc["username"], "KAGGLE_KEY": acc["key"]}
    for nb in notebooks:
        slug = nb.stem.lower().replace("_", "-")
        print(f"[{acc['username']}] {slug}")
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shutil.copy(nb, work / nb.name)
            meta = {
                "id": f"{acc['username']}/{slug}",
                "title": slug,
                "code_file": nb.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,   # notebook cần git clone repo
                "dataset_sources": [DATASET],
                "competition_sources": [],
            }
            (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            r = subprocess.run(["kaggle", "kernels", "push", "-p", str(work)], env=env)
            if r.returncode != 0:
                failed.append(f"{acc['username']}/{slug}")

if failed:
    print("LỖI:", *failed, sep="\n  ")
    raise SystemExit(1)
print("Xong.")
```

## 5. Quy trình làm việc

```
# 1. Sinh notebook bằng script có sẵn (lặp cho đủ 10 notebook, đổi --backbone / --seeds)
python scripts/generate_cbm_kaggle_notebook.py --seeds 42 44
python scripts/generate_cbm_kaggle_notebook.py --seeds 46 48

# 2. Test trên 1 account (xem mục 6)

# 3. Khi đã ổn, đẩy lên cả 5 account
python scripts/push_kaggle.py
```

## 6. Test an toàn trước khi đẩy hàng loạt

1. Xem trước, chưa đẩy thật:
   ```
   python scripts/push_kaggle.py --dry-run
   ```
2. Đẩy thử 1 notebook lên 1 account:
   ```
   python scripts/push_kaggle.py seeds_42_44 --accounts acc1
   ```
3. Kiểm tra trạng thái:
   ```
   kaggle kernels status acc1/dual-gpu-cbm-mobilenetv3-small-seeds-42-44
   ```
   Trạng thái: `queued` → `running` → `complete` (hoặc `error`). Mở log trên web để xác nhận chạy đúng, đủ 2 GPU, rồi mới đẩy toàn bộ.

---

## Lưu ý quan trọng

- **Đẩy là chạy luôn:** `kaggle kernels push` lưu phiên bản mới *và* kích hoạt chạy. CLI không có tùy chọn "chỉ lưu". Đẩy đủ 10 × 5 = 50 phiên chạy, tốn quota GPU hằng tuần của từng account và có thể phải xếp hàng.
- **Giảm lãng phí:** đẩy riêng notebook vừa sửa (`python scripts/push_kaggle.py <chuỗi-tên>`), và luôn thử 1 account trước.
- **Hủy phiên chạy:** CLI không có lệnh hủy; vào trang notebook trên web và bấm dừng.
- **GPU T4 x2:** `"enable_gpu": true` có thể không đảm bảo đúng 2 GPU. Ở lần chạy thử, kiểm tra log `torch.cuda.device_count()` bằng 2. Nếu không, kiểm tra `kaggle kernels push -h` xem có tùy chọn accelerator không, hoặc chọn T4 x2 thủ công trên web một lần.
- **Dataset:** đã share nên 4 account còn lại gắn được `dangduykhoab2207532/mykhmer`. Đường dẫn `/kaggle/input/datasets/dangduykhoab2207532/mykhmer/...` trong notebook giữ nguyên cho mọi account.
- **`n_concepts` lệch trong script sinh notebook:** hàm mặc định là `49`, tham số dòng lệnh `--n-concepts` mặc định là `61`. Chạy bằng CLI thực tế dùng `61`. Hãy chắc `image_concepts.json` đúng 61 concept, hoặc thống nhất lại hai giá trị.
- **Tên notebook không trùng:** slug lấy từ tên file (đã chứa backbone và cặp seed). Push lên notebook đã tồn tại sẽ tạo phiên bản mới, không tạo bản trùng.
- **Bảo mật:** không commit `accounts.json`; nếu lỡ lộ key, vào Kaggle tạo lại token (Create New Token sẽ vô hiệu key cũ).
- **Xác thực:** nếu CLI báo lỗi với `KAGGLE_USERNAME`/`KAGGLE_KEY`, kiểm tra tài liệu phiên bản CLI bạn cài, vì Kaggle có thể đã đổi kiểu token.
