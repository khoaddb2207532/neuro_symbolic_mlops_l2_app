# Hướng dẫn chia 10 notebook lên 5 account Kaggle (chạy local)

Mục tiêu: sinh 10 notebook (5 backbone × 2 cặp seed), sau khi test xong thì chạy **một lệnh** để **chia đều lên 5 account, mỗi account 2 notebook**. Không tự động đẩy khi commit; chỉ đẩy khi bạn gõ lệnh.

Tổng cộng chỉ có **10 lần chạy** trên Kaggle (mỗi notebook đúng 1 account), nên quota GPU của mỗi account chỉ phải gánh 2 notebook.

---

## 1. Cài Kaggle CLI

```
pip install kaggle
kaggle --version
```

## 2. Lấy token của 5 account

Với mỗi account: đăng nhập Kaggle → https://www.kaggle.com/settings/api → mục **API** → bấm **Generate New Token**. Sao chép chuỗi token hiện ra (chỉ hiện một lần, hãy lưu ngay). Cùng với **username** của account đó (phần cuối URL hồ sơ `kaggle.com/<username>`).

Kaggle CLI bản mới dùng token này qua biến `KAGGLE_API_TOKEN`; cặp `KAGGLE_USERNAME` + `KAGGLE_KEY` kiểu cũ sẽ báo lỗi "Authentication required".

## 3. Tạo `accounts.json` ở thư mục gốc repo

```json
[
  {"username": "acc1", "token": "TOKEN1"},
  {"username": "acc2", "token": "TOKEN2"},
  {"username": "acc3", "token": "TOKEN3"},
  {"username": "acc4", "token": "TOKEN4"},
  {"username": "acc5", "token": "TOKEN5"}
]
```

**Bắt buộc:** thêm vào `.gitignore` để không lỡ commit token:

```
accounts.json
```

Thứ tự trong file này quyết định notebook nào vào account nào, nên đừng đổi thứ tự sau khi đã đẩy.

## 4. Sinh 10 notebook: `scripts/generate_all.py`

```python
import subprocess, sys

BACKBONES = ["alexnet", "mobilenetv3_small", "resnet50", "swin_t", "vit_b_32"]
SEED_PAIRS = [(42, 44), (46, 48)]   # đổi nếu bạn dùng cặp seed khác

for bb in BACKBONES:
    for a, b in SEED_PAIRS:
        subprocess.run(
            [sys.executable, "scripts/generate_cbm_kaggle_notebook.py",
             "--backbone", bb, "--seeds", str(a), str(b),
             "--n-concepts", "61"],
            check=True,
        )
```

Chạy: `python scripts/generate_all.py`. Kết quả là 10 file trong `generated_cbm_notebooks/`.

## 5. Script chia và đẩy: `scripts/push_kaggle.py`

Cách chia: sắp xếp 10 notebook theo tên rồi chia thành các khối liên tiếp, mỗi account một khối. Tên file bắt đầu bằng backbone nên mỗi account nhận đúng **một backbone với cả hai cặp seed**. Cách chia này cố định, nên đẩy lại một notebook thì nó luôn về đúng account cũ.

```python
"""Chia đều các notebook trong generated_cbm_notebooks/ lên các account Kaggle.

  python scripts/push_kaggle.py --dry-run            # xem bảng phân công, chưa đẩy
  python scripts/push_kaggle.py                      # đẩy tất cả theo phân công
  python scripts/push_kaggle.py resnet50             # chỉ đẩy notebook có tên chứa chuỗi này
  python scripts/push_kaggle.py --accounts acc1      # chỉ đẩy phần của acc1 (để thử trước)
"""
import argparse, json, os, shutil, subprocess, tempfile
from pathlib import Path

NB_DIR = Path("generated_cbm_notebooks")
DATASET = "dangduykhoab2207532/mykhmer"

p = argparse.ArgumentParser()
p.add_argument("names", nargs="*", help="lọc notebook theo chuỗi con của tên file")
p.add_argument("--accounts", nargs="*", help="chỉ đẩy phần của các username này")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

accounts = json.loads(Path("accounts.json").read_text(encoding="utf-8"))
all_nbs = sorted(NB_DIR.glob("*.ipynb"))

# Phân công dựa trên TOÀN BỘ danh sách để không đổi khi bạn lọc
base, extra = divmod(len(all_nbs), len(accounts))
assignment, start = [], 0
for i, acc in enumerate(accounts):
    n = base + (1 if i < extra else 0)
    for nb in all_nbs[start:start + n]:
        assignment.append((acc, nb))
    start += n

# Lọc theo yêu cầu
jobs = [(a, nb) for a, nb in assignment
        if (not args.names or any(s in nb.stem for s in args.names))
        and (not args.accounts or a["username"] in args.accounts)]

print(f"{len(all_nbs)} notebook / {len(accounts)} account. Sẽ đẩy {len(jobs)}:")
for a, nb in jobs:
    print(f"  {a['username']:<15} <- {nb.name}")
if args.dry_run:
    raise SystemExit(0)

failed = []
for acc, nb in jobs:
    slug = nb.stem.lower().replace("_", "-")
    env = {k: v for k, v in os.environ.items()
           if k not in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")}
    env["KAGGLE_API_TOKEN"] = acc["token"]
    env["KAGGLE_USERNAME"] = acc["username"]
    print(f"\n[{acc['username']}] {slug}")
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
    print("\nLỖI:", *failed, sep="\n  ")
    raise SystemExit(1)
print("\nXong.")
```

Với 10 notebook và 5 account, bảng phân công sẽ là:

| Account | Notebook |
|---|---|
| acc1 | alexnet, seeds 42_44 và 46_48 |
| acc2 | mobilenetv3_small, seeds 42_44 và 46_48 |
| acc3 | resnet50, seeds 42_44 và 46_48 |
| acc4 | swin_t, seeds 42_44 và 46_48 |
| acc5 | vit_b_32, seeds 42_44 và 46_48 |

## 6. Quy trình làm việc

```
# 1. Sinh 10 notebook
python scripts/generate_all.py

# 2. Test từng notebook trước (xem mục 7)

# 3. Xem bảng phân công
python scripts/push_kaggle.py --dry-run

# 4. Đẩy thật, chia đều lên 5 account
python scripts/push_kaggle.py
```

## 7. Test an toàn trước khi đẩy

1. Xem bảng phân công, chưa đẩy: `python scripts/push_kaggle.py --dry-run`
2. Đẩy thử phần của 1 account: `python scripts/push_kaggle.py --accounts acc1`
3. Kiểm tra trạng thái:
   ```
   kaggle kernels status acc1/dual-gpu-cbm-alexnet-seeds-42-44
   ```
   Trạng thái: `queued` → `running` → `complete` (hoặc `error`). Mở log trên web, xác nhận chạy đúng và đủ 2 GPU, rồi mới đẩy phần còn lại.

---

## Lưu ý quan trọng

- **Đẩy là chạy luôn:** `kaggle kernels push` lưu phiên bản mới *và* kích hoạt chạy; CLI không có tùy chọn "chỉ lưu". Hủy phiên chạy phải làm trên web.
- **Đẩy lại sau khi sửa:** `python scripts/push_kaggle.py swin_t` chỉ đẩy các notebook swin_t, về đúng account cũ và tạo phiên bản mới (không tạo bản trùng).
- **GPU T4 x2:** `"enable_gpu": true` có thể không đảm bảo đúng 2 GPU. Ở lần chạy thử, kiểm tra log `torch.cuda.device_count()` bằng 2. Nếu không, xem `kaggle kernels push -h` có tùy chọn accelerator không, hoặc chọn T4 x2 thủ công trên web một lần.
- **Backbone nặng:** `swin_t`, `vit_b_32`, `resnet50` có thể cần `--batch-size 32` nếu hết VRAM; thêm tham số này vào `generate_all.py` cho các backbone đó.
- **Dataset:** đã share nên 4 account còn lại gắn được `dangduykhoab2207532/mykhmer`. Đường dẫn `/kaggle/input/datasets/dangduykhoab2207532/mykhmer/...` giữ nguyên.
- **`n_concepts` lệch trong script sinh notebook:** hàm mặc định `49`, tham số dòng lệnh `--n-concepts` mặc định `61`. `generate_all.py` truyền rõ `61`; hãy chắc `image_concepts.json` đúng 61 concept.
- **Bảo mật:** không commit `accounts.json`; nếu lỡ lộ key, tạo lại token trên Kaggle (Create New Token vô hiệu key cũ).
- **Xác thực:** CLI bản mới dùng `KAGGLE_API_TOKEN`. Script đã xóa các biến Kaggle cũ trong môi trường trước khi đặt token của từng account. Đừng chạy `kaggle auth login` hay lưu `~/.kaggle/access_token` trên máy này, vì thông tin đăng nhập lưu sẵn có thể làm các lần đẩy dính vào một account duy nhất.
- **Kiểm tra token từng account** trước khi đẩy:
  ```
  # Windows (cmd)
  set KAGGLE_API_TOKEN=TOKEN1
  kaggle kernels list --mine
  ```
  (Linux/macOS dùng `export KAGGLE_API_TOKEN=TOKEN1`). Liệt kê được mà không báo lỗi là token đúng.