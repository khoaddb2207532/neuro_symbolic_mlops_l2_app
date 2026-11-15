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
