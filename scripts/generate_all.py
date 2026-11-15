import subprocess, sys

BACKBONES = ["alexnet", "mobilenetv3_small", "resnet50", "swin_t", "vit_b_32"]
SEED_PAIRS = [(42, 44), (46, 48)]

for bb in BACKBONES:
    for a, b in SEED_PAIRS:
        subprocess.run(
            [sys.executable, "scripts/generate_cbm_kaggle_notebook.py",
             "--backbone", bb, "--seeds", str(a), str(b),
             "--n-concepts", "54"],
            check=True,
        )