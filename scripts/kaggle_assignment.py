"""Ánh xạ notebook concept rules -> notebook CBM cũ (kernel nguồn) và account phụ trách.

Kaggle chỉ cho phép ``kernel_sources`` trỏ tới kernel mà tài khoản đẩy notebook có quyền
đọc. Kernel cũ thường là riêng tư và thuộc đúng account đã đẩy nó, nên notebook mới phải
được đẩy bởi CHÍNH account đó. Vì vậy hàm ``owner_assignment`` không chia vòng tròn mà gán
theo chủ sở hữu notebook cũ.

Quy ước tên:
  concept_rules_<backbone>_seeds_<a>_<b>.ipynb  <-  cbm_<backbone>_seeds_<a>_<b>.ipynb
  slug Kaggle = tên file viết thường, '_' -> '-'
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

CONCEPT_PREFIX = "concept_rules_"
PREVIOUS_PREFIX = "cbm_"


def slug_of(stem: str) -> str:
    return stem.lower().replace("_", "-")


def round_robin_assignment(nb_paths: Sequence[Path], accounts: Sequence[dict]) -> list[tuple[dict, Path]]:
    """Chia đều theo thứ tự (hành vi cũ của push_kaggle)."""
    base, extra = divmod(len(nb_paths), len(accounts))
    out: list[tuple[dict, Path]] = []
    start = 0
    for i, acc in enumerate(accounts):
        n = base + (1 if i < extra else 0)
        for nb in nb_paths[start:start + n]:
            out.append((acc, nb))
        start += n
    return out


def previous_notebook_name(concept_name: str) -> str:
    if not concept_name.startswith(CONCEPT_PREFIX):
        raise ValueError(f"Không phải notebook concept rules: {concept_name}")
    return PREVIOUS_PREFIX + concept_name[len(CONCEPT_PREFIX):]


def build_previous_index(prev_nbs: Sequence[Path], accounts: Sequence[dict]) -> dict[str, tuple[str, str]]:
    """Tên notebook cũ -> (username chủ sở hữu, 'username/slug' của kernel).

    Dùng đúng phép chia vòng tròn mà lần đẩy trước đã dùng, nên ánh xạ khớp với
    kernel thực tế đã tạo trên từng account.
    """
    index: dict[str, tuple[str, str]] = {}
    for acc, nb in round_robin_assignment(list(prev_nbs), accounts):
        index[nb.name] = (acc["username"], f"{acc['username']}/{slug_of(nb.stem)}")
    return index


def owner_assignment(
    concept_nbs: Sequence[Path],
    prev_index: dict[str, tuple[str, str]],
    accounts: Sequence[dict],
) -> list[tuple[dict, Path, str]]:
    """Trả về [(account, notebook concept, kernel_source 'user/slug')].

    Dừng với lỗi nếu thiếu notebook cũ hoặc account chủ sở hữu không có trong accounts.json,
    để không đẩy một nửa lô rồi mới phát hiện.
    """
    by_username = {acc["username"]: acc for acc in accounts}
    problems: list[str] = []
    plan: list[tuple[dict, Path, str]] = []
    for nb in concept_nbs:
        try:
            old_name = previous_notebook_name(nb.name)
        except ValueError as e:
            problems.append(str(e))
            continue
        if old_name not in prev_index:
            problems.append(f"{nb.name}: thiếu notebook cũ {old_name}")
            continue
        owner, kernel = prev_index[old_name]
        if owner not in by_username:
            problems.append(f"{nb.name}: chủ sở hữu {owner} không có trong accounts.json")
            continue
        plan.append((by_username[owner], nb, kernel))
    if problems:
        raise RuntimeError("Không thể ánh xạ kernel nguồn:\n  " + "\n  ".join(problems))
    return plan
