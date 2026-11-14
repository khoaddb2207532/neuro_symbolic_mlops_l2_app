# pyright: strict
"""Audit chất lượng per-image concept annotations (``image_concepts.json``).

Trả lời 3 câu hỏi TRƯỚC khi huấn luyện lại CBM:

1. Mỗi ảnh đang được gán bao nhiêu concept (so với kỳ vọng của lớp đó)?
2. Concept nào hiếm / quá phổ biến / không phân biệt lớp / "rò" sang lớp khác?
3. Nhãn có khớp với cách ``ConceptAnnotatedDataset`` tra cứu không (path, trùng tên file)?

Khái niệm "home class" của concept:
  - Có prototype (``class_concepts`` trong ``concept_annotations.json``): home = các lớp
    có concept đó trong prototype.
  - Không có prototype: home = các lớp mà tần suất concept >= ``shared_rel`` x tần suất cao nhất
    (concept dùng chung như "crowd", "monks" sẽ có nhiều home class, không bị coi là rò).
Concept active ở ảnh thuộc lớp KHÔNG phải home gọi là *foreign* (concept ngoài lớp).

Module chỉ phụ thuộc numpy, không đụng torch, nên chạy được ở bất kỳ đâu.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


# ---------------------------------------------------------------------------
# Cấu hình ngưỡng (đều là heuristic, chỉnh qua CLI)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditConfig:
    rare_freq: float = 0.02          # concept hiếm nếu tần suất < ngưỡng này
    common_freq: float = 0.80        # concept quá phổ biến nếu tần suất > ngưỡng này
    shared_rel: float = 0.5          # (không prototype) lớp là home nếu freq >= shared_rel * freq_max
    max_foreign_rate: float = 0.15   # concept đặc thù 1 lớp mà bật ở lớp khác > ngưỡng → "leaky"
    min_separation: float = 0.30     # home_rate - foreign_rate < ngưỡng → "low_separation"
    min_nmi: float = 0.05            # I(concept; class) / H(class) < ngưỡng → "low_nmi"
    jaccard_redundant: float = 0.80  # Jaccard >= ngưỡng → cặp concept trùng lặp
    min_pair_count: int = 5          # bỏ qua cặp có concept xuất hiện < n ảnh (Jaccard không ổn định)
    top_pairs: int = 15              # số cặp Jaccard cao nhất đưa vào báo cáo
    expected_per_image: float | None = None  # None = tự suy từ home class
    warn_ratio: float = 1.25         # mean_concepts / expected
    noisy_ratio: float = 1.50
    warn_foreign_frac: float = 0.25  # tỉ lệ concept ngoài lớp trong các concept active của ảnh
    noisy_foreign_frac: float = 0.40


# ---------------------------------------------------------------------------
# Nạp dữ liệu (biên: JSON không có kiểu → kiểm tra ở đây, phần sau dùng kiểu chắc chắn)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawRecord:
    path: str
    class_name: str | None
    vector: object


@dataclass(frozen=True)
class ConceptTable:
    concept_names: list[str]
    class_names: list[str]
    paths: list[str]
    labels: IntArray                # [N] chỉ số lớp của từng ảnh
    x: BoolArray                    # [N, K] nhãn concept nhị phân
    prototype: BoolArray | None     # [C, K] hoặc None
    issues: list[str]               # vấn đề phát hiện lúc nạp


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: cần là JSON object, nhận {type(data).__name__}")
    return cast(dict[str, object], data)


def _str_list(value: object, what: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"'{what}' cần là list[str]")
    out: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ValueError(f"'{what}' chứa phần tử không phải chuỗi: {item!r}")
        out.append(item)
    return out


def _class_names(value: object) -> list[str] | None:
    """Chấp nhận list hoặc dict {"0": "tên", ...} (định dạng của notebook labeling)."""
    if value is None:
        return None
    if isinstance(value, list):
        return _str_list(cast(list[object], value), "class_names")
    if not isinstance(value, dict):
        raise ValueError("'class_names' cần là list hoặc dict")
    pairs: list[tuple[int, str]] = []
    for key, name in cast(dict[object, object], value).items():
        if not isinstance(name, str):
            raise ValueError(f"class_names[{key!r}] không phải chuỗi")
        try:
            pairs.append((int(str(key)), name))
        except ValueError as exc:
            raise ValueError(f"class_names có key không phải số nguyên: {key!r}") from exc
    pairs.sort()
    if [index for index, _ in pairs] != list(range(len(pairs))):
        raise ValueError("class_names có chỉ số không liên tục từ 0")
    return [name for _, name in pairs]


def _vector(value: object, n_concepts: int) -> list[float] | None:
    """Trả list[float] nếu đúng độ dài và đúng kiểu số, ngược lại None."""
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if len(items) != n_concepts:
        return None
    out: list[float] = []
    for item in items:
        if isinstance(item, bool):
            out.append(1.0 if item else 0.0)
        elif isinstance(item, (int, float)):
            out.append(float(item))
        else:
            return None
    return out


def _read_records(path: Path, issues: list[str]) -> tuple[list[RawRecord], dict[str, object]]:
    """Đọc 1 file nhãn: ``.jsonl`` (từng dòng {path,class,vector}) hoặc ``.json``."""
    records: list[RawRecord] = []
    if path.suffix.lower() == ".jsonl":
        bad_lines = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed: object = json.loads(text)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if not isinstance(parsed, dict):
                    bad_lines += 1
                    continue
                row = cast(dict[str, object], parsed)
                raw_path = row.get("path")
                raw_class = row.get("class")
                if not isinstance(raw_path, str):
                    bad_lines += 1
                    continue
                records.append(
                    RawRecord(raw_path, raw_class if isinstance(raw_class, str) else None, row.get("vector"))
                )
        if bad_lines:
            issues.append(f"{path.name}: {bad_lines} dòng JSONL hỏng/thiếu path (đã bỏ qua)")
        return records, {}

    data = _read_json_object(path)
    items = data.get("image_concepts") or data.get("annotations")
    if not isinstance(items, dict):
        raise ValueError(f"{path}: thiếu 'image_concepts' hoặc 'annotations' (dict)")
    for key, value in cast(dict[object, object], items).items():
        if isinstance(value, dict):
            entry = cast(dict[str, object], value)
            raw_class = entry.get("class")
            records.append(
                RawRecord(str(key), raw_class if isinstance(raw_class, str) else None, entry.get("vector"))
            )
        else:
            records.append(RawRecord(str(key), None, value))
    return records, data


def _build_prototype(
    value: object,
    class_names: list[str],
    n_concepts: int,
    issues: list[str],
) -> BoolArray | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        issues.append("class_concepts không phải dict: bỏ qua prototype")
        return None
    proto = np.zeros((len(class_names), n_concepts), dtype=np.bool_)
    seen: set[int] = set()
    for key, vector in cast(dict[object, object], value).items():
        try:
            index = int(str(key))
        except ValueError:
            issues.append(f"class_concepts có key không phải số: {key!r}: bỏ qua prototype")
            return None
        parsed = _vector(vector, n_concepts)
        if parsed is None or not 0 <= index < len(class_names):
            issues.append(f"class_concepts[{key!r}] sai độ dài/chỉ số: bỏ qua prototype")
            return None
        proto[index] = np.asarray(parsed, dtype=np.float64) >= 0.5
        seen.add(index)
    if len(seen) != len(class_names):
        issues.append("class_concepts không đủ mọi lớp: bỏ qua prototype")
        return None
    return proto


def _resolve_class(path: str, declared: str | None, class_set: set[str] | None) -> str | None:
    if declared is not None:
        return declared
    parts = path.replace("\\", "/").strip("/").split("/")[:-1]
    if class_set is not None:
        return next((part for part in reversed(parts) if part in class_set), None)
    return parts[-1] if parts else None


def load_concept_table(paths: Sequence[Path], meta_path: Path | None = None) -> ConceptTable:
    """Nạp 1 hoặc nhiều file nhãn (``image_concepts.json`` hoặc các shard ``.jsonl``).

    ``meta_path`` (thường là ``concept_annotations.json``) cung cấp ``concept_names``,
    ``class_names`` và prototype ``class_concepts`` khi file nhãn không có (shard JSONL).
    Cùng 1 path xuất hiện nhiều lần: bản đọc sau cùng thắng (giống bước merge của notebook).
    """
    issues: list[str] = []
    headers: list[dict[str, object]] = []
    merged: dict[str, RawRecord] = {}
    overridden = 0
    for file_path in paths:
        records, header = _read_records(file_path, issues)
        headers.append(header)
        for record in records:
            if record.path in merged:
                overridden += 1
            merged[record.path] = record
    if overridden:
        issues.append(f"{overridden} path bị ghi đè do trùng giữa các file/shard (bản sau cùng được giữ)")
    meta = _read_json_object(meta_path) if meta_path is not None else {}

    def pick(key: str) -> object:
        for source in (*headers, meta):
            if source.get(key) is not None:
                return source[key]
        return None

    own_concepts = next((h["concept_names"] for h in headers if h.get("concept_names") is not None), None)
    names_value = own_concepts if own_concepts is not None else meta.get("concept_names")
    if names_value is None:
        raise ValueError("Không tìm thấy 'concept_names': truyền --meta concept_annotations.json")
    concept_names = _str_list(names_value, "concept_names")
    n_concepts = len(concept_names)

    own_classes = next((h["class_names"] for h in headers if h.get("class_names") is not None), None)
    class_names = _class_names(own_classes if own_classes is not None else meta.get("class_names"))
    class_set = set(class_names) if class_names is not None else None

    kept: list[tuple[str, str, list[float]]] = []
    bad_vector = unknown_class = soft_values = 0
    for record in merged.values():
        vector = _vector(record.vector, n_concepts)
        if vector is None:
            bad_vector += 1
            continue
        class_name = _resolve_class(record.path, record.class_name, class_set)
        if class_name is None or (class_set is not None and class_name not in class_set):
            unknown_class += 1
            continue
        if any(value not in (0.0, 1.0) for value in vector):
            soft_values += 1
        kept.append((record.path, class_name, vector))
    if bad_vector:
        issues.append(f"{bad_vector} ảnh có vector sai độ dài/kiểu (cần {n_concepts} số): đã bỏ qua")
    if unknown_class:
        issues.append(f"{unknown_class} ảnh không xác định được lớp hợp lệ: đã bỏ qua")
    if soft_values:
        issues.append(f"{soft_values} vector có giá trị ngoài {{0,1}}: nhị phân hoá với ngưỡng 0.5")
    if not kept:
        raise ValueError("Không có ảnh hợp lệ nào sau khi nạp nhãn")

    if class_names is None:
        class_names = sorted({class_name for _, class_name, _ in kept})
    index_of = {name: index for index, name in enumerate(class_names)}
    matrix = np.asarray([vector for _, _, vector in kept], dtype=np.float64) >= 0.5

    prototype: BoolArray | None = None
    proto_value = pick("class_concepts")
    if proto_value is not None:
        meta_concepts = meta.get("concept_names")
        meta_classes = _class_names(meta.get("class_names"))
        if meta_concepts is not None and _str_list(meta_concepts, "concept_names") != concept_names:
            issues.append("concept_names của --meta khác file nhãn: bỏ qua prototype (sẽ lệch cột)")
        elif meta_classes is not None and meta_classes != class_names:
            issues.append("class_names của --meta khác file nhãn: bỏ qua prototype (sẽ lệch lớp)")
        else:
            prototype = _build_prototype(proto_value, class_names, n_concepts, issues)

    return ConceptTable(
        concept_names=concept_names,
        class_names=class_names,
        paths=[path for path, _, _ in kept],
        labels=np.asarray([index_of[class_name] for _, class_name, _ in kept], dtype=np.int64),
        x=matrix,
        prototype=prototype,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Mô hình báo cáo
# ---------------------------------------------------------------------------

@dataclass
class CountStats:
    mean: float
    median: float
    std: float
    minimum: int
    maximum: int
    p90: float
    p95: float
    zero_fraction: float
    histogram: dict[str, int]


@dataclass
class ConceptRow:
    name: str
    n_pos: int
    freq: float
    top_class: str
    top_class_freq: float
    purity: float          # P(lớp = top_class | concept = 1)
    lift: float            # freq trong top_class / freq toàn cục
    mi_bits: float         # thông tin tương hỗ I(concept; lớp)
    nmi: float             # mi_bits / H(lớp)
    home_classes: list[str]
    home_rate: float       # P(concept = 1 | ảnh thuộc home class)
    foreign_rate: float    # P(concept = 1 | ảnh KHÔNG thuộc home class)
    separation: float      # home_rate - foreign_rate
    flags: list[str]


@dataclass
class ClassRow:
    name: str
    n_images: int
    n_home_concepts: int
    mean_concepts: float
    mean_home_active: float
    mean_foreign_active: float
    foreign_frac: float    # trung bình tỉ lệ concept ngoài lớp (chỉ tính ảnh có >= 1 concept)
    home_recall: float     # trung bình tỉ lệ concept home của lớp thực sự bật


@dataclass
class PairRow:
    a: str
    b: str
    jaccard: float
    n_both: int
    p_b_given_a: float
    p_a_given_b: float
    same_home: bool
    redundant: bool


@dataclass
class ImageRow:
    path: str
    class_name: str
    n_active: int
    n_home: int
    n_foreign: int
    foreign_frac: float
    foreign_concepts: list[str]


@dataclass
class Verdict:
    level: str             # "ok" | "warn" | "noisy"
    expected_per_image: float
    expected_source: str
    count_ratio: float
    mean_foreign_frac: float
    frac_majority_foreign: float
    messages: list[str]


@dataclass
class AuditReport:
    n_images: int
    n_concepts: int
    n_classes: int
    home_source: str
    issues: list[str]
    count_stats: CountStats
    concepts: list[ConceptRow]
    classes: list[ClassRow]
    top_pairs: list[PairRow]
    n_redundant_pairs: int
    freq_by_class: list[list[float]]   # [K][C]
    verdict: Verdict
    images: list[ImageRow]


# ---------------------------------------------------------------------------
# Phân tích
# ---------------------------------------------------------------------------

def _safe_div(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _mutual_information(
    n11: FloatArray, n_pos: FloatArray, n_y: FloatArray, n: int
) -> tuple[FloatArray, float]:
    """I(concept; lớp) theo bit cho từng concept, và entropy H(lớp)."""
    p_y = n_y / n
    entropy_y = float(-np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0])))
    mi = np.zeros(n11.shape[0], dtype=np.float64)
    for joint, p_x in ((n11, n_pos / n), (n_y[None, :] - n11, 1.0 - n_pos / n)):
        p_xy = joint / n
        denom = p_x[:, None] * p_y[None, :]
        ratio = np.ones_like(p_xy)
        np.divide(p_xy, denom, out=ratio, where=(p_xy > 0) & (denom > 0))
        mi += np.sum(np.where(p_xy > 0, p_xy * np.log2(ratio), 0.0), axis=1)
    return np.maximum(mi, 0.0), entropy_y


def _count_stats(counts: IntArray) -> CountStats:
    bins = np.bincount(counts)
    histogram = {str(i): int(bins[i]) for i in range(len(bins)) if bins[i] > 0}
    return CountStats(
        mean=float(counts.mean()),
        median=float(np.median(counts)),
        std=float(counts.std()),
        minimum=int(counts.min()),
        maximum=int(counts.max()),
        p90=float(np.percentile(counts, 90)),
        p95=float(np.percentile(counts, 95)),
        zero_fraction=float((counts == 0).sum()) / float(counts.size),
        histogram=histogram,
    )


def _verdict(
    cfg: AuditConfig,
    mean_count: float,
    expected: float,
    expected_source: str,
    mean_foreign_frac: float,
    frac_majority_foreign: float,
) -> Verdict:
    ratio = mean_count / expected if expected > 0 else math.inf
    messages: list[str] = []
    level = "ok"
    if ratio >= cfg.noisy_ratio:
        level = "noisy"
        messages.append(
            f"Trung bình {mean_count:.2f} concept/ảnh, gấp {ratio:.2f} lần kỳ vọng {expected:.2f}: "
            "nhiều khả năng dư concept."
        )
    elif ratio >= cfg.warn_ratio:
        level = "warn"
        messages.append(f"Trung bình {mean_count:.2f} concept/ảnh, cao hơn kỳ vọng {expected:.2f} ({ratio:.2f}x).")
    if mean_foreign_frac >= cfg.noisy_foreign_frac:
        level = "noisy"
        messages.append(f"{mean_foreign_frac:.0%} concept active của một ảnh nằm ngoài lớp của ảnh: nhãn nhiễu nặng.")
    elif mean_foreign_frac >= cfg.warn_foreign_frac:
        if level == "ok":
            level = "warn"
        messages.append(f"{mean_foreign_frac:.0%} concept active của một ảnh nằm ngoài lớp của ảnh.")
    if not messages:
        messages.append("Số concept/ảnh và tỉ lệ concept ngoài lớp trong ngưỡng chấp nhận.")
    return Verdict(
        level=level,
        expected_per_image=expected,
        expected_source=expected_source,
        count_ratio=ratio if math.isfinite(ratio) else 0.0,
        mean_foreign_frac=mean_foreign_frac,
        frac_majority_foreign=frac_majority_foreign,
        messages=messages,
    )


def analyze(table: ConceptTable, cfg: AuditConfig | None = None) -> AuditReport:
    cfg = cfg or AuditConfig()
    x, labels = table.x, table.labels
    n, k = x.shape
    c = len(table.class_names)
    xf = x.astype(np.float64)
    onehot = np.zeros((n, c), dtype=np.float64)
    onehot[np.arange(n), labels] = 1.0
    n_y = onehot.sum(axis=0)
    n_pos = xf.sum(axis=0)
    n11 = xf.T @ onehot                                    # [K, C]
    freq_by_class = n11 / np.maximum(n_y, 1.0)[None, :]    # [K, C]
    freq = n_pos / n

    # --- home class [C, K]
    if table.prototype is not None:
        home = table.prototype
        home_source = "prototype (class_concepts)"
    else:
        peak = freq_by_class.max(axis=1)                   # [K]
        home = ((freq_by_class >= cfg.shared_rel * peak[:, None]) & (peak[:, None] > 0)).T
        home_source = f"data-driven (lớp có freq >= {cfg.shared_rel:g} x freq cao nhất)"
    img_home: BoolArray = home[labels]                     # [N, K]
    foreign_mask: BoolArray = ~img_home

    home_pos = (x & img_home).sum(axis=0).astype(np.float64)
    home_n = img_home.sum(axis=0).astype(np.float64)
    foreign_pos = (x & foreign_mask).sum(axis=0).astype(np.float64)
    foreign_n = foreign_mask.sum(axis=0).astype(np.float64)
    home_rate = _safe_div(home_pos, home_n)
    foreign_rate = _safe_div(foreign_pos, foreign_n)

    top_idx = freq_by_class.argmax(axis=1)                 # [K]
    top_freq = freq_by_class[np.arange(k), top_idx]
    purity = _safe_div(n11[np.arange(k), top_idx], n_pos)
    lift = _safe_div(top_freq, freq)
    mi_bits, entropy_y = _mutual_information(n11, n_pos, n_y, n)
    nmi = mi_bits / entropy_y if entropy_y > 0 else np.zeros_like(mi_bits)

    concept_rows: list[ConceptRow] = []
    for j, name in enumerate(table.concept_names):
        home_ids = np.flatnonzero(home[:, j])
        flags: list[str] = []
        if freq[j] < cfg.rare_freq:
            flags.append("rare")
        if freq[j] > cfg.common_freq:
            flags.append("common")
        if len(home_ids) == 1 and foreign_rate[j] > cfg.max_foreign_rate:
            flags.append("leaky")
        if foreign_n[j] > 0 and home_rate[j] - foreign_rate[j] < cfg.min_separation:
            flags.append("low_separation")
        if nmi[j] < cfg.min_nmi:
            flags.append("low_nmi")
        if len(home_ids) > 1:
            flags.append("shared")
        concept_rows.append(
            ConceptRow(
                name=name,
                n_pos=int(n_pos[j]),
                freq=float(freq[j]),
                top_class=table.class_names[int(top_idx[j])],
                top_class_freq=float(top_freq[j]),
                purity=float(purity[j]),
                lift=float(lift[j]),
                mi_bits=float(mi_bits[j]),
                nmi=float(nmi[j]),
                home_classes=[table.class_names[int(i)] for i in home_ids],
                home_rate=float(home_rate[j]),
                foreign_rate=float(foreign_rate[j]),
                separation=float(home_rate[j] - foreign_rate[j]),
                flags=flags,
            )
        )

    # --- theo ảnh
    counts: IntArray = x.sum(axis=1).astype(np.int64)
    n_home_img: IntArray = (x & img_home).sum(axis=1).astype(np.int64)
    n_foreign_img: IntArray = counts - n_home_img
    foreign_frac_img = _safe_div(n_foreign_img.astype(np.float64), counts.astype(np.float64))
    active = counts > 0
    mean_foreign_frac = float(foreign_frac_img[active].mean()) if active.any() else 0.0
    frac_majority = float(np.mean(foreign_frac_img[active] > 0.5)) if active.any() else 0.0

    image_rows: list[ImageRow] = []
    for i in range(n):
        foreign_ids = np.flatnonzero(x[i] & foreign_mask[i])
        image_rows.append(
            ImageRow(
                path=table.paths[i],
                class_name=table.class_names[int(labels[i])],
                n_active=int(counts[i]),
                n_home=int(n_home_img[i]),
                n_foreign=int(n_foreign_img[i]),
                foreign_frac=float(foreign_frac_img[i]),
                foreign_concepts=[table.concept_names[int(j)] for j in foreign_ids],
            )
        )
    image_rows.sort(key=lambda row: (-row.foreign_frac, -row.n_foreign, row.path))

    # --- theo lớp
    class_rows: list[ClassRow] = []
    for ci, name in enumerate(table.class_names):
        mask = labels == ci
        n_home_concepts = int(home[ci].sum())
        if not mask.any():
            class_rows.append(ClassRow(name, 0, n_home_concepts, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        sel_active = mask & active
        class_rows.append(
            ClassRow(
                name=name,
                n_images=int(mask.sum()),
                n_home_concepts=n_home_concepts,
                mean_concepts=float(counts[mask].mean()),
                mean_home_active=float(n_home_img[mask].mean()),
                mean_foreign_active=float(n_foreign_img[mask].mean()),
                foreign_frac=float(foreign_frac_img[sel_active].mean()) if sel_active.any() else 0.0,
                home_recall=float(n_home_img[mask].mean() / n_home_concepts) if n_home_concepts else 0.0,
            )
        )

    # --- cặp concept trùng lặp (Jaccard)
    inter = xf.T @ xf
    union = n_pos[:, None] + n_pos[None, :] - inter
    jaccard = _safe_div(inter, union)
    pairs: list[PairRow] = []
    n_redundant = 0
    for a in range(k):
        for b in range(a + 1, k):
            if n_pos[a] < cfg.min_pair_count or n_pos[b] < cfg.min_pair_count:
                continue
            redundant = bool(jaccard[a, b] >= cfg.jaccard_redundant)
            n_redundant += int(redundant)
            pairs.append(
                PairRow(
                    a=table.concept_names[a],
                    b=table.concept_names[b],
                    jaccard=float(jaccard[a, b]),
                    n_both=int(inter[a, b]),
                    p_b_given_a=float(inter[a, b] / n_pos[a]),
                    p_a_given_b=float(inter[a, b] / n_pos[b]),
                    same_home=bool(np.array_equal(home[:, a], home[:, b])),
                    redundant=redundant,
                )
            )
    pairs.sort(key=lambda row: -row.jaccard)

    if cfg.expected_per_image is not None:
        expected, expected_source = cfg.expected_per_image, "tham số --expected-per-image"
    else:
        expected = float(img_home.sum(axis=1).mean())
        expected_source = "số concept thuộc lớp của ảnh (trung bình)"
    verdict = _verdict(cfg, float(counts.mean()), expected, expected_source, mean_foreign_frac, frac_majority)

    return AuditReport(
        n_images=n,
        n_concepts=k,
        n_classes=c,
        home_source=home_source,
        issues=list(table.issues),
        count_stats=_count_stats(counts),
        concepts=concept_rows,
        classes=class_rows,
        top_pairs=pairs[: cfg.top_pairs],
        n_redundant_pairs=n_redundant,
        freq_by_class=[[float(v) for v in row] for row in freq_by_class],
        verdict=verdict,
        images=image_rows,
    )


# ---------------------------------------------------------------------------
# Khớp nhãn với dataset trên đĩa (mô phỏng tra cứu của ConceptAnnotatedDataset)
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    n_images_on_disk: int
    n_matched_path: int               # khớp theo path tương đối hoặc "lớp/tên"
    n_matched_basename_only: int      # chỉ khớp theo tên file (dễ nhầm nếu trùng tên)
    n_unmatched: int                  # không khớp: dataset sẽ ÂM THẦM dùng prototype lớp
    n_annotated_not_on_disk: int
    n_basename_collisions: int        # tên file có > 1 path trong file nhãn
    n_basename_collisions_cross_class: int
    unmatched_examples: list[str]
    basename_only_examples: list[str]
    collision_examples: list[str]


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _last_two(norm_path: str) -> str:
    return "/".join(norm_path.split("/")[-2:])


def check_dataset_coverage(
    table: ConceptTable, dataset_dir: Path, max_examples: int = 10
) -> CoverageReport:
    exact_keys: set[str] = set()
    basenames: set[str] = set()
    for path in table.paths:
        norm = _norm(path)
        exact_keys.update((path, norm, _last_two(norm)))
        basenames.add(Path(norm).name)

    disk_rel: set[str] = set()
    matched = basename_only = unmatched = total = 0
    unmatched_examples: list[str] = []
    basename_examples: list[str] = []
    for file_path in sorted(dataset_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        total += 1
        rel = file_path.relative_to(dataset_dir).as_posix()
        disk_rel.add(rel)
        disk_rel.add(_last_two(rel))
        if rel in exact_keys or _last_two(rel) in exact_keys:
            matched += 1
        elif file_path.name in basenames:
            basename_only += 1
            if len(basename_examples) < max_examples:
                basename_examples.append(rel)
        else:
            unmatched += 1
            if len(unmatched_examples) < max_examples:
                unmatched_examples.append(rel)

    not_on_disk = sum(1 for p in table.paths if _norm(p) not in disk_rel and _last_two(_norm(p)) not in disk_rel)

    by_name: dict[str, list[int]] = defaultdict(list)
    for index, path in enumerate(table.paths):
        by_name[Path(_norm(path)).name].append(index)
    collisions = {name: ids for name, ids in by_name.items() if len(ids) > 1}
    cross = 0
    collision_examples: list[str] = []
    for name, ids in sorted(collisions.items()):
        classes = {table.class_names[int(table.labels[i])] for i in ids}
        if len(classes) > 1:
            cross += 1
        if len(collision_examples) < max_examples:
            collision_examples.append(f"{name}: " + ", ".join(table.paths[i] for i in ids[:3]))

    return CoverageReport(
        n_images_on_disk=total,
        n_matched_path=matched,
        n_matched_basename_only=basename_only,
        n_unmatched=unmatched,
        n_annotated_not_on_disk=not_on_disk,
        n_basename_collisions=len(collisions),
        n_basename_collisions_cross_class=cross,
        unmatched_examples=unmatched_examples,
        basename_only_examples=basename_examples,
        collision_examples=collision_examples,
    )


# ---------------------------------------------------------------------------
# Xuất kết quả
# ---------------------------------------------------------------------------

FLAG_HELP: dict[str, str] = {
    "rare": "hiếm: gần như không có dữ liệu để học",
    "common": "quá phổ biến: không phân biệt lớp",
    "leaky": "đặc thù 1 lớp nhưng bật nhiều ở lớp khác",
    "low_separation": "tỉ lệ bật ở lớp home ~ ở lớp khác",
    "low_nmi": "gần như không mang thông tin về lớp",
    "shared": "dùng chung nhiều lớp (bình thường, chỉ để biết)",
}


def format_summary(report: AuditReport, coverage: CoverageReport | None = None, top_n: int = 15) -> str:
    cs, vd = report.count_stats, report.verdict
    label = {"ok": "OK", "warn": "CẢNH BÁO", "noisy": "NHIỄU"}[vd.level]
    lines: list[str] = [
        "=== AUDIT image_concepts ===",
        f"Ảnh: {report.n_images} | Concept: {report.n_concepts} | Lớp: {report.n_classes}",
        f"Home class: {report.home_source}",
        "",
        f"[Concept/ảnh] mean={cs.mean:.2f} median={cs.median:.1f} std={cs.std:.2f} "
        f"min={cs.minimum} max={cs.maximum} p90={cs.p90:.0f} p95={cs.p95:.0f} | ảnh 0 concept: {cs.zero_fraction:.1%}",
        f"  Kỳ vọng {vd.expected_per_image:.2f} ({vd.expected_source}) → tỉ lệ {vd.count_ratio:.2f}x",
        f"  Concept ngoài lớp: trung bình {vd.mean_foreign_frac:.1%} số concept active/ảnh; "
        f"{vd.frac_majority_foreign:.1%} ảnh có > 50% concept ngoài lớp",
        f"  → KẾT LUẬN: {label}",
    ]
    lines += [f"    - {message}" for message in vd.messages]
    lines += [
        "",
        "[Theo lớp]",
        f"  {'lớp':<24}{'n':>6}{'cpt':>6}{'mean':>7}{'home':>7}{'foreign':>9}{'fr.frac':>9}{'recall':>8}",
    ]
    for row in report.classes:
        lines.append(
            f"  {row.name[:23]:<24}{row.n_images:>6}{row.n_home_concepts:>6}{row.mean_concepts:>7.2f}"
            f"{row.mean_home_active:>7.2f}{row.mean_foreign_active:>9.2f}"
            f"{row.foreign_frac:>9.2f}{row.home_recall:>8.2f}"
        )
    lines.append("  (cpt = số concept home của lớp; foreign/fr.frac = concept ngoài lớp; recall = % concept home bật)")

    flagged = [r for r in report.concepts if any(f != "shared" for f in r.flags)]
    flagged.sort(key=lambda r: (-len([f for f in r.flags if f != "shared"]), -r.foreign_rate))
    lines += ["", f"[Concept bị gắn cờ: {len(flagged)}/{report.n_concepts}]"]
    if flagged:
        lines.append(f"  {'concept':<30}{'freq':>6}{'top':>16}{'pur':>6}{'forgn':>7}{'nmi':>6}  flags")
        for r in flagged[:top_n * 2]:
            lines.append(
                f"  {r.name[:29]:<30}{r.freq:>6.2f}{r.top_class[:15]:>16}{r.purity:>6.2f}"
                f"{r.foreign_rate:>7.2f}{r.nmi:>6.2f}  {','.join(r.flags)}"
            )
    used = sorted({f for r in flagged for f in r.flags})
    lines += [f"  {f}: {FLAG_HELP[f]}" for f in used]

    lines += [
        "",
        f"[Cặp concept trùng lặp: {report.n_redundant_pairs} cặp Jaccard >= ngưỡng; top {len(report.top_pairs)}]",
    ]
    for p in report.top_pairs:
        mark = "TRÙNG" if p.redundant else "     "
        home = "cùng home" if p.same_home else "khác home"
        lines.append(f"  {mark} J={p.jaccard:.2f} n={p.n_both:<5} {p.a} ~ {p.b} ({home})")

    if coverage is not None:
        lines += [
            "",
            "[Khớp nhãn với dataset trên đĩa]",
            f"  ảnh trên đĩa: {coverage.n_images_on_disk} | khớp path: {coverage.n_matched_path} | "
            f"chỉ khớp tên file: {coverage.n_matched_basename_only} | KHÔNG khớp: {coverage.n_unmatched}",
            f"  có nhãn nhưng không thấy trên đĩa: {coverage.n_annotated_not_on_disk}",
            f"  tên file trùng trong file nhãn: {coverage.n_basename_collisions} "
            f"(khác lớp: {coverage.n_basename_collisions_cross_class})",
        ]
        if coverage.n_unmatched:
            lines.append(
                "  ảnh không khớp sẽ bị gán prototype lớp (âm thầm) khi train: ví dụ "
                + "; ".join(coverage.unmatched_examples[:3])
            )
        if coverage.n_basename_collisions_cross_class:
            lines.append(
                "  tên file trùng khác lớp có thể bị tra nhầm vector qua khoá basename: "
                + "; ".join(coverage.collision_examples[:3])
            )
    if report.issues:
        lines += ["", "[Vấn đề khi nạp]"] + [f"  - {issue}" for issue in report.issues]
    return "\n".join(lines)


def _write_csv(path: Path, header: list[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_report(
    out_dir: Path,
    table: ConceptTable,
    report: AuditReport,
    coverage: CoverageReport | None = None,
) -> list[Path]:
    """Ghi JSON + CSV. ``image_stats.csv`` đã sắp xếp từ ảnh nghi ngờ nhất (đầu vào cho bước verification)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary = asdict(report)
    summary.pop("images")
    summary["coverage"] = asdict(coverage) if coverage is not None else None
    json_path = out_dir / "audit_report.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(json_path)

    path = out_dir / "concept_summary.csv"
    _write_csv(
        path,
        ["concept", "n_pos", "freq", "top_class", "top_class_freq", "purity", "lift", "mi_bits", "nmi",
         "home_classes", "home_rate", "foreign_rate", "separation", "flags"],
        (
            [r.name, r.n_pos, f"{r.freq:.4f}", r.top_class, f"{r.top_class_freq:.4f}", f"{r.purity:.4f}",
             f"{r.lift:.3f}", f"{r.mi_bits:.4f}", f"{r.nmi:.4f}", "|".join(r.home_classes),
             f"{r.home_rate:.4f}", f"{r.foreign_rate:.4f}", f"{r.separation:.4f}", "|".join(r.flags)]
            for r in report.concepts
        ),
    )
    written.append(path)

    path = out_dir / "concept_class_freq.csv"
    _write_csv(
        path,
        ["concept", *table.class_names],
        ([name, *(f"{v:.4f}" for v in row)] for name, row in zip(table.concept_names, report.freq_by_class)),
    )
    written.append(path)

    path = out_dir / "class_summary.csv"
    _write_csv(
        path,
        ["class", "n_images", "n_home_concepts", "mean_concepts", "mean_home_active", "mean_foreign_active",
         "foreign_frac", "home_recall"],
        (
            [r.name, r.n_images, r.n_home_concepts, f"{r.mean_concepts:.3f}", f"{r.mean_home_active:.3f}",
             f"{r.mean_foreign_active:.3f}", f"{r.foreign_frac:.3f}", f"{r.home_recall:.3f}"]
            for r in report.classes
        ),
    )
    written.append(path)

    path = out_dir / "top_pairs.csv"
    _write_csv(
        path,
        ["concept_a", "concept_b", "jaccard", "n_both", "p_b_given_a", "p_a_given_b", "same_home", "redundant"],
        (
            [p.a, p.b, f"{p.jaccard:.4f}", p.n_both, f"{p.p_b_given_a:.4f}", f"{p.p_a_given_b:.4f}",
             p.same_home, p.redundant]
            for p in report.top_pairs
        ),
    )
    written.append(path)

    path = out_dir / "image_stats.csv"
    _write_csv(
        path,
        ["path", "class", "n_active", "n_home", "n_foreign", "foreign_frac", "foreign_concepts"],
        (
            [r.path, r.class_name, r.n_active, r.n_home, r.n_foreign, f"{r.foreign_frac:.3f}",
             "|".join(r.foreign_concepts)]
            for r in report.images
        ),
    )
    written.append(path)
    return written
