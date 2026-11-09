"""Generate five thesis-ready Kaggle classification-report notebooks."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


DEFAULT_BACKBONES = (
    "alexnet",
    "mobilenetv3_small",
    "resnet50",
    "swin_t",
    "vit_b_32",
)
DEFAULT_SEEDS = (42, 44, 46, 48, 50)
DATASETS = ("culture-a", "culture-b")


REPORT_MARKDOWN_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Tổng hợp classification report cho luận văn\n",
        "Cell dưới đây chỉ đọc các run đã được bước tạo bundle xác thực và sao chép vào "
        "`bundle_dir/runs`; không quét hoặc ghép trực tiếp dữ liệu từ `/kaggle/input`. "
        "Các chỉ số theo lớp được tổng hợp dưới dạng mean ± SD qua 5 seed, kèm mức "
        "thay đổi so với CNN baseline và ma trận nhầm lẫn chuẩn hóa.\n",
    ],
}


REPORT_CODE = r'''import tarfile
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

EXACT_METHOD_NAMES = {
    'cnn_baseline': 'baseline',
    'gflownet_db': 'gflownet_static',
    'gflownet_db_bayesian': 'gflownet_marginal',
    'random': 'random',
    'topk_confidence': 'topk_confidence',
    'greedy_coverage': 'greedy_coverage',
}
EXPECTED_METHODS = set(EXACT_METHOD_NAMES.values())

def mean_sd(mean_value, sd_value):
    sd_value = 0.0 if pd.isna(sd_value) else sd_value
    return f'{mean_value:.3f} ± {sd_value:.3f}'

def markdown_table(frame):
    columns = list(frame.columns)
    rows = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(str(value).replace('|', '\\|') for value in values) + ' |')
    return '\n'.join(rows)

def latex_escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')

def latex_table(frame, caption, label):
    alignment = 'l' + 'c' * (len(frame.columns) - 1)
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{{latex_escape(caption)}}}', rf'\label{{{label}}}',
        rf'\begin{{tabular}}{{{alignment}}}', r'\hline',
        ' & '.join(latex_escape(column) for column in frame.columns) + r' \\', r'\hline',
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append(' & '.join(latex_escape(value) for value in values) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}', r'\end{table}'])
    return '\n'.join(lines)

reports_root = bundle_dir / 'classification_reports'
matrices_root = bundle_dir / 'confusion_matrices'
tables_root = bundle_dir / 'thesis_tables'
figures_root = bundle_dir / 'thesis_figures'
for output_root in (reports_root, matrices_root, tables_root, figures_root):
    output_root.mkdir(parents=True, exist_ok=True)

records = []
metric_rows = []
matrix_records = []
markdown_parts = [f'# Classification reports - {BACKBONE}', '']

for run_id in EXPECTED_RUN_IDS:
    run_dir = bundle_dir / 'runs' / run_id
    manifest_path = run_dir / 'run_manifest.json'
    exact_path = run_dir / 'exact_test_metrics.json'
    if not manifest_path.exists() or not exact_path.exists():
        raise RuntimeError(
            f'Bundle đã xác thực nhưng thiếu run_manifest.json hoặc exact_test_metrics.json: {run_id}'
        )
    run_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if str(run_manifest.get('run_id')) != run_id or run_manifest.get('status') != 'complete':
        raise RuntimeError(f'Run trong bundle không hợp lệ: {run_id}')
    dataset_id = str(run_manifest['dataset_id'])
    seed = int(run_manifest['seed'])
    exact_results = json.loads(exact_path.read_text(encoding='utf-8'))
    if not isinstance(exact_results, list):
        exact_results = [exact_results]
    for result in exact_results:
        raw_method = str(result['method'])
        if raw_method not in EXACT_METHOD_NAMES:
            continue
        method = EXACT_METHOD_NAMES[raw_method]
        class_items = sorted(
            result['per_class'].items(),
            key=lambda item: int(item[1].get('class_index', 0)),
        )
        class_names = [name for name, _ in class_items]
        report_lines = [
            f'Classification report - {dataset_id} - seed {seed} - {method}', '',
            f'{"class":<30} {"precision":>10} {"recall":>10} {"f1-score":>10} {"support":>10}',
        ]
        for class_name, values in class_items:
            metric_rows.append({
                'dataset_id': dataset_id, 'backbone': BACKBONE, 'seed': seed,
                'method': method, 'class_name': class_name,
                'class_index': int(values['class_index']),
                'precision': float(values['precision']),
                'recall': float(values['recall']),
                'f1_score': float(values['f1']),
                'support': int(values['support']),
                'source_run_id': run_id,
            })
            report_lines.append(
                f'{class_name:<30} {float(values["precision"]):>10.4f} '
                f'{float(values["recall"]):>10.4f} {float(values["f1"]):>10.4f} '
                f'{int(values["support"]):>10d}'
            )
        report_lines.extend([
            '',
            f'accuracy: {float(result["accuracy"]):.6f}',
            f'macro precision: {float(result["macro_precision"]):.6f}',
            f'macro recall: {float(result["macro_recall"]):.6f}',
            f'macro F1: {float(result["macro_f1"]):.6f}',
            f'weighted F1: {float(result["weighted_f1"]):.6f}',
        ])
        report_text = '\n'.join(report_lines)
        report_path = reports_root / dataset_id / f'seed_{seed}' / f'{method}_classification_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding='utf-8')
        records.append({
            'dataset_id': dataset_id, 'seed': seed, 'backbone': BACKBONE,
            'method': method, 'report_file': str(report_path.relative_to(bundle_dir)),
            'source_run_id': run_id, 'metric_source': str(exact_path.relative_to(bundle_dir)),
        })
        markdown_parts.extend([
            f'## {dataset_id} - seed {seed} - {method}', '',
            '```text', report_text, '```', '',
        ])
        matrix = np.asarray(result['confusion_matrix'], dtype=np.int64)
        if matrix.shape != (len(class_names), len(class_names)):
            raise RuntimeError(f'Ma trận nhầm lẫn sai kích thước: {run_id}/{method}')
        per_seed_matrix_dir = matrices_root / dataset_id / f'seed_{seed}'
        per_seed_matrix_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
            per_seed_matrix_dir / f'{method}_confusion_matrix.csv'
        )
        matrix_records.append({
            'dataset_id': dataset_id, 'seed': seed, 'method': method,
            'class_names': class_names, 'matrix': matrix,
        })

if not records or not metric_rows:
    raise RuntimeError('Không tìm thấy đủ classification report và exact metrics')

report_index = pd.DataFrame(records).sort_values(['dataset_id', 'seed', 'method'])
per_seed = pd.DataFrame(metric_rows).sort_values(
    ['dataset_id', 'seed', 'method', 'class_index']
)
expected_pairs = {(dataset_id, seed) for dataset_id in ('culture-a', 'culture-b') for seed in SEEDS}
observed_pairs = set(zip(report_index['dataset_id'], report_index['seed']))
if observed_pairs != expected_pairs:
    raise RuntimeError(f'Report thiếu dataset/seed: {sorted(expected_pairs - observed_pairs)}')
method_sets = per_seed.groupby(['dataset_id', 'seed'])['method'].agg(set)
incomplete = {
    f'{dataset_id}/seed_{seed}': sorted(EXPECTED_METHODS - methods)
    for (dataset_id, seed), methods in method_sets.items()
    if methods != EXPECTED_METHODS
}
if incomplete:
    raise RuntimeError(f'Exact metrics chưa đủ sáu phương pháp: {incomplete}')

class_sets = per_seed.groupby(['dataset_id', 'seed', 'method'])['class_name'].agg(tuple)
for dataset_id in per_seed['dataset_id'].unique():
    variants = set(class_sets.loc[dataset_id])
    if len(variants) != 1:
        raise RuntimeError(f'Thứ tự lớp không nhất quán trong {dataset_id}')

report_index.to_csv(bundle_dir / 'classification_report_index.csv', index=False)
per_seed.to_csv(bundle_dir / 'classification_report_per_seed.csv', index=False)
(bundle_dir / 'classification_reports.md').write_text('\n'.join(markdown_parts), encoding='utf-8')

group_columns = ['dataset_id', 'backbone', 'method', 'class_index', 'class_name']
summary = per_seed.groupby(group_columns, as_index=False).agg(
    n_seeds=('seed', 'nunique'),
    precision_mean=('precision', 'mean'), precision_sd=('precision', 'std'),
    recall_mean=('recall', 'mean'), recall_sd=('recall', 'std'),
    f1_mean=('f1_score', 'mean'), f1_sd=('f1_score', 'std'),
    support=('support', 'first'), support_variants=('support', 'nunique'),
)
if not (summary['n_seeds'] == len(SEEDS)).all():
    raise RuntimeError('Có lớp/phương pháp không đủ số seed để tính mean ± SD')
if not (summary['support_variants'] == 1).all():
    raise RuntimeError('Support của cùng một lớp thay đổi giữa các seed')
summary.drop(columns='support_variants').to_csv(
    bundle_dir / 'classification_report_mean_std.csv', index=False
)

baseline = per_seed[per_seed['method'] == 'baseline'][
    ['dataset_id', 'backbone', 'seed', 'class_name', 'precision', 'recall', 'f1_score']
].rename(columns={
    'precision': 'baseline_precision', 'recall': 'baseline_recall',
    'f1_score': 'baseline_f1_score',
})
delta = per_seed[per_seed['method'] != 'baseline'].merge(
    baseline, on=['dataset_id', 'backbone', 'seed', 'class_name'], validate='many_to_one'
)
for metric in ('precision', 'recall', 'f1_score'):
    delta[f'delta_{metric}'] = delta[metric] - delta[f'baseline_{metric}']
delta.to_csv(bundle_dir / 'classification_report_delta_vs_baseline_per_seed.csv', index=False)
delta_summary = delta.groupby(
    ['dataset_id', 'backbone', 'method', 'class_index', 'class_name'], as_index=False
).agg(
    n_seeds=('seed', 'nunique'),
    delta_precision_mean=('delta_precision', 'mean'), delta_precision_sd=('delta_precision', 'std'),
    delta_recall_mean=('delta_recall', 'mean'), delta_recall_sd=('delta_recall', 'std'),
    delta_f1_mean=('delta_f1_score', 'mean'), delta_f1_sd=('delta_f1_score', 'std'),
)
delta_summary.to_csv(bundle_dir / 'classification_report_delta_vs_baseline.csv', index=False)

for dataset_id in sorted(summary['dataset_id'].unique()):
    selected = summary[
        (summary['dataset_id'] == dataset_id) &
        (summary['method'] == 'gflownet_marginal')
    ].copy()
    baseline_selected = summary[
        (summary['dataset_id'] == dataset_id) & (summary['method'] == 'baseline')
    ][['class_name', 'f1_mean', 'f1_sd']].rename(columns={
        'f1_mean': 'baseline_f1_mean', 'f1_sd': 'baseline_f1_sd',
    })
    thesis = selected.merge(baseline_selected, on='class_name', validate='one_to_one')
    thesis['delta_f1'] = thesis['f1_mean'] - thesis['baseline_f1_mean']
    thesis_display = pd.DataFrame({
        'Lớp': thesis['class_name'],
        'Precision': [mean_sd(m, s) for m, s in zip(thesis['precision_mean'], thesis['precision_sd'])],
        'Recall': [mean_sd(m, s) for m, s in zip(thesis['recall_mean'], thesis['recall_sd'])],
        'F1-score': [mean_sd(m, s) for m, s in zip(thesis['f1_mean'], thesis['f1_sd'])],
        'F1 baseline': [mean_sd(m, s) for m, s in zip(thesis['baseline_f1_mean'], thesis['baseline_f1_sd'])],
        'ΔF1': [f'{value:+.3f}' for value in thesis['delta_f1']],
        'Support': thesis['support'].astype(int),
    })
    stem = tables_root / f'{dataset_id}_gflownet_marginal_per_class'
    thesis_display.to_csv(stem.with_suffix('.csv'), index=False)
    stem.with_suffix('.md').write_text(markdown_table(thesis_display), encoding='utf-8')
    stem.with_suffix('.tex').write_text(
        latex_table(
            thesis_display,
            f'Classification report của GFlowNet-Marginal trên {dataset_id} ({BACKBONE})',
            f'tab:{dataset_id}-{BACKBONE}-classification-report',
        ),
        encoding='utf-8',
    )

for dataset_id in sorted({item['dataset_id'] for item in matrix_records}):
    for method in sorted(EXPECTED_METHODS):
        selected = [
            item for item in matrix_records
            if item['dataset_id'] == dataset_id and item['method'] == method
        ]
        if len(selected) != len(SEEDS):
            raise RuntimeError(f'Ma trận nhầm lẫn thiếu seed: {dataset_id}/{method}')
        reference_names = selected[0]['class_names']
        if any(item['class_names'] != reference_names for item in selected):
            raise RuntimeError(f'Thứ tự lớp của ma trận không nhất quán: {dataset_id}/{method}')
        matrix_sum = np.sum([item['matrix'] for item in selected], axis=0)
        row_sums = matrix_sum.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix_sum, row_sums, out=np.zeros_like(matrix_sum, dtype=float), where=row_sums != 0
        )
        matrix_csv = pd.DataFrame(normalized, index=reference_names, columns=reference_names)
        matrix_csv.to_csv(figures_root / f'{dataset_id}_{method}_confusion_matrix_normalized.csv')
        fig_size = max(8, len(reference_names) * 0.75)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        image = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(reference_names)), reference_names, rotation=45, ha='right')
        ax.set_yticks(range(len(reference_names)), reference_names)
        ax.set_xlabel('Nhãn dự đoán')
        ax.set_ylabel('Nhãn thực tế')
        ax.set_title(f'{dataset_id} - {method} - {BACKBONE} (gộp {len(SEEDS)} seed)')
        threshold = normalized.max() / 2 if normalized.size else 0
        for row in range(normalized.shape[0]):
            for column in range(normalized.shape[1]):
                ax.text(
                    column, row, f'{normalized[row, column]:.2f}',
                    ha='center', va='center',
                    color='white' if normalized[row, column] > threshold else 'black',
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Tỷ lệ theo lớp thực tế')
        fig.tight_layout()
        fig.savefig(
            figures_root / f'{dataset_id}_{method}_confusion_matrix_normalized.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

archive_path = bundle_dir.parent / f'{bundle_dir.name}.tar.gz'
with tarfile.open(archive_path, 'w:gz') as archive:
    archive.add(bundle_dir, arcname=bundle_dir.name)

print(f'Đã lưu {len(report_index)} report và {len(per_seed)} hàng metric theo lớp')
print('Bảng tổng hợp:', bundle_dir / 'classification_report_mean_std.csv')
print('Bảng luận văn:', tables_root)
print('Hình ma trận nhầm lẫn:', figures_root)
print('Archive cập nhật:', archive_path)
display(summary)
for table_path in sorted(tables_root.glob('*.md')):
    display(Markdown(table_path.read_text(encoding='utf-8')))
'''


REPORT_CODE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in REPORT_CODE.splitlines()],
}


def _parameter_cell(notebook: dict) -> dict:
    for cell in notebook["cells"]:
        if "bundle-parameters" in cell.get("metadata", {}).get("tags", []):
            return cell
    raise ValueError("Template has no cell tagged 'bundle-parameters'")


def _template_commit(notebook: dict) -> str:
    source = "".join(_parameter_cell(notebook).get("source", []))
    for line in source.splitlines():
        if line.strip().startswith("GIT_COMMIT"):
            return str(ast.literal_eval(line.split("=", 1)[1].strip()))
    raise ValueError("Template parameter cell has no GIT_COMMIT")


def _run_ids(backbone: str, seeds: tuple[int, ...]) -> list[str]:
    return [
        f"{dataset_id}__{backbone}__db__seed_{seed}"
        for seed in seeds
        for dataset_id in DATASETS
    ]


def generate(
    template_path: Path,
    output_dir: Path,
    backbones: tuple[str, ...] = DEFAULT_BACKBONES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    git_commit: str | None = None,
) -> list[Path]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    commit = git_commit or _template_commit(template)
    generated: list[Path] = []

    for backbone in backbones:
        notebook = json.loads(json.dumps(template, ensure_ascii=False))
        notebook.get("metadata", {}).pop("papermill", None)
        # Omit Kaggle-owned metadata entirely. An absent block leaves the
        # existing notebook's title, inputs, accelerator and internet settings
        # under Kaggle's control when this file is imported as a new version.
        # Setting dataSources=[] would actively clear the current attachments.
        notebook.get("metadata", {}).pop("kaggle", None)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        parameter_cell = _parameter_cell(notebook)
        parameter_cell["source"] = [
            f"BACKBONE = {backbone!r}\n",
            f"SEEDS = {list(seeds)!r}\n",
            f"EXPECTED_RUN_IDS = {_run_ids(backbone, seeds)!r}\n",
            f"GIT_COMMIT = {commit!r}",
        ]

        cells = notebook["cells"]
        insert_at = next(
            (
                index
                for index, cell in enumerate(cells)
                if "## 5. Danh sách file xuất" in "".join(cell.get("source", []))
                or "## 5. Danh sÃ¡ch file xuáº¥t" in "".join(cell.get("source", []))
            ),
            len(cells),
        )
        if insert_at < len(cells):
            cells[insert_at]["source"] = ["## 6. Danh sách file xuất"]
        cells[insert_at:insert_at] = [REPORT_MARKDOWN_CELL, REPORT_CODE_CELL]

        model_dir = output_dir / backbone
        model_dir.mkdir(parents=True, exist_ok=True)
        destination = model_dir / f"bundle_{backbone}_classification_reports.ipynb"
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        generated.append(destination)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_classification_report_notebooks"),
    )
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    for destination in generate(
        template_path=args.template,
        output_dir=args.output_dir,
        backbones=tuple(args.backbones),
        seeds=tuple(args.seeds),
        git_commit=args.git_commit,
    ):
        print(destination)


if __name__ == "__main__":
    main()
