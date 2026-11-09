"""Research dashboard for exploring visual evidence across three pipelines.

Run with: python -m app.demo_gradio
"""
from __future__ import annotations

import html
import io
from collections import Counter

import gradio as gr

from app.inference import MODEL_ORDER, InferenceService
from app.rule_localization import base64_to_image

service = InferenceService(params_path="params.yaml")

MODEL_META = {
    "baseline": {
        "number": "01",
        "title": "Baseline CNN",
        "eyebrow": "Phân tích dữ liệu gốc",
        "class_name": "baseline-card",
    },
    "stage5": {
        "number": "02",
        "title": "Theo luật",
        "eyebrow": "Bằng chứng theo luật",
        "class_name": "stage5-card",
    },
    "stage5_bayesian": {
        "number": "03",
        "title": "Đa tập luật",
        "eyebrow": "Bằng chứng tổng hợp",
        "class_name": "multi-rule-card",
    },
}


def _status_strip() -> str:
    items = []
    for status in service.model_statuses():
        state_class = "ready" if status["available"] else "missing"
        state_label = "Sẵn sàng" if status["available"] else "Thiếu artifact"
        items.append(
            f'<span class="status-chip {state_class}"><span class="status-dot"></span>'
            f'{html.escape(status["model_label"])} · {state_label}</span>'
        )
    return '<div class="status-strip">' + "".join(items) + "</div>"


def _model_header(model_key: str) -> str:
    meta = MODEL_META[model_key]
    return f"""
    <div class="model-heading">
      <span class="model-number">{meta['number']}</span>
      <div>
        <span class="model-eyebrow">{meta['eyebrow']}</span>
        <h2>{meta['title']}</h2>
      </div>
    </div>
    """


def _format_result(result: dict) -> str:
    if not result["available"]:
        expected = html.escape(result.get("expected_checkpoint", ""))
        return (
            "### Chưa có kết quả cho nhánh phân tích này\n\n"
            f"{result.get('error', 'Thiếu artifact')}\n\n"
            f"Checkpoint mong đợi: `{expected}`\n\n"
            "Tạo artifact bằng pipeline tổng hợp đa tập luật đã cấu hình trong `dvc.yaml`."
        )

    rules = result["matched_rules"]
    region_counts = {
        region["rule_number"]: len(region["boxes"])
        for region in result["rule_regions"]
    }
    if rules:
        rule_lines = []
        for index, rule in enumerate(rules, start=1):
            rule_lines.append(
                f"**Luật {index}** · khớp `{rule['satisfaction_score']:.2f}` · "
                f"tin cậy `{rule['rule_confidence']:.2f}` · "
                f"{region_counts.get(index, 0)} vùng\n\n`{rule['rule']}`"
            )
        rule_markdown = "\n\n---\n\n".join(rule_lines)
    elif result["model_key"] == "baseline":
        rule_markdown = "Dữ liệu ảnh được phân tích trực tiếp, không bổ sung ràng buộc luật."
    else:
        rule_markdown = "Không có luật hỗ trợ lớp dự đoán vượt ngưỡng khớp `0.50`."

    return (
        f"## {result['predicted_class']}\n\n"
        f"<span class='metric-pill'>Confidence {result['confidence'] * 100:.1f}%</span> "
        f"<span class='metric-pill'>Predict {result['inference_ms']:.0f} ms</span> "
        f"<span class='metric-pill'>Explain {result['explanation_ms']:.0f} ms</span>\n\n"
        f"{result['model_description']}\n\n"
        f"### Bằng chứng\n\n{rule_markdown}"
    )


def _synthesis_html(results: list[dict]) -> str:
    available = [result for result in results if result["available"]]
    predictions = [result["predicted_class"] for result in available]
    if not predictions:
        return '<div class="synthesis neutral"><strong>Chưa có kết quả</strong></div>'

    counts = Counter(predictions)
    winner, votes = counts.most_common(1)[0]
    unanimous = votes == len(predictions) and len(predictions) > 1
    state_class = "agree" if unanimous else "mixed"
    headline = "Các nhánh phân tích nhất quán" if unanimous else "Phát hiện khác biệt trong dữ liệu"
    detail = " · ".join(
        f"{MODEL_META[result['model_key']]['title']}: {html.escape(result['predicted_class'])}"
        for result in available
    )
    return (
        f'<div class="synthesis {state_class}">'
        f'<span class="synthesis-kicker">Tổng hợp {len(available)}/3 nhánh phân tích</span>'
        f'<strong>{headline}: {html.escape(winner)}</strong><span>{detail}</span></div>'
    )


def classify(image):
    if image is None:
        raise gr.Error("Vui lòng tải một mẫu ảnh trước khi phân tích.")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    results = service.compare(buffer.getvalue())["models"]
    by_key = {result["model_key"]: result for result in results}

    outputs = [_synthesis_html(results)]
    for key in MODEL_ORDER:
        result = by_key[key]
        if result["available"]:
            annotated = base64_to_image(result["annotated_image_base64"])
            scores = {
                item["class"]: item["confidence"] for item in result["top5"]
            }
        else:
            annotated = None
            scores = {}
        outputs.extend([annotated, scores, _format_result(result)])
    return tuple(outputs)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@500;600;700;800&family=Noto+Sans:wght@400;500;600&display=swap');
:root {
  --ink: #172554; --muted: #475569; --surface: rgba(255,255,255,.92);
  --baseline: #1D4ED8; --stage5: #7C3AED; --multi-rule: #C2410C;
  --teal: #0F766E; --line: rgba(99,102,241,.18);
}
body { background: #EEF2FF; }
.gradio-container {
  max-width: 1480px !important; margin: 0 auto !important; padding: 0 18px 40px !important;
  color: var(--ink); font-family: 'Noto Sans', sans-serif;
  background:
    radial-gradient(circle at 8% 18%, rgba(34,211,238,.24), transparent 25rem),
    radial-gradient(circle at 92% 34%, rgba(249,115,22,.19), transparent 27rem),
    linear-gradient(180deg, #EEF2FF 0%, #F8FAFC 52%, #ECFEFF 100%);
}
.hero-shell {
  position: relative; overflow: hidden; margin: 18px 0 14px; padding: clamp(28px,5vw,56px);
  border-radius: 28px; color: white;
  background: linear-gradient(125deg, #172554 0%, #4338CA 50%, #0F766E 100%);
  box-shadow: 0 24px 70px rgba(30,41,109,.28);
}
.hero-shell::after {
  content: ''; position: absolute; width: 280px; height: 280px; right: -70px; top: -105px;
  border: 55px solid rgba(255,255,255,.1); border-radius: 50%;
}
.hero-kicker { display: block; color: #A5F3FC; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .78rem; }
.hero-shell h1 { position: relative; z-index: 1; max-width: 900px; margin: 10px 0 12px; color: white; font: 800 clamp(2rem,5vw,4rem)/1.08 'Be Vietnam Pro', sans-serif; }
.hero-shell p { position: relative; z-index: 1; max-width: 780px; margin: 0; color: #E0E7FF; font-size: clamp(1rem,2vw,1.18rem); line-height: 1.65; }
.status-strip { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 2px 22px; }
.status-chip { display: inline-flex; align-items: center; gap: 8px; min-height: 34px; padding: 6px 12px; border-radius: 999px; font-size: .82rem; font-weight: 700; background: white; border: 1px solid #CBD5E1; }
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: currentColor; box-shadow: 0 0 0 4px currentColor; opacity: .72; }
.status-chip.ready { color: #047857; border-color: #6EE7B7; background: #ECFDF5; }
.status-chip.missing { color: #B45309; border-color: #FCD34D; background: #FFFBEB; }
.upload-panel, .model-card {
  background: var(--surface); border: 1px solid rgba(255,255,255,.85); border-radius: 22px;
  box-shadow: 0 16px 48px rgba(30,64,175,.10); backdrop-filter: blur(14px);
}
.upload-panel { padding: 12px; margin-bottom: 16px; border-left: 5px solid #06B6D4; }
.upload-copy h2 { color: var(--ink); font: 700 clamp(1.25rem,3vw,1.75rem) 'Be Vietnam Pro', sans-serif; margin: 4px 0 8px; }
.upload-copy p { color: var(--muted); line-height: 1.65; }
.primary-btn { min-height: 52px !important; cursor: pointer; font-weight: 800 !important; border: 0 !important; color: white !important; background: linear-gradient(100deg, #2563EB, #7C3AED, #DB2777) !important; box-shadow: 0 10px 28px rgba(124,58,237,.28) !important; transition: filter 200ms ease, box-shadow 200ms ease !important; }
.primary-btn:hover { filter: brightness(1.08); box-shadow: 0 13px 34px rgba(124,58,237,.36) !important; }
.primary-btn:focus-visible { outline: 3px solid #172554 !important; outline-offset: 3px; }
.synthesis { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin: 8px 0 20px; padding: 18px 22px; border-radius: 18px; border: 1px solid; }
.synthesis-kicker { width: 100%; font-size: .75rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
.synthesis strong { font: 700 1.08rem 'Be Vietnam Pro', sans-serif; }
.synthesis span:last-child { color: #334155; }
.synthesis.agree { background: #ECFDF5; border-color: #6EE7B7; color: #065F46; }
.synthesis.mixed { background: #FFF7ED; border-color: #FDBA74; color: #9A3412; }
.synthesis.neutral { background: #F1F5F9; border-color: #CBD5E1; }
.model-card { position: relative; overflow: hidden; min-width: 0; padding: 10px; transition: transform 220ms ease, box-shadow 220ms ease; }
.model-card:hover { transform: translateY(-3px); box-shadow: 0 22px 56px rgba(30,64,175,.16); }
.model-card::before { content: ''; position: absolute; inset: 0 0 auto; height: 6px; background: var(--card-accent); }
.baseline-card { --card-accent: var(--baseline); }
.stage5-card { --card-accent: var(--stage5); }
.multi-rule-card { --card-accent: var(--multi-rule); }
.model-heading { display: flex; align-items: center; gap: 12px; padding: 16px 8px 8px; }
.model-number { display: grid; place-items: center; width: 44px; height: 44px; border-radius: 13px; color: white; background: var(--card-accent); font: 800 .92rem 'Be Vietnam Pro', sans-serif; box-shadow: 0 8px 20px color-mix(in srgb, var(--card-accent) 30%, transparent); }
.model-eyebrow { display: block; color: var(--card-accent); font-size: .72rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
.model-heading h2 { margin: 2px 0 0; color: var(--ink); font: 700 1.35rem 'Be Vietnam Pro', sans-serif; }
.result-image img { max-height: 360px; object-fit: contain; border-radius: 12px; }
.metric-pill { display: inline-block; margin: 2px 3px 2px 0; padding: 5px 9px; border-radius: 999px; background: #EEF2FF; color: #3730A3; border: 1px solid #C7D2FE; font-size: .78rem; font-weight: 800; }
.method-note { margin: 22px 2px 0; padding: 16px 18px; color: #334155; background: rgba(255,255,255,.74); border: 1px solid #BAE6FD; border-radius: 14px; font-size: .9rem; line-height: 1.6; }
@media (max-width: 768px) {
  .gradio-container { padding-inline: 10px !important; }
  .hero-shell { border-radius: 20px; padding: 28px 22px; }
  .model-card { margin-bottom: 12px; }
  .synthesis { align-items: flex-start; flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
  .model-card:hover { transform: none; }
}
"""

with gr.Blocks(title="Khám phá bằng chứng ẩn trong hình ảnh") as demo:
    gr.HTML(
        """
        <header class="hero-shell">
          <span class="hero-kicker">Không gian nghiên cứu dữ liệu thị giác</span>
          <h1>Khám phá bằng chứng ẩn trong hình ảnh</h1>
          <p>Phân tích kết quả dự đoán, độ tin cậy và các vùng dữ liệu quan trọng
          góp phần hình thành quyết định.</p>
        </header>
        """
    )
    gr.HTML(_status_strip())

    with gr.Row(equal_height=False, elem_classes="upload-panel"):
        with gr.Column(scale=7):
            input_image = gr.Image(
                type="pil", label="Ảnh đầu vào dùng chung", height=360
            )
        with gr.Column(scale=5, elem_classes="upload-copy"):
            gr.HTML(
                """
                <h2>Thiết lập mẫu nghiên cứu</h2>
                <p>Một mẫu ảnh được chuẩn hóa thành center-crop 224 × 224 và đưa qua
                ba nhánh phân tích để đối chiếu dự đoán cùng bằng chứng không gian.</p>
                """
            )
            analyze = gr.Button(
                "Phân tích dữ liệu ảnh", variant="primary", elem_classes="primary-btn"
            )

    synthesis = gr.HTML(
        '<div class="synthesis neutral"><strong>Tải một mẫu ảnh để bắt đầu phân tích</strong></div>'
    )

    card_outputs = []
    with gr.Row(equal_height=False):
        for model_key in MODEL_ORDER:
            meta = MODEL_META[model_key]
            with gr.Column(
                scale=1,
                min_width=300,
                elem_classes=["model-card", meta["class_name"]],
            ):
                gr.HTML(_model_header(model_key))
                output_image = gr.Image(
                    type="pil",
                    label=f"Bản đồ bằng chứng · {meta['title']}",
                    interactive=False,
                    height=340,
                    elem_classes="result-image",
                )
                scores = gr.Label(
                    num_top_classes=5, label=f"Phân bố dự đoán · {meta['title']}"
                )
                explanation = gr.Markdown("Chưa có kết quả.")
                card_outputs.extend([output_image, scores, explanation])

    gr.HTML(
        """
        <p class="method-note"><strong>Ghi chú phương pháp:</strong> Nhánh Baseline
        phản ánh tín hiệu phân loại trực tiếp từ ảnh. Hai nhánh neuro-symbolic định vị
        vùng dữ liệu quan trọng bằng rule-aware occlusion: lần lượt che từng vùng và
        đo mức suy giảm độ thỏa luật. Bounding box thể hiện bằng chứng đóng góp ước
        lượng, không phải nhãn object-detection ground truth.</p>
        """
    )

    analyze.click(
        fn=classify,
        inputs=input_image,
        outputs=[synthesis, *card_outputs],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=CSS,
    )
