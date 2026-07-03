"""Demo UI trực quan bằng Gradio — dùng để giới thiệu/thuyết trình, không thay
thế cho FastAPI service (dùng cho tích hợp hệ thống).

Chạy: python -m app.demo_gradio
"""
import gradio as gr

from app.inference import InferenceService

service = InferenceService(params_path="params.yaml")


def classify(image) -> tuple:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    result = service.predict(buf.getvalue())

    label_scores = {item["class"]: item["confidence"] for item in result["top5"]}

    if result["matched_rules"]:
        rules_md = "\n".join(
            f"- **{r['rule']}**  \n  (độ khớp: {r['satisfaction_score']:.2f}, độ tin cậy luật: {r['rule_confidence']:.2f})"
            for r in result["matched_rules"]
        )
    else:
        rules_md = "_Không có luật nào khớp rõ ràng với ảnh này._"

    summary = f"### Dự đoán: **{result['predicted_class']}** ({result['confidence']*100:.1f}%)\n\n#### Luật hỗ trợ quyết định:\n{rules_md}"
    return label_scores, summary


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Tải ảnh văn hoá Việt Nam"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top-5 dự đoán"),
        gr.Markdown(label="Giải thích bằng luật"),
    ],
    title="Neuro-Symbolic Vietnamese Cultural Classifier",
    description="Phân loại ảnh và giải thích quyết định bằng các luật được trích từ Random Forest và chọn lọc bởi GFlowNet.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
