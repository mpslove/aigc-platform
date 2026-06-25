"""Gradio Web UI for the AIGC platform.

Provides a browser-based interface for:
  - Selecting templates or entering topic
  - Running the full pipeline
  - Viewing results and quality scores
  - Downloading generated videos

Launch with: python src/webui/app.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from src.agent.orchestrator import VideoAgent
from src.pipeline.script_writer import ScriptWriter
from src.pipeline.schema import VideoProject


# Global agent
_agent = VideoAgent(
    asset_dir=os.environ.get("AIGC_ASSETS_DIR", "./assets"),
    output_dir=os.environ.get("AIGC_OUTPUT_DIR", "./output"),
)
_script_writer = ScriptWriter()


def list_templates() -> list[str]:
    """Get available templates for the dropdown."""
    return [""] + _script_writer.list_templates()


def create_project(template: str, topic: str) -> str:
    """Create project from template or topic."""
    try:
        if template:
            _agent.from_template(template)
        elif topic:
            _agent.from_topic(topic)
        else:
            return "Please select a template or enter a topic."

        p = _agent.current_project
        parts = [
            f"✅ Project: {p.title}",
            f"   Scenes: {len(p.scenes)}",
            f"   Duration: ~{p.total_duration():.0f}s",
        ]
        for s in p.scenes:
            parts.append(f"   [{s.id}] {s.scene_type.value}: {s.prompt[:50]}...")
        return "\n".join(parts)
    except Exception as e:
        return f"❌ Error: {e}"


def run_pipeline(template: str, topic: str) -> tuple:
    """Run the full pipeline and return video + results."""
    try:
        if template:
            _agent.from_template(template)
        elif topic:
            _agent.from_topic(topic)
        else:
            return None, "❌ No template or topic specified."

        # Generate + Compose
        _agent.generate()
        video_path = _agent.compose()

        # Evaluate
        eval_result = _agent.evaluate(video_path)

        # Format report
        report = [
            f"✅ Pipeline Complete",
            f"   Video: {video_path}",
            f"   Duration: {eval_result['duration_s']:.1f}s",
            f"   Resolution: {eval_result['resolution'][0]}x{eval_result['resolution'][1]}",
            f"   Size: {eval_result['file_size_kb']:.0f}KB",
            f"   Quality Score: {eval_result['quality_score']:.0f}/100",
        ]
        report_str = "\n".join(report)

        return video_path, report_str

    except Exception as e:
        return None, f"❌ Pipeline failed: {e}"


def evaluate_video(video_path: str) -> str:
    """Evaluate a video file."""
    if not video_path or not os.path.exists(video_path):
        return "❌ No video file found."
    try:
        result = _agent.evaluate(video_path)
        lines = [
            f"📊 Quality Score: {result['quality_score']:.0f}/100",
            f"   Duration: {result['duration_s']:.1f}s",
            f"   Resolution: {result['resolution'][0]}x{result['resolution'][1]}",
            f"   Size: {result['file_size_kb']:.0f}KB",
            f"   Audio: {'Yes' if result['has_audio'] else 'No'}",
        ]
        if result['errors']:
            lines.append(f"   ⚠ Errors: {', '.join(result['errors'])}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Evaluation error: {e}"


def clear_state() -> str:
    """Reset agent state."""
    import src.agent.orchestrator as orch
    orch._agent = None
    global _agent
    _agent = VideoAgent(
        asset_dir=os.environ.get("AIGC_ASSETS_DIR", "./assets"),
        output_dir=os.environ.get("AIGC_OUTPUT_DIR", "./output"),
    )
    return "✅ State cleared. Ready for new project."


# ── Build UI ─────────────────────────────────────────────────────────

with gr.Blocks(title="AIGC Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 AIGC Video Studio")
    gr.Markdown("Multi-modal AIGC production platform — generate videos from text descriptions.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Project Setup")

            template_dd = gr.Dropdown(
                choices=_script_writer.list_templates(),
                label="Template",
                info="Select a pre-built template",
                interactive=True,
                allow_custom_value=False,
            )
            topic_input = gr.Textbox(
                label="Or describe your topic",
                placeholder="e.g. 'nursing care ad', 'travel destination', 'product launch'",
            )
            with gr.Row():
                create_btn = gr.Button("📝 Create Project", variant="secondary")
                run_btn = gr.Button("🚀 Full Pipeline", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 📄 Project & Results")

            project_output = gr.Textbox(label="Project Info", lines=8)
            result_video = gr.Video(label="Generated Video", interactive=False)
            eval_output = gr.Textbox(label="Quality Evaluation", lines=6)

    with gr.Row():
        eval_btn = gr.Button("📊 Evaluate Video", variant="secondary")
        clear_btn = gr.Button("🔄 Clear State", variant="stop", size="sm")

    # Event handlers
    create_btn.click(
        fn=create_project,
        inputs=[template_dd, topic_input],
        outputs=[project_output],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[template_dd, topic_input],
        outputs=[result_video, project_output],
    ).then(
        fn=evaluate_video,
        inputs=[result_video],
        outputs=[eval_output],
    )

    eval_btn.click(
        fn=evaluate_video,
        inputs=[result_video],
        outputs=[eval_output],
    )

    clear_btn.click(
        fn=clear_state,
        inputs=[],
        outputs=[project_output],
    )

    # Template auto-fill topic
    def on_template_change(template):
        if template:
            return f"Using template: {template}"
        return ""

    template_dd.change(
        fn=on_template_change,
        inputs=[template_dd],
        outputs=[topic_input],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
