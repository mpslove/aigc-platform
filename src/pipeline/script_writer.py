"""ScriptWriter — generates video scripts from a topic using AI.

Transforms a topic description (e.g. "nursing care ad") into a structured
VideoProject with scenes, prompts, and text overlays.

This is the bridge between user intent and pipeline execution.
"""

import json
import os
from typing import Optional

from .schema import (
    Scene, SceneType, GeneratorType, TextOverlay, VideoProject, Transition,
)

# Template-based script generation — no LLM dependency
SCRIPT_TEMPLATES = {
    "nursing-ad": {
        "title": "Nursing Care Advertisement",
        "topic": "护理服务宣传",
        "scenes": [
            {
                "id": "title", "type": "title", "duration": 3.0,
                "prompt": "Dark blue gradient background with soft healing light, clean medical style, elegant and calm",
                "text": ["Nursing Care", "Compassionate & Professional"],
            },
            {
                "id": "care", "type": "video", "duration": 4.0,
                "prompt": "A female nurse in white uniform gently holding an elderly patient hand, both sitting by a sunny window, the patient hand slowly moves, warm golden light",
            },
            {
                "id": "vitals", "type": "video", "duration": 4.0,
                "prompt": "A female nurse in white uniform standing in front of a hospital monitor, she reaches and presses a button, blue LED light on face, bright modern hospital room",
            },
            {
                "id": "walk", "type": "video", "duration": 4.0,
                "prompt": "An elderly man in hospital gown slowly walking forward with a nurse supporting his arm, both walking together across bright room, sunlight from right side",
            },
            {
                "id": "room", "type": "video", "duration": 4.0,
                "prompt": "Slow pan across a clean hospital room, a bed with white sheets, flowers on bedside table, warm afternoon light through window curtains",
                "text": ["Healing Environment"],
            },
            {
                "id": "end", "type": "end_card", "duration": 3.0,
                "prompt": "Dark blue background with a glowing medical cross symbol in center, peaceful healing atmosphere, minimalist",
                "text": ["Your Health, Our Mission"],
            },
        ],
    },
    "product-ad": {
        "title": "Product Advertisement",
        "topic": "产品宣传",
        "scenes": [
            {
                "id": "title", "type": "title", "duration": 3.0,
                "prompt": "Clean white gradient background, minimalist product showcase style, elegant",
                "text": ["New Product", "Innovation for Life"],
            },
            {
                "id": "hero", "type": "video", "duration": 4.0,
                "prompt": "A sleek modern product on a rotating display stand, clean white background, studio lighting, smooth rotation",
            },
            {
                "id": "feature", "type": "video", "duration": 4.0,
                "prompt": "Close-up shot of hands using a modern device, finger pressing buttons, blue LED indicators, shallow depth of field",
            },
            {
                "id": "lifestyle", "type": "video", "duration": 5.0,
                "prompt": "A person using a device in a bright modern living room, smiling, natural lighting, warm atmosphere",
            },
            {
                "id": "end", "type": "end_card", "duration": 3.0,
                "prompt": "Dark elegant background with product silhouette, gold accents",
                "text": ["Available Now"],
            },
        ],
    },
    "travel-ad": {
        "title": "Travel Destination",
        "topic": "旅游宣传",
        "scenes": [
            {
                "id": "title", "type": "title", "duration": 3.0,
                "prompt": "Sunrise over mountain peaks, golden clouds, warm orange and purple sky, epic landscape",
                "text": ["Discover Paradise"],
            },
            {
                "id": "scene1", "type": "video", "duration": 5.0,
                "prompt": "Slow aerial shot over a tropical beach, crystal clear turquoise water, white sand, palm trees swaying in wind",
            },
            {
                "id": "scene2", "type": "video", "duration": 5.0,
                "prompt": "People walking along a boardwalk at sunset, silhouettes against golden sky, peaceful atmosphere",
            },
            {
                "id": "scene3", "type": "video", "duration": 5.0,
                "prompt": "A person snorkeling in clear water, colorful coral reef visible below, sunlight rays through water surface",
            },
            {
                "id": "end", "type": "end_card", "duration": 3.0,
                "prompt": "Aerial view of a tropical island from above, deep blue ocean, white sand edges, sunset lighting",
                "text": ["Book Your Adventure"],
            },
        ],
    },
}


class ScriptWriter:
    """Generate video scripts from topic descriptions."""

    def __init__(self):
        self.templates = SCRIPT_TEMPLATES

    def list_templates(self) -> list[str]:
        return list(self.templates.keys())

    def from_template(self, template_name: str) -> VideoProject:
        """Build VideoProject from a named template."""
        data = self.templates.get(template_name)
        if not data:
            raise ValueError(f"Unknown template: {template_name}. "
                             f"Available: {list(self.templates.keys())}")

        scenes = []
        for sd in data["scenes"]:
            overlays = []
            texts = sd.get("text", [])
            for i, t in enumerate(texts):
                y_offset = -20 if i == 0 else 30
                overlays.append(TextOverlay(
                    text=t,
                    font_size=42 if i == 0 else 28,
                    y=f"(h-text_h)/2+{y_offset}",
                    box_border_w=10 if i == 0 else 6,
                ))

            scenes.append(Scene(
                id=sd["id"],
                scene_type=SceneType(sd["type"]),
                prompt=sd["prompt"],
                duration=sd.get("duration", 5.0),
                generator=(GeneratorType.AGNES_IMAGE
                           if sd["type"] in ("title", "end_card", "image")
                           else GeneratorType.AGNES_VIDEO),
                text_overlays=overlays,
            ))

        return VideoProject(
            title=data["title"],
            topic=data["topic"],
            scenes=scenes,
            script_source="template",
        )

    def from_prompt(self, topic: str, num_scenes: int = 5) -> VideoProject:
        """Generate script from a natural language topic.

        Current: returns a generic template based on keyword matching.
        Future: will use LLM to generate structured scripts.
        """
        topic_lower = topic.lower()

        # Keyword-based template matching
        if any(k in topic_lower for k in ("nurse", "护理", "medical", "医院", "care")):
            return self.from_template("nursing-ad")
        elif any(k in topic_lower for k in ("product", "产品", "device", "gadget")):
            return self.from_template("product-ad")
        elif any(k in topic_lower for k in ("travel", "旅游", "beach", "海", "island")):
            return self.from_template("travel-ad")

        # Generic: fallback to nursing-ad example
        return self.from_template("nursing-ad")

    def to_dict(self, project: VideoProject) -> dict:
        """Serialize project to dict (for JSON export)."""
        return {
            "title": project.title,
            "topic": project.topic,
            "scenes": [
                {
                    "id": s.id,
                    "type": s.scene_type.value,
                    "prompt": s.prompt,
                    "duration": s.duration,
                    "transition": s.transition.value,
                    "text_overlays": [
                        {"text": t.text, "font_size": t.font_size}
                        for t in s.text_overlays
                    ],
                }
                for s in project.scenes
            ],
        }

    def save_script(self, project: VideoProject, path: str):
        """Save project script to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(project), f, indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(path: str) -> VideoProject:
        """Load project from JSON file."""
        with open(path) as f:
            data = json.load(f)
        scenes = []
        for sd in data["scenes"]:
            overlays = [
                TextOverlay(text=t["text"], font_size=t.get("font_size", 36))
                for t in sd.get("text_overlays", [])
            ]
            scenes.append(Scene(
                id=sd["id"],
                scene_type=SceneType(sd["type"]),
                prompt=sd["prompt"],
                duration=sd.get("duration", 5.0),
                generator=(GeneratorType.AGNES_IMAGE
                           if sd["type"] in ("title", "end_card", "image")
                           else GeneratorType.AGNES_VIDEO),
                text_overlays=overlays,
                transition=Transition(sd.get("transition", "cut")),
            ))
        return VideoProject(
            title=data["title"],
            topic=data.get("topic", ""),
            scenes=scenes,
            script_source="json",
        )
