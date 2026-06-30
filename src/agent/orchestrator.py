"""Orchestrator — ReAct-style agent loop for automated video production.

The VideoAgent uses a genuine Reason-Act-Observe (ReAct) loop:
  1. Observe user request and current state
  2. Think — LLM decides which tool to call and why
  3. Act — execute the chosen tool
  4. Observe — feed the tool result back to LLM
  5. Repeat until task complete or max steps reached

Key capabilities over the old hardcoded pipeline:
  - LLM-driven tool selection (not a fixed sequence)
  - Error recovery: LLM can retry or switch strategy on failure
  - Fallback routing: Agnes fail → try ComfyUI or mock
  - State tracking across the full conversation
  - Transparent reasoning chain (every step logged)
"""

import os
import json
import time
import threading
import logging
from typing import Optional

from src.agent.tools import Tool, register_core_tools
from src.pipeline.schema import VideoProject
from src.pipeline.script_writer import ScriptWriter
from src.pipeline.generator import AssetProducer
from src.pipeline.composer import Composer
from src.eval.metrics import VideoQualityAnalyzer
from src.gateway.base import GeneratorBase
from src.gateway.factory import create_generator

logger = logging.getLogger(__name__)

# ── ReAct System Prompt ──────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an AIGC video production agent. You help users create videos from topics or templates.

You have these tools available:
{tools_description}

Follow the ReAct pattern:
1. **Thought**: Analyze the current state and decide what to do next
2. **Action**: Call exactly one tool
3. **Observation**: Review the tool result
4. Repeat until the task is complete

Rules:
- Always call create_project before generate_assets
- Always generate assets before composing video
- If a tool returns an error, reason about why and try an alternative
- After composing, evaluate the video quality
- When finished, respond with a summary of what was produced

Respond in JSON format:
{{"thought": "your reasoning", "tool": "tool_name", "args": {{...}}}}

When done, respond:
{{"thought": "task complete", "tool": "finish", "args": {{"summary": "..."}}}}
"""


class ToolRegistry(dict):
    """Dictionary-based tool registry with helpful errors."""

    def register(self, tool: Tool):
        self[tool.name] = tool

    def tool_descriptions(self) -> str:
        """Generate tool description string for system prompt."""
        lines = []
        for name, tool in self.items():
            params_str = ""
            if tool.parameters:
                props = tool.parameters.get("properties", {})
                required = tool.parameters.get("required", [])
                parts = []
                for pname, pdef in props.items():
                    req = "required" if pname in required else "optional"
                    parts.append(f"    - {pname} ({pdef.get('type', 'any')}, {req}): {pdef.get('description', '')}")
                params_str = "\n".join(parts)
            lines.append(f"- {name}: {tool.description}")
            if params_str:
                lines.append(params_str)
        return "\n".join(lines)

    def to_openai_functions(self) -> list[dict]:
        """Export all tools in OpenAI function-calling format."""
        return [tool.to_openai_function() for tool in self.values()]


class VideoAgent:
    """ReAct-style agent for AIGC video production.

    Two execution modes:
      1. **ReAct mode** (run): LLM decides each step — true agentic loop
      2. **Pipeline mode** (run_end_to_end): Fixed 4-phase sequence — backward compat
    """

    def __init__(self, generator: Optional[GeneratorBase] = None,
                 asset_dir: str = "./assets",
                 output_dir: str = "./output"):
        self.generator = generator
        self.asset_dir = asset_dir
        self.output_dir = output_dir
        self.script_writer = ScriptWriter()
        self.composer = Composer(work_dir=output_dir)
        self.quality_analyzer = VideoQualityAnalyzer()
        self.tools = ToolRegistry()

        # Execution state
        self.current_project: Optional[VideoProject] = None
        self.log: list[dict] = []
        self._lock = threading.Lock()

        # Register tools — pass self so production tools can delegate
        register_core_tools(self.tools, agent=self)

        # LLM client (lazy init)
        self._llm_client = None
        self._llm_model = os.environ.get("AIGC_AGENT_MODEL", "gpt-4o-mini")
        self._llm_base_url = os.environ.get("AIGC_AGENT_BASE_URL", None)
        self._llm_api_key = os.environ.get("AIGC_AGENT_API_KEY",
                                           os.environ.get("OPENAI_API_KEY", ""))

    # ── Logging ──────────────────────────────────────────────────

    def _log(self, action: str, detail: str, status: str = "ok"):
        self.log.append({
            "time": time.strftime("%H:%M:%S"),
            "action": action,
            "detail": detail,
            "status": status,
        })
        if len(self.log) > 200:
            self.log = self.log[-100:]

    # ── Generator init ──────────────────────────────────────────

    def ensure_generator(self):
        """Thread-safe lazy-init generator."""
        if self.generator is not None:
            return
        with self._lock:
            if self.generator is not None:
                return
            self.generator = create_generator(
                "agnes", output_dir=self.asset_dir,
            )
            self._log("init", f"Generator: {self.generator.get_provider_name()}")

    # ── LLM Client (OpenAI-compatible) ───────────────────────────

    def _get_llm_client(self):
        """Lazy-init LLM client (OpenAI-compatible API)."""
        if self._llm_client is not None:
            return self._llm_client

        try:
            from openai import OpenAI
            kwargs = {"api_key": self._llm_api_key}
            if self._llm_base_url:
                kwargs["base_url"] = self._llm_base_url
            self._llm_client = OpenAI(**kwargs)
            return self._llm_client
        except ImportError:
            logger.warning("openai package not installed. ReAct mode unavailable.")
            return None

    def _llm_decide(self, messages: list[dict]) -> dict:
        """Ask LLM to decide next action given conversation history.

        Returns: {"thought": str, "tool": str, "args": dict}
        """
        client = self._get_llm_client()
        if client is None:
            # Fallback: rule-based decision
            return self._rule_based_decide(messages)

        try:
            response = client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON response
            return self._parse_action(text)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Using rule-based fallback.")
            return self._rule_based_decide(messages)

    def _parse_action(self, text: str) -> dict:
        """Parse LLM response into structured action."""
        # Try JSON extraction
        text = text.strip()
        if text.startswith("```"):
            import re
            m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if m:
                text = m.group(1).strip()

        try:
            action = json.loads(text)
            if "tool" in action and "args" in action:
                return action
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to find JSON in text
        import re
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            try:
                action = json.loads(text[s:e+1])
                if "tool" in action:
                    return action
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: return as thought
        return {"thought": text, "tool": "finish", "args": {"summary": text}}

    def _rule_based_decide(self, messages: list[dict]) -> dict:
        """Rule-based fallback when LLM is unavailable.

        Implements the standard 4-phase workflow as discrete steps.
        """
        # Check what has been accomplished
        log_actions = [e.get("action", "") for e in self.log]
        has_project = "project" in log_actions
        has_generate = "generate" in log_actions
        has_compose = "compose" in log_actions
        has_evaluate = "evaluate" in log_actions

        if not has_project:
            return {
                "thought": "No project created yet. Creating from user request.",
                "tool": "create_project",
                "args": {"source": "template", "value": "nursing-ad"},
            }
        elif not has_generate:
            return {
                "thought": "Project created. Generating assets.",
                "tool": "generate_assets",
                "args": {},
            }
        elif not has_compose:
            return {
                "thought": "Assets generated. Composing video.",
                "tool": "compose_video",
                "args": {},
            }
        elif not has_evaluate:
            # Find the video path from compose result
            video_path = None
            for entry in reversed(self.log):
                if entry.get("action") == "compose":
                    detail = entry.get("detail", "")
                    # Extract path from "output/xxx.mp4 ..."
                    import re
                    m = re.search(r"(output[/\\][\w.-]+\.mp4)", detail)
                    if m:
                        video_path = m.group(1)
                    break
            return {
                "thought": "Video composed. Evaluating quality.",
                "tool": "evaluate_quality",
                "args": {"video_path": video_path or "output/output.mp4"},
            }
        else:
            return {
                "thought": "All phases complete.",
                "tool": "finish",
                "args": {"summary": "Video production pipeline completed successfully."},
            }

    # ── ReAct Loop ──────────────────────────────────────────────

    def run(self, user_request: str, max_steps: int = 12) -> dict:
        """Execute a ReAct loop: the LLM decides each step.

        Each iteration:
          1. Send conversation (system + user + tool results) to LLM
          2. LLM returns {thought, tool, args}
          3. Execute the chosen tool
          4. Append result to conversation

        Terminates when:
          - LLM calls "finish" tool
          - Max steps reached
          - Unrecoverable error

        Args:
            user_request: Natural language request (e.g. "create a nursing care ad")
            max_steps: Safety limit on ReAct iterations

        Returns:
            Dict with video_path, steps trace, and evaluation results
        """
        self.log = []
        start = time.time()

        # Build initial messages
        tools_desc = self.tools.tool_descriptions()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ]

        # Parse initial request to infer source/value
        source, value = self._parse_user_request(user_request)

        # ReAct loop
        step_trace = []
        for step in range(max_steps):
            if step == 0:
                # First step: use parsed request for create_project
                action = {
                    "thought": f"User wants: {user_request}. Creating project ({source}: {value}).",
                    "tool": "create_project",
                    "args": {"source": source, "value": value},
                }
            else:
                action = self._llm_decide(messages)

            thought = action.get("thought", "")
            tool_name = action.get("tool", "finish")
            tool_args = action.get("args", {})

            self._log("think", thought, "info")

            # Check termination
            if tool_name == "finish":
                self._log("finish", thought)
                break

            # Execute tool
            tool = self.tools.get(tool_name)
            if tool is None:
                error_msg = f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
                self._log("error", error_msg, "warn")
                observation = {"error": error_msg, "status": "failed"}
            else:
                self._log("action", f"{tool_name}({json.dumps(tool_args)[:100]})")
                try:
                    observation = tool(**tool_args)
                except Exception as e:
                    observation = {"error": str(e), "tool": tool_name}
                    self._log("action", f"Tool error: {e}", "warn")

            # Record step
            step_trace.append({
                "step": step + 1,
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "observation": str(observation)[:500],  # truncate for context
            })

            # Feed observation back to LLM
            messages.append({
                "role": "assistant",
                "content": json.dumps(action),
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {json.dumps(observation, default=str)[:800]}",
            })

        # Compile result
        elapsed = time.time() - start
        self._log("done", f"ReAct loop completed in {elapsed:.0f}s, {len(step_trace)} steps")

        # Collect outputs from trace
        result = {
            "elapsed_s": elapsed,
            "steps": len(step_trace),
            "step_trace": step_trace,
            "log": self.log,
            "project_summary": self.current_project.summary() if self.current_project else None,
        }

        # Extract final video path and eval from trace
        for st in step_trace:
            if st["tool"] == "compose_video":
                obs = st["observation"]
                try:
                    obs_dict = json.loads(obs)
                    if "video_path" in obs_dict:
                        result["video_path"] = obs_dict["video_path"]
                except (json.JSONDecodeError, ValueError):
                    pass
            if st["tool"] == "evaluate_quality":
                obs = st["observation"]
                try:
                    obs_dict = json.loads(obs)
                    result["evaluation"] = obs_dict
                except (json.JSONDecodeError, ValueError):
                    pass

        return result

    def _parse_user_request(self, request: str) -> tuple[str, str]:
        """Infer source and value from natural language request.

        Returns (source, value) — either ("template", name) or ("topic", description).
        """
        request_lower = request.lower()

        # Check for explicit template names
        templates = self.script_writer.list_templates()
        for tmpl in templates:
            if tmpl in request_lower:
                return "template", tmpl

        # Check for template-related keywords
        template_keywords = ["template", "模板"]
        if any(kw in request_lower for kw in template_keywords):
            # Try to extract template name
            for tmpl in templates:
                tmpl_name = tmpl.replace("-", " ")
                if tmpl_name in request_lower:
                    return "template", tmpl

        # Default: treat as topic
        return "topic", request.strip()[:200]

    # ── Phase methods (backward compat) ──────────────────────────

    def from_template(self, template_name: str) -> VideoProject:
        """Create project from named template."""
        project = self.script_writer.from_template(template_name)
        self.current_project = project
        self._log("project", f"From template '{template_name}': "
                             f"{len(project.scenes)} scenes, "
                             f"{project.total_duration():.0f}s")
        return project

    def from_topic(self, topic: str) -> VideoProject:
        """Create project from a topic description."""
        project = self.script_writer.from_prompt(topic)
        self.current_project = project
        self._log("project", f"From topic '{topic}': "
                             f"{len(project.scenes)} scenes, "
                             f"{project.total_duration():.0f}s")
        return project

    def from_json(self, json_path: str) -> VideoProject:
        """Load project from JSON file."""
        project = ScriptWriter.from_json(json_path)
        self.current_project = project
        self._log("project", f"Loaded from JSON: {len(project.scenes)} scenes")
        return project

    # ── Phase 2: Generate Assets ──────────────────────────────────

    def generate(self, project: Optional[VideoProject] = None,
                 skip_existing: bool = True) -> dict:
        """Generate all assets for the project."""
        proj = project or self.current_project
        if proj is None:
            raise ValueError("No project. Call from_template/topic/json first.")

        self.ensure_generator()
        producer = AssetProducer(
            self.generator,
            asset_dir=self.asset_dir,
            progress_file=os.path.join(self.asset_dir, ".progress.json")
            if skip_existing else None,
        )

        start = time.time()
        results = producer.produce_all(proj)
        elapsed = time.time() - start

        success = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)

        self._log("generate",
                  f"{success}/{len(results)} assets ({elapsed:.0f}s)")
        if failed:
            self._log("generate", f"{failed} failures", "warn")

        return {
            "total": len(results),
            "success": success,
            "failed": failed,
            "elapsed_s": elapsed,
            "results": {
                sid: {"success": r.success, "path": r.file_path,
                      "error": r.error}
                for sid, r in results.items()
            },
        }

    # ── Phase 3: Compose ──────────────────────────────────────────

    def compose(self, project: Optional[VideoProject] = None,
                output_filename: Optional[str] = None) -> str:
        """Compose final video from generated assets."""
        proj = project or self.current_project
        if proj is None:
            raise ValueError("No project set.")

        if output_filename:
            proj.output_filename = output_filename

        start = time.time()
        output_path = self.composer.compose_project(proj, self.asset_dir)
        elapsed = time.time() - start

        info = Composer.get_video_info(output_path)
        self._log("compose",
                  f"{output_path} ({info.get('duration_str', '?')}, "
                  f"{info.get('size_kb', 0):.0f}KB, {elapsed:.0f}s)")

        return output_path

    # ── Phase 4: Evaluate ──────────────────────────────────────────

    def evaluate(self, video_path: str) -> dict:
        """Evaluate quality of a generated video."""
        score = self.quality_analyzer.analyze(video_path)
        self._log("evaluate",
                  f"Quality: {score.quality_score:.0f}/100 "
                  f"({score.resolution[0]}x{score.resolution[1]}, "
                  f"{score.duration_s:.1f}s)")
        return {
            "path": score.path,
            "quality_score": score.quality_score,
            "duration_s": score.duration_s,
            "resolution": score.resolution,
            "file_size_kb": score.file_size_kb,
            "has_audio": score.has_audio,
            "errors": score.errors,
        }

    # ── Legacy pipeline mode (backward compat) ───────────────────

    def run_end_to_end(self, source: str = "template",
                       value: str = "nursing-ad",
                       output_filename: Optional[str] = None) -> dict:
        """Run the full workflow: create → generate → compose → evaluate.

        This is the legacy fixed-pipeline mode. For the ReAct loop
        (LLM-driven decisions), use run() instead.
        """
        self.log = []
        start = time.time()

        # Phase 1: Project
        if source == "template":
            project = self.from_template(value)
        elif source == "topic":
            project = self.from_topic(value)
        elif source == "json":
            project = self.from_json(value)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Phase 2: Generate
        gen_result = self.generate(project)
        if gen_result["failed"] > 0:
            self._log("end_to_end", f"{gen_result['failed']} failed assets",
                      "warn")

        # Phase 3: Compose
        video_path = self.compose(project, output_filename)

        # Phase 4: Evaluate
        eval_result = self.evaluate(video_path)

        total_elapsed = time.time() - start
        self._log("end_to_end",
                  f"Done in {total_elapsed:.0f}s → {video_path}")

        return {
            "video_path": video_path,
            "elapsed_s": total_elapsed,
            "generation": gen_result,
            "evaluation": eval_result,
            "project_summary": self.current_project.summary(),
            "log": self.log,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the last run."""
        lines = ["=== AIGC Video Agent Summary ===\n"]
        if self.current_project:
            s = self.current_project.summary()
            lines.append(f"Project: {s['title']}")
            lines.append(f"Scenes: {s['scenes']} ({s['duration_s']:.0f}s)")
        lines.append("")
        for entry in self.log:
            icon = {"ok": "✓", "warn": "⚠", "info": "→"}.get(entry["status"], "?")
            lines.append(f"  {icon} [{entry['time']}] {entry['action']}: "
                         f"{entry['detail']}")
        return "\n".join(lines)
