"""OmniEcho processing component."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

from omniecho_audio.api_client import (
    call_llm_with_system,
    APIPermanentError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    """OmniEcho processing component."""

    if not text:
        raise ValueError("Empty LLM response")
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    first_obj = stripped.find("{")
    last_obj = stripped.rfind("}")
    first_arr = stripped.find("[")
    last_arr = stripped.rfind("]")

    candidates: list[str] = []
    if first_obj != -1 and last_obj > first_obj:
        candidates.append(stripped[first_obj : last_obj + 1])
    if first_arr != -1 and last_arr > first_arr:
        candidates.append(stripped[first_arr : last_arr + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No JSON found in response. Head: {stripped[:300]}")


def _parse_llm_text(data: Any) -> str:
    """OmniEcho processing component."""
    if isinstance(data, str):
        return data
    raise RuntimeError(f"Unrecognized LLM response shape: {str(data)[:300]}")


# ---------------------------------------------------------------------------
# 3. System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a spatial-audio timeline director. Review each source's onset_t/offset_t so timing follows physical behavior and narrative logic. Write explanations in English.

Timing principles:
1. Continuous ambient sounds must have onset_t=0.0 and offset_t=duration:
      - Flowing water: streams, rivers, waterfalls, and pipes.
      - Wind: breezes, strong wind, and wind through pine trees.
      - Rain: light rain, heavy rain, and thunderstorms.
      - Background ambience: HVAC, ventilation, traffic, and insects.
      - Any source labeled ambient or background.
      Breaking this rule sounds unnatural, such as water or wind stopping abruptly.

2. Natural event timing:
      - Footsteps need a plausible arrival, continuation, and departure, typically 2-5 seconds.
      - Knocks/impacts may be short (0.5-2 seconds), but not just 0.1 seconds.
      - Animal calls often repeat over 1-4 seconds.
      - Periodic sounds such as pendulums or dripping usually span most or all of the scene.

3. Overlap and temporal layers:
      - The scene must not begin or end in complete silence.
      - At least one source must have onset_t=0.
      - Continuous ambience prevents silent gaps.
      - Stagger event onsets to create temporal layers.

4. Narrative logic:
      - Source ordering must agree with the scene's story.
      - If someone enters the scene, footsteps should begin at the described entrance time.

Return JSON in this format:
{
  "decisions": [
    {
      "object_id": "source_001",
      "onset_t": 0.0,
      "offset_t": 6.0,
      "reason": "The stream is continuous ambience and must span the full scene."
    }
  ]
}

Constraints:
- decisions must have one entry per input object, with matching object_id values.
- onset_t >=0 and <duration; offset_t >onset_t and <=duration.
- Return one JSON object only, with English text.
"""


# ---------------------------------------------------------------------------
# 4. DurationAgent
# ---------------------------------------------------------------------------


class DurationAgent:
    """OmniEcho processing component."""

    LLM_TIMEOUT = 60.0
    MAX_HTTP_RETRIES = 3

    def __init__(
        self,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        
        pass

    # ---------------------------------------------------------------- LLM I/O
    def close(self) -> None:
        """OmniEcho processing component."""
        pass

    def __enter__(self) -> "DurationAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """OmniEcho processing component."""
        try:
            return call_llm_with_system(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=8192,
                timeout=self.LLM_TIMEOUT,
                max_retries=self.MAX_HTTP_RETRIES,
            )
        except APIPermanentError as e:
            raise RuntimeError(
                f"DurationAgent permanent LLM error: {e}"
            ) from e

    # ----------------------------------------------------------- main entry
    def decide_durations(
        self,
        scene_description: str,
        objects: list[dict],
        scene_duration: float,
    ) -> dict:
        """OmniEcho processing component."""

        if not objects:
            return {"decisions": [], "raw_response": ""}

        prompt = self._build_user_prompt(
            scene_description=scene_description,
            objects=objects,
            scene_duration=scene_duration,
        )

        raw_response = self._call_llm(_SYSTEM_PROMPT, prompt)

        decisions = self._parse_response(raw_response, objects, scene_duration)
        return {
            "decisions": decisions,
            "raw_response": raw_response,
            "model": "OmniEcho",
        }

    # ------------------------------------------------------- prompt building
    @staticmethod
    def _summarize_object(obj: dict) -> dict:
        """OmniEcho processing component."""

        def _g(*keys, default=None):
            for k in keys:
                if k in obj and obj[k] is not None:
                    return obj[k]
            return default


        distance = _g("distance", default=None)
        if distance is None:
            traj = obj.get("trajectory") or []
            if traj and isinstance(traj[0], dict):
                distance = traj[0].get("distance")


        vis_window = obj.get("visibility_window", {}) or {}
        video_enter = vis_window.get("video_enter_t")
        video_exit = vis_window.get("video_exit_t")

        result = {
            "object_id": obj.get("id", ""),
            "label": obj.get("label", ""),
            "type": obj.get("type", obj.get("sound_type", "")),
            "distance": _round_or_none(distance, 2),
            "director_onset_t": _round_or_none(obj.get("onset_t"), 2),
            "director_offset_t": _round_or_none(obj.get("offset_t"), 2),
            "video_enter_t": _round_or_none(video_enter, 2),
            "video_exit_t": _round_or_none(video_exit, 2),
        }
        return result

    def _build_user_prompt(
        self,
        scene_description: str,
        objects: list[dict],
        scene_duration: float,
    ) -> str:
        summary_objs = [self._summarize_object(obj) for obj in objects]

        lines: list[str] = []
        lines.append(f"## Scene information (duration={scene_duration:.1f}s)")
        lines.append("")
        lines.append("## Scene description")
        lines.append((scene_description or "(no description)").strip())
        lines.append("")
        lines.append(f"## Sound sources ({len(summary_objs)} total)")
        lines.append("director_onset_t / director_offset_t are the original Director intervals and may be adjusted. "
                     "video_enter_t / video_exit_t are visibility intervals; null means off-screen or unavailable visibility.")
        lines.append(json.dumps(summary_objs, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("## Source category hints")
        for obj in objects:
            obj_id = obj.get("id", "")
            label = obj.get("label", "")
            obj_type = obj.get("type", obj.get("sound_type", ""))
            lines.append(f"  - {obj_id}: {label} | type={obj_type}")
        lines.append("")
        lines.append("## Decision requirements")
        lines.append("1. Continuous ambience (water/wind/rain/background) must use onset_t=0, offset_t=duration.")
        lines.append("2. Use natural event intervals; avoid durations <0.5s or implausible interruptions.")
        lines.append(f"3. At least one source must start at onset_t=0. All offset_t values must be <= {scene_duration:.1f}.")
        lines.append("4. Consider the original Director intervals and visibility; sound may begin slightly before the source appears.")
        lines.append("")
        lines.append(
            "Return decisions using the JSON schema in the system prompt. "
            "The array length and object_id values must match the sound sources above."
        )
        return "\n".join(lines)

    # -------------------------------------------------------- response parse
    def _parse_response(
        self,
        response: str,
        objects: list[dict],
        scene_duration: float,
    ) -> list[dict]:
        """OmniEcho processing component."""

        parsed = _extract_json(response)
        if isinstance(parsed, dict):
            items = parsed.get("decisions")
            if items is None:
                
                items = [parsed] if "object_id" in parsed else []
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError(f"Unexpected JSON top-level type: {type(parsed)}")

        items_by_id: dict[str, dict] = {}
        for it in items:
            if isinstance(it, dict) and it.get("object_id"):
                items_by_id[str(it["object_id"])] = it

        decisions: list[dict] = []
        for idx, obj in enumerate(objects):
            oid = obj.get("id", "")
            raw_item = items_by_id.get(oid)
            if raw_item is None and idx < len(items) and isinstance(items[idx], dict):
                raw_item = items[idx]

            fallback_onset = float(obj.get("onset_t", 0.0))
            fallback_offset = float(obj.get("offset_t", scene_duration))

            if raw_item is None:
                decisions.append({
                    "object_id": oid,
                    "onset_t": round(fallback_onset, 2),
                    "offset_t": round(fallback_offset, 2),
                    "reason": "No LLM decision returned for this source; retaining the original Director values",
                })
                continue

            try:
                onset_t = float(raw_item.get("onset_t", fallback_onset))
            except (TypeError, ValueError):
                onset_t = fallback_onset
            try:
                offset_t = float(raw_item.get("offset_t", fallback_offset))
            except (TypeError, ValueError):
                offset_t = fallback_offset


            onset_t = max(0.0, min(scene_duration, onset_t))
            offset_t = max(0.0, min(scene_duration, offset_t))
            if offset_t <= onset_t:
                
                offset_t = min(scene_duration, max(onset_t + 0.5, fallback_offset))
                if offset_t <= onset_t:
                    onset_t = 0.0
                    offset_t = scene_duration

            reason = str(raw_item.get("reason", "")).strip()
            decisions.append({
                "object_id": oid,
                "onset_t": round(onset_t, 2),
                "offset_t": round(offset_t, 2),
                "reason": reason or "(no reason)",
            })

        return decisions


# ---------------------------------------------------------------------------
# 5. helpers
# ---------------------------------------------------------------------------


def _round_or_none(v: Any, ndigits: int) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


__all__ = ["DurationAgent"]
