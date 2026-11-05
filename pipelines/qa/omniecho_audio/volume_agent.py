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
You are a spatial-audio mixing engineer. Decide each source's final target loudness in LUFS from the scene description and source attributes. Write explanations in English.

Mixing principles:
1. Distance: nearby sources should be louder; doubling distance reduces level by about 6 dB.
2. Foreground/background: events such as knocks, footsteps, and page turns should stand 6-12 dB above ambience.
3. Dynamic range: the loudest and quietest sources should differ by at least 8 dB and at most 20 dB.
4. Plausibility: relative source levels should follow real-world physics.
5. Ambient reference: set continuous background sources to -30 to -40 LUFS.
6. Nearby foreground (<2 m): -14 to -18 LUFS.
7. Narrative importance has three levels:
      - Narrative focus: the main event should be most prominent, -14 to -20 LUFS.
      - Supporting sources: add scene detail with moderate presence, -20 to -26 LUFS.
      - Ambient sources: create atmosphere without drawing attention, -28 to -38 LUFS.
8. Visual salience: a prominent visible source should have a matching audible presence.
      Off-screen sources should be 3-6 dB quieter than visible sources at the same distance.
9. Relative balance:
      - Narrative sources such as footsteps and speech must remain intelligible, never masked by ambience or machinery.
      - Repetitive mechanical sounds such as pendulums or drips should be moderate unless they are the explicit narrative focus.
      - Every source should remain perceptible; none should be completely masked.

Return JSON in this format:
{
  "decisions": [
    {
      "object_id": "source_001",
      "target_lufs": -18.0,
      "reason": "The clock is 1.5 m away and should remain prominent as the main foreground source."
    }
  ]
}

Constraints:
- decisions must contain one entry per input object with matching object_id values.
- target_lufs must be a float in [-50.0,-10.0].
- Return one JSON object only, with English text and no markdown fences or additional explanation.
"""


# ---------------------------------------------------------------------------
# 4. VolumeAgent
# ---------------------------------------------------------------------------


class VolumeAgent:
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

    def __enter__(self) -> "VolumeAgent":
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
                f"VolumeAgent permanent LLM error: {e}"
            ) from e

    # ----------------------------------------------------------- main entry
    def decide_volumes(
        self,
        scene_description: str,
        objects: list[dict],
        measured_lufs: dict[str, float],
        scene_duration: float,
    ) -> dict:
        """OmniEcho processing component."""

        if not objects:
            return {"decisions": [], "raw_response": ""}

        prompt = self._build_user_prompt(
            scene_description=scene_description,
            objects=objects,
            measured_lufs=measured_lufs,
            scene_duration=scene_duration,
        )

        raw_response = self._call_llm(_SYSTEM_PROMPT, prompt)

        decisions = self._parse_response(raw_response, objects)
        return {
            "decisions": decisions,
            "raw_response": raw_response,
            "model": "OmniEcho",
        }

    # ------------------------------------------------------- prompt building
    @staticmethod
    def _summarize_object(obj: dict, measured: Optional[float]) -> dict:
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


        azimuth = _g("azimuth", default=None)
        elevation = _g("elevation", default=None)
        if azimuth is None and obj.get("trajectory"):
            traj0 = obj["trajectory"][0] if isinstance(obj["trajectory"][0], dict) else {}
            azimuth = traj0.get("azimuth")
            elevation = traj0.get("elevation")

        result = {
            "object_id": obj.get("id", ""),
            "label": obj.get("label", ""),
            "type": obj.get("type", obj.get("sound_type", "")),
            "azimuth": _round_or_none(azimuth, 1),
            "elevation": _round_or_none(elevation, 1),
            "distance": _round_or_none(distance, 2),
            "onset_t": _round_or_none(obj.get("onset_t"), 2),
            "offset_t": _round_or_none(obj.get("offset_t"), 2),
            "director_target_lufs": _round_or_none(obj.get("loudness_lufs"), 1),
            "measured_lufs": _round_or_none(measured, 1) if measured is not None else None,
        }


        audio_stats = obj.get("_audio_stats")
        if audio_stats:
            result["rms_db"] = audio_stats.get("rms_db")
            result["peak_dbfs"] = audio_stats.get("peak_dbfs")
            result["spectral_balance"] = audio_stats.get("spectral_balance")

        return result

    def _build_user_prompt(
        self,
        scene_description: str,
        objects: list[dict],
        measured_lufs: dict[str, float],
        scene_duration: float,
    ) -> str:
        summary_objs: list[dict] = []
        for obj in objects:
            obj_id = obj.get("id", "")
            measured = measured_lufs.get(obj_id)
            summary_objs.append(self._summarize_object(obj, measured))

        lines: list[str] = []
        lines.append(f"## Scene information (duration={scene_duration:.1f}s)")
        lines.append("Coordinates: positive azimuth=listener's left, negative=right; distance is in meters.")
        lines.append("")
        lines.append("## Scene description")
        lines.append((scene_description or "(no description)").strip())
        lines.append("")
        lines.append(f"## Sound sources ({len(summary_objs)} total)")
        lines.append("director_target_lufs is the initial target loudness and may be adjusted; "
                     "measured_lufs is the measured loudness and is provided only as a reference.")
        lines.append(json.dumps(summary_objs, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("")
        lines.append("## Narrative analysis")
        lines.append("Use the following information to assess each source's narrative importance:")
        for obj in objects:
            vis_window = obj.get("visibility_window", {})
            is_visible = vis_window.get("video_enter_t", -1) >= 0
            obj_type = obj.get("type", obj.get("sound_type", ""))
            lines.append(f"  - {obj.get('id', '')}: {obj.get('label', '')} | visible={is_visible} | type={obj_type}")
        lines.append("")
        lines.append("## Decision requirements")
        lines.append("1. Keep narrative sources such as footsteps and speech audible and unmasked.")
        lines.append("2. Keep repetitive mechanical sounds (pendulum/drips/HVAC) moderate unless they are the explicit focus.")
        lines.append("3. Verify that final target_lufs values span 8-18 dB from the loudest to the quietest source.")
        lines.append("")
        lines.append(
            "Return decisions using the system prompt's JSON schema, "
            "with matching source count and object_id values."
        )
        return "\n".join(lines)

    # -------------------------------------------------------- response parse
    def _parse_response(
        self,
        response: str,
        objects: list[dict],
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
            if raw_item is None:
                
                fallback_lufs = float(obj.get("loudness_lufs", -23.0))
                decisions.append({
                    "object_id": oid,
                    "target_lufs": fallback_lufs,
                    "reason": "No LLM decision returned for this source; retaining the original Director value",
                })
                continue

            try:
                target_lufs = float(raw_item.get("target_lufs"))
            except (TypeError, ValueError):
                target_lufs = float(obj.get("loudness_lufs", -23.0))
            
            target_lufs = max(-50.0, min(-10.0, target_lufs))

            reason = str(raw_item.get("reason", "")).strip()
            decisions.append({
                "object_id": oid,
                "target_lufs": round(target_lufs, 2),
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


__all__ = ["VolumeAgent"]
