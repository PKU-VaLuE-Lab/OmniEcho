"""OmniEcho processing component."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from omniecho_audio.api_client import (
    call_llm_with_system,
    APIPermanentError,
)

from omniecho_audio.schema import (
    AnnotatedAudioObject,
    OffScreenStrategy,
    AnnotatedTrajectoryPoint,
    VisibilityState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Output containers
# ---------------------------------------------------------------------------


@dataclass
class OffScreenReasoning:
    """OmniEcho processing component."""

    object_id: str
    original_strategy: OffScreenStrategy
    
    direction_plausible: bool = True
    reasoning: str = ""
    
    suggested_trajectory: Optional[list[dict]] = None
    trajectory_confidence: float = 0.4
    
    acoustic_adjustments: dict = field(default_factory=dict)
    
    reviewed: bool = True
    
    error: str = ""


@dataclass
class OffScreenAgentReport:
    """OmniEcho processing component."""

    total_offscreen_objects: int
    reviewed_objects: int
    adjustments_made: int
    reasonings: list[OffScreenReasoning] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_offscreen_objects": self.total_offscreen_objects,
            "reviewed_objects": self.reviewed_objects,
            "adjustments_made": self.adjustments_made,
            "reasonings": [
                {
                    **asdict(r),
                    "original_strategy": r.original_strategy.value,
                }
                for r in self.reasonings
            ],
        }


# ---------------------------------------------------------------------------
# 3. JSON extraction helper
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


# ---------------------------------------------------------------------------
# 4. OffScreenAgent
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are a Spatial Audio Scene Analyst.
Review off-screen sources in a static-camera video for semantic plausibility and suggest trajectories and acoustics. Write explanations in English.

Input includes:
1. A natural-language scene description (video_description).
2. Camera and coordinates: fixed position, FOV approximately 90 degrees, positive azimuth=left and negative=right.
3. offscreen_objects, each containing:
   - object_id, label, off_screen_strategy
      - last_visible: (t, azimuth, elevation, distance) at the last visible moment.
          For strategy=trust_metadata, this is the original metadata starting point.
      - extrapolated_tail: current extrapolated off-screen trajectory for review.
   - acoustic_window: (acoustic_start, acoustic_end)
      - duration: total scene duration.

Assess:
A. Whether the exit direction fits the scene.
      Example: a vehicle leaving through the right edge should continue away with negative azimuth,
          rather than suddenly reverse toward azimuth=0.
B. How motion should evolve after exit:
      - Continue straight (vehicle, bird, pedestrian).
      - Slow to a stop (walking into a corner; a cat settling after a door closes).
      - Change direction (turning or moving around a wall).
C. Acoustic suggestions:
      - Behind a wall or around a corner: reverb_increase_db in [+3,+10], direct_attenuation_db
     ∈ [-3, -10], hf_rolloff_hz ∈ [1500, 3500]
      - Increasing distance only: reverb_increase_db in [+1,+3], direct_attenuation_db in [-3,-6].
      - Continuing through open space: all adjustments zero (retain the current model).

=== Output format ===
Return one JSON object only, with no markdown or explanations outside the object. Its top-level key is "reasonings",
an array of objects with this schema:
{
  "object_id": "<original object_id>",
  "direction_plausible": true | false,
  "reasoning": "<1-3 English sentences>",
  "suggested_trajectory": [
    {"t": <float>, "azimuth": <float>, "elevation": <float>, "distance": <float>},
    ...
  ] | null,
  "trajectory_confidence": <float in [0.0, 1.0]>,
  "acoustic_adjustments": {
    "reverb_increase_db": <float>,
    "direct_attenuation_db": <float>,
    "hf_rolloff_hz": <float; 0 means no roll-off>
  }
}

Constraints:
- If direction_plausible=true and no correction is needed, suggested_trajectory must be null.
    Do not add trajectory points unnecessarily.
- Suggested trajectory azimuth magnitude must be >45 degrees (off-screen) and <=135.
- trajectory_confidence must be in [0.2,0.6], because this is LLM inference, not visual measurement.
- Supply all three acoustic_adjustments fields; zero means no change.
- reasonings must match the length and ordering of offscreen_objects.
"""


class OffScreenAgent:
    """OmniEcho processing component."""


    OFFSCREEN_STRATEGIES = {
        OffScreenStrategy.EXTRAPOLATE,
        OffScreenStrategy.TRUST_METADATA,
    }

    SUGGESTED_CONFIDENCE = 0.4
    LLM_TIMEOUT = 120.0
    MAX_HTTP_RETRIES = 3

    def __init__(
        self,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        
        self.model = "OmniEcho"

    # ---------------------------------------------------------------- LLM I/O
    def close(self) -> None:
        """OmniEcho processing component."""
        pass

    def __enter__(self) -> "OffScreenAgent":
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
                f"OffScreenAgent permanent LLM error: {e}"
            ) from e

    # ----------------------------------------------------------- main entry
    def review_offscreen_sources(
        self,
        corrected_scene_path: Path | str,
        original_scene_path: Path | str,
        video_description: str = "",
    ) -> OffScreenAgentReport:
        """OmniEcho processing component."""

        corrected_scene_path = Path(corrected_scene_path)
        original_scene_path = Path(original_scene_path)

        if not corrected_scene_path.is_file():
            raise FileNotFoundError(
                f"corrected_scene not found: {corrected_scene_path}"
            )

        scene_data = json.loads(corrected_scene_path.read_text(encoding="utf-8"))
        original_data: dict[str, Any] = {}
        if original_scene_path.is_file():
            try:
                original_data = json.loads(
                    original_scene_path.read_text(encoding="utf-8")
                )
            except Exception as exc:  
                logger.warning("OffScreenAgent: failed to read original scene: %s", exc)

        if not video_description:
            video_description = _infer_video_description(scene_data, original_data)

        offscreen_objects = self._collect_offscreen_objects(scene_data)
        total = len(offscreen_objects)
        if total == 0:
            logger.info("OffScreenAgent: no off-screen sources need review")
            return OffScreenAgentReport(
                total_offscreen_objects=0,
                reviewed_objects=0,
                adjustments_made=0,
                reasonings=[],
            )

        prompt = self._build_review_prompt(
            scene_data, offscreen_objects, video_description
        )

        try:
            raw_response = self._call_llm(_SYSTEM_PROMPT, prompt)
        except Exception as exc:
            logger.warning(
                "OffScreenAgent: LLM call failed; skipping review for all sources: %s", exc
            )
            reasonings = [
                OffScreenReasoning(
                    object_id=o["object_id"],
                    original_strategy=OffScreenStrategy(o["off_screen_strategy"]),
                    direction_plausible=True,
                    reasoning="LLM unavailable, skip review",
                    reviewed=False,
                    error=str(exc),
                )
                for o in offscreen_objects
            ]
            return OffScreenAgentReport(
                total_offscreen_objects=total,
                reviewed_objects=0,
                adjustments_made=0,
                reasonings=reasonings,
            )

        try:
            reasonings = self._parse_llm_response(raw_response, offscreen_objects)
        except Exception as exc:
            logger.warning(
                "OffScreenAgent: failed to parse LLM response (%s), head=%s",
                exc, raw_response[:200],
            )
            reasonings = [
                OffScreenReasoning(
                    object_id=o["object_id"],
                    original_strategy=OffScreenStrategy(o["off_screen_strategy"]),
                    direction_plausible=True,
                    reasoning="LLM response unparsable, skip review",
                    reviewed=False,
                    error=str(exc),
                )
                for o in offscreen_objects
            ]

        adjustments = sum(
            1 for r in reasonings
            if r.reviewed and (
                not r.direction_plausible or r.suggested_trajectory is not None
            )
        )
        reviewed = sum(1 for r in reasonings if r.reviewed)

        report = OffScreenAgentReport(
            total_offscreen_objects=total,
            reviewed_objects=reviewed,
            adjustments_made=adjustments,
            reasonings=reasonings,
        )

        for r in reasonings:
            logger.info(
                "  [%s] reviewed=%s plausible=%s reasoning=%s",
                r.object_id, r.reviewed, r.direction_plausible,
                (r.reasoning or "")[:80],
            )

        return report

    # ------------------------------------------------------- prompt building
    def _collect_offscreen_objects(self, scene_data: dict) -> list[dict]:
        """OmniEcho processing component."""

        duration = float(
            scene_data.get("duration_sec", scene_data.get("duration", 6.0))
        )
        out: list[dict] = []
        for obj in scene_data.get("objects", []):
            try:
                strategy = OffScreenStrategy(
                    obj.get("off_screen_strategy", "trust_metadata")
                )
            except ValueError:
                continue
            if strategy not in self.OFFSCREEN_STRATEGIES:
                continue

            visual_enter = float(obj.get("visual_enter", -1.0))
            visual_exit = float(obj.get("visual_exit", -1.0))
            traj = obj.get("trajectory", []) or []


            last_visible: Optional[dict] = None
            tail_pts: list[dict] = []
            for p in traj:
                state = p.get("visibility_state")
                if state in ("on_screen", "entering"):
                    last_visible = p
                if state in ("off_screen_dynamic", "off_screen_always"):
                    tail_pts.append(p)
            if last_visible is None and traj:
                last_visible = traj[0]

            out.append(
                {
                    "object_id": obj.get("id", ""),
                    "label": obj.get("label", obj.get("type", "")),
                    "off_screen_strategy": strategy.value,
                    "visual_enter": visual_enter,
                    "visual_exit": visual_exit,
                    "last_visible": _condense_point(last_visible) if last_visible else None,
                    "extrapolated_tail": [
                        _condense_point(p) for p in tail_pts[:8]
                    ],
                    "acoustic_window": [
                        float(obj.get("acoustic_start", 0.0)),
                        float(obj.get("acoustic_end", duration)),
                    ],
                    "duration": duration,
                }
            )
        return out

    def _build_review_prompt(
        self,
        scene_data: dict,
        offscreen_objects: list,
        video_description: str,
    ) -> str:
        """OmniEcho processing component."""

        scene_id = scene_data.get("scene_id", "?")
        duration = float(
            scene_data.get("duration_sec", scene_data.get("duration", 6.0))
        )
        environment = scene_data.get("environment", {}) or {}
        env_summary = {
            "type": environment.get("type"),
            "description": environment.get("description"),
            "room": environment.get("room"),
        }

        lines: list[str] = []
        lines.append(f"## Scene information (scene_id={scene_id}, duration={duration:.1f}s)")
        lines.append("Camera: fixed (camera_mode=static), FOV approximately 90 degrees (in-frame |az|<45 degrees).")
        lines.append("Coordinates: positive azimuth=listener's left, negative=right; positive elevation=up.")
        lines.append("")
        lines.append("## Video/scene description")
        lines.append(video_description.strip() or "(none)")
        lines.append("")
        lines.append("## Environment summary")
        lines.append(json.dumps(env_summary, ensure_ascii=False))
        lines.append("")
        lines.append(f"## Off-screen sources ({len(offscreen_objects)} total)")
        lines.append(json.dumps(offscreen_objects, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append(
            "Return the reasonings array using the system prompt's JSON schema, "
            "with the same length and ordering as the off-screen sources above."
        )
        return "\n".join(lines)

    # -------------------------------------------------------- response parse
    def _parse_llm_response(
        self,
        response: str,
        offscreen_objects: list[dict],
    ) -> list[OffScreenReasoning]:
        """OmniEcho processing component."""

        parsed = _extract_json(response)
        if isinstance(parsed, dict):
            items = parsed.get("reasonings")
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

        results: list[OffScreenReasoning] = []
        for idx, src in enumerate(offscreen_objects):
            oid = src["object_id"]
            strategy = OffScreenStrategy(src["off_screen_strategy"])
            raw_item = items_by_id.get(oid)
            if raw_item is None and idx < len(items) and isinstance(items[idx], dict):
                raw_item = items[idx]
            if raw_item is None:
                results.append(
                    OffScreenReasoning(
                        object_id=oid,
                        original_strategy=strategy,
                        reviewed=False,
                        error="missing in LLM response",
                    )
                )
                continue
            try:
                results.append(_build_reasoning(oid, strategy, raw_item))
            except Exception as exc:
                results.append(
                    OffScreenReasoning(
                        object_id=oid,
                        original_strategy=strategy,
                        reviewed=False,
                        error=f"parse error: {exc}",
                    )
                )
        return results

    # --------------------------------------------------- apply to scene file
    def apply_adjustments(
        self,
        corrected_scene_path: Path | str,
        report: OffScreenAgentReport,
        output_path: Optional[Path | str] = None,
    ) -> Path:
        """OmniEcho processing component."""

        corrected_scene_path = Path(corrected_scene_path)
        if output_path is None:
            output_path = corrected_scene_path
        else:
            output_path = Path(output_path)

        scene_data = json.loads(corrected_scene_path.read_text(encoding="utf-8"))
        objects = scene_data.get("objects", []) or []
        objs_by_id = {obj.get("id", ""): obj for obj in objects}

        modified = 0
        for r in report.reasonings:
            if not r.reviewed:
                continue
            obj = objs_by_id.get(r.object_id)
            if obj is None:
                continue


            if r.acoustic_adjustments:
                obj["acoustic_adjustments"] = {
                    k: float(v) for k, v in r.acoustic_adjustments.items()
                }
                obj["acoustic_adjustments"]["source"] = "off_screen_agent"


            if (not r.direction_plausible) and r.suggested_trajectory:
                self._rewrite_offscreen_trajectory(obj, r)
                modified += 1
            elif r.suggested_trajectory and not r.direction_plausible:
                
                pass

        scene_data.setdefault("annotation_metadata", {})
        scene_data["annotation_metadata"]["off_screen_agent"] = {
            "total_offscreen_objects": report.total_offscreen_objects,
            "reviewed_objects": report.reviewed_objects,
            "adjustments_made": modified,
            "model": self.model,
        }

        output_path.write_text(
            json.dumps(scene_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "OffScreenAgent: applied changes (%d/%d modified) -> %s",
            modified, report.total_offscreen_objects, output_path,
        )
        return output_path

    def _rewrite_offscreen_trajectory(
        self,
        obj: dict,
        reasoning: OffScreenReasoning,
    ) -> None:
        """OmniEcho processing component."""

        suggested = reasoning.suggested_trajectory or []
        if not suggested:
            return


        kept: list[dict] = []
        offscreen_state = VisibilityState.OFF_SCREEN_DYNAMIC.value
        for p in obj.get("trajectory", []) or []:
            state = p.get("visibility_state")
            if state in ("off_screen_dynamic", "off_screen_always"):
                offscreen_state = state
                continue
            kept.append(p)


        conf = float(
            reasoning.trajectory_confidence
            if reasoning.trajectory_confidence is not None
            else self.SUGGESTED_CONFIDENCE
        )
        conf = max(0.2, min(0.6, conf))

        new_offscreen_pts: list[dict] = []
        for raw in suggested:
            try:
                t = float(raw.get("t") if "t" in raw else raw.get("time", 0.0))
                az = float(raw.get("azimuth", 0.0))
                el = float(raw.get("elevation", 0.0))
                d = float(raw.get("distance", 3.0))
            except (TypeError, ValueError):
                continue
            
            if abs(az) <= 45.0:
                az = 50.0 if az >= 0 else -50.0
            if az > 135.0:
                az = 135.0
            elif az < -135.0:
                az = -135.0
            new_offscreen_pts.append(
                {
                    "t": round(t, 2),
                    "azimuth": round(az, 2),
                    "elevation": round(el, 2),
                    "distance": round(max(0.3, d), 3),
                    "gain_db": 0.0,
                    "visibility_state": offscreen_state,
                    "confidence": round(conf, 2),
                }
            )


        merged = kept + new_offscreen_pts
        merged.sort(key=lambda p: float(p.get("t", 0.0)))
        obj["trajectory"] = merged
        obj["off_screen_strategy"] = "trust_metadata"  
        obj.setdefault("corrections", []).append(
            {
                "source": "off_screen_agent",
                "reasoning": reasoning.reasoning[:200],
                "n_new_points": len(new_offscreen_pts),
            }
        )

    # --------------------------------------------------- helper for renderer
    def _format_acoustic_suggestion(
        self, reasoning: OffScreenReasoning
    ) -> dict:
        """OmniEcho processing component."""

        adj = reasoning.acoustic_adjustments or {}
        return {
            "reverb_increase_db": float(_clamp(adj.get("reverb_increase_db", 0.0), 0.0, 12.0)),
            "direct_attenuation_db": float(_clamp(adj.get("direct_attenuation_db", 0.0), -15.0, 0.0)),
            "hf_rolloff_hz": float(_clamp(adj.get("hf_rolloff_hz", 0.0), 0.0, 20000.0)),
        }


# ---------------------------------------------------------------------------
# 5. Module-level helpers
# ---------------------------------------------------------------------------


def _condense_point(point: Optional[dict]) -> Optional[dict]:
    """OmniEcho processing component."""

    if not point:
        return None
    return {
        "t": round(float(point.get("t", 0.0)), 2),
        "azimuth": round(float(point.get("azimuth", 0.0)), 2),
        "elevation": round(float(point.get("elevation", 0.0)), 2),
        "distance": round(float(point.get("distance", 0.0)), 3),
        "visibility_state": point.get("visibility_state"),
        "confidence": round(float(point.get("confidence", 0.0)), 2),
    }


def _clamp(value: Any, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(lo, min(hi, v))


def _build_reasoning(
    object_id: str,
    strategy: OffScreenStrategy,
    raw: dict,
) -> OffScreenReasoning:
    """OmniEcho processing component."""

    plausible = bool(raw.get("direction_plausible", True))
    reasoning_text = str(raw.get("reasoning", "")).strip()

    suggested = raw.get("suggested_trajectory")
    if suggested in (None, [], "", "null"):
        suggested_traj: Optional[list[dict]] = None
    elif isinstance(suggested, list):
        clean: list[dict] = []
        for p in suggested:
            if not isinstance(p, dict):
                continue
            try:
                clean.append(
                    {
                        "t": float(p.get("t", p.get("time", 0.0))),
                        "azimuth": float(p.get("azimuth", 0.0)),
                        "elevation": float(p.get("elevation", 0.0)),
                        "distance": float(p.get("distance", 3.0)),
                    }
                )
            except (TypeError, ValueError):
                continue
        suggested_traj = clean if clean else None
    else:
        suggested_traj = None

    try:
        traj_conf = float(raw.get("trajectory_confidence", 0.4))
    except (TypeError, ValueError):
        traj_conf = 0.4
    traj_conf = max(0.2, min(0.6, traj_conf))

    raw_adj = raw.get("acoustic_adjustments") or {}
    adjustments = {
        "reverb_increase_db": _clamp(raw_adj.get("reverb_increase_db", 0.0), 0.0, 12.0),
        "direct_attenuation_db": _clamp(raw_adj.get("direct_attenuation_db", 0.0), -15.0, 0.0),
        "hf_rolloff_hz": _clamp(raw_adj.get("hf_rolloff_hz", 0.0), 0.0, 20000.0),
    }

    return OffScreenReasoning(
        object_id=object_id,
        original_strategy=strategy,
        direction_plausible=plausible,
        reasoning=reasoning_text,
        suggested_trajectory=suggested_traj,
        trajectory_confidence=traj_conf,
        acoustic_adjustments=adjustments,
        reviewed=True,
    )


def _infer_video_description(
    corrected_scene: dict,
    original_scene: dict,
) -> str:
    """OmniEcho processing component."""

    parts: list[str] = []
    env = (original_scene or corrected_scene).get("environment", {}) or {}
    desc = env.get("description")
    if desc:
        parts.append(str(desc))

    objs = (original_scene or corrected_scene).get("objects", []) or []
    if objs:
        labels = [str(o.get("label") or o.get("type") or o.get("id")) for o in objs]
        parts.append("Sound sources: " + ", ".join([l for l in labels if l]))

    video_prompt = original_scene.get("video_prompt") if original_scene else None
    if video_prompt and isinstance(video_prompt, str):
        parts.append(video_prompt[:400])

    return "\n".join(parts) if parts else "(no description)"


def _parse_llm_text(data: Any) -> str:
    """OmniEcho processing component."""

    if isinstance(data, dict) and "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if content:
            return content

    if isinstance(data, dict) and "content" in data:
        c = data["content"]
        if isinstance(c, list) and c:
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "")
                    if txt:
                        return txt
            for block in c:
                if isinstance(block, dict) and block.get("text"):
                    return block["text"]
        if isinstance(c, str):
            return c

    raise RuntimeError(f"Unrecognized LLM response shape: {str(data)[:300]}")


__all__ = [
    "OffScreenAgent",
    "OffScreenAgentReport",
    "OffScreenReasoning",
]
