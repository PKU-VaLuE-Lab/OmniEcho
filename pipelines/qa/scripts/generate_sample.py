"""OmniEcho processing component."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_validation.metadata_correction import MetadataCorrector  # noqa: F401,E402

from omniecho_audio.orchestrator import (  # noqa: E402
    PipelineConfig,
    generate_single_sample,
)
from omniecho_audio.schema import CameraMode  # noqa: E402


_DEFAULT_OUTPUT_ROOT = Path(os.environ.get("OMNIECHO_QA_OUTPUT", "outputs/OmniEchoBench-QA"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one OmniEcho sample with a static camera")
    parser.add_argument("sample_id", help="Sample ID, e.g. omniecho_qa_001")
    parser.add_argument(
        "prev_desc_file",
        type=Path,
        help="JSON list[str] containing environment.description values from previously generated scenes",
    )
    parser.add_argument(
        "--camera-mode",
        default="static",
        choices=[m.value for m in CameraMode],
        help="Camera mode (only static is supported; pan/rotate are not implemented)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help="Output root directory (default: outputs/OmniEchoBench-QA/)",
    )
    parser.add_argument("--n-foreground", type=int, default=3)
    parser.add_argument("--scene-duration", type=float, default=6.0)
    parser.add_argument("--no-ambient", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--max-stage-retries", type=int, default=3)
    return parser.parse_args(argv)


def _load_previous_scenes(prev_desc_file: Path) -> list[str]:
    if not prev_desc_file.exists():
        return []
    try:
        data = json.loads(prev_desc_file.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    logger = logging.getLogger("generate_sample")

    camera_mode = CameraMode(args.camera_mode)
    if camera_mode != CameraMode.STATIC:
        logger.warning(
            "camera_mode=%s has schema support only; metadata correction will raise NotImplementedError",
            camera_mode.value,
        )

    previous_scenes = _load_previous_scenes(args.prev_desc_file)

    config = PipelineConfig(
        output_root=args.output_root,
        n_foreground=args.n_foreground,
        include_ambient=not args.no_ambient,
        
        allow_camera_rotation=False,
        camera_mode=camera_mode,
        scene_duration=args.scene_duration,
        skip_video=args.skip_video,
        max_stage_retries=args.max_stage_retries,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "OmniEcho pipeline start: sample_id=%s camera_mode=%s output_root=%s",
        args.sample_id, camera_mode.value, config.output_root,
    )

    sample_dir = generate_single_sample(
        config=config,
        sample_id=args.sample_id,
        previous_scenes=previous_scenes,
    )
    print(f"DONE {args.sample_id} -> {sample_dir}")


if __name__ == "__main__":
    main()
