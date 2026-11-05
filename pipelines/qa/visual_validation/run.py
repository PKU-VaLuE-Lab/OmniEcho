"""OmniEcho processing component."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .config import ValidationConfig
from .pipeline import VisualValidationPipeline
from .reporter import ValidationReport


logger = logging.getLogger("visual_validation")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visual validation pipeline: check alignment between video and spatial metadata",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--sample",
        type=str,
        help="Single sample directory containing scene.json and video.mp4",
    )
    target.add_argument(
        "--batch",
        type=str,
        help="Parent directory for batch processing; discovers omniecho_qa_* subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Single-sample report directory (default: <sample>/validation)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video generation; keyframe PNG files are still written",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda / cuda:0 / cpu)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=4.0,
        help="Frame sampling rate for model inference",
    )
    parser.add_argument(
        "--depth-encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl"],
        help="Depth-Anything-V2 encoder",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed DEBUG logging",
    )
    return parser


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_config(args: argparse.Namespace) -> ValidationConfig:
    cfg = ValidationConfig.default()
    cfg.device = str(args.device)
    cfg.sample_fps = float(args.sample_fps)
    cfg.depth_encoder = str(args.depth_encoder)
    # Re-run __post_init__ side-effects manually – encoder validation.
    if cfg.depth_encoder not in {"vits", "vitb", "vitl"}:
        raise ValueError(f"invalid depth encoder {cfg.depth_encoder!r}")
    return cfg


def _print_report_summary(report: ValidationReport) -> None:
    """Print a one-screen human-readable summary to stdout."""

    print("=" * 60)
    print(f"sample        : {report.sample_id}")
    print(f"timestamp     : {report.timestamp}")
    print(f"overall_status: {report.overall_status}")

    summary = report.summary or {}
    vis = summary.get("visibility")
    if vis:
        print(
            "visibility    : {n_pass} pass / {n_warn} warn / {n_fail} fail "
            "(mean IoU={iou})".format(
                n_pass=vis.get("n_pass"),
                n_warn=vis.get("n_warn"),
                n_fail=vis.get("n_fail"),
                iou=vis.get("mean_temporal_iou"),
            )
        )
    depth = summary.get("depth")
    if depth:
        print(
            "depth         : {status} (mean rho={mean}, min rho={mn}, "
            "outliers={out})".format(
                status=depth.get("status"),
                mean=depth.get("mean_rank_correlation"),
                mn=depth.get("min_rank_correlation"),
                out=depth.get("n_outlier_frames"),
            )
        )
    spatial = summary.get("spatial")
    if spatial:
        print(
            "spatial       : {status} (az_rmse={az}, el_rmse={el}, "
            "max_dev={mx})".format(
                status=spatial.get("status"),
                az=spatial.get("overall_azimuth_rmse_deg"),
                el=spatial.get("overall_elevation_rmse_deg"),
                mx=spatial.get("overall_max_deviation_deg"),
            )
        )
    camera = summary.get("camera")
    if camera:
        print(
            "camera        : {status} (rotation expected={meta}, detected={det}, "
            "corr={c}, max_err={e})".format(
                status=camera.get("status"),
                meta=camera.get("metadata_has_rotation"),
                det=camera.get("motion_detected"),
                c=camera.get("yaw_correlation"),
                e=camera.get("max_yaw_error_deg"),
            )
        )
    errors = summary.get("errors")
    if errors:
        print("errors        :")
        for k, v in errors.items():
            print(f"  - {k}: {v}")
    print("=" * 60)


def _print_batch_summary(reports: List[ValidationReport]) -> None:
    n = len(reports)
    n_pass = sum(1 for r in reports if r.overall_status == "PASS")
    n_warn = sum(1 for r in reports if r.overall_status == "WARN")
    n_fail = sum(1 for r in reports if r.overall_status == "FAIL")
    print("=" * 60)
    print(f"batch summary : {n} samples")
    print(f"  PASS        : {n_pass}")
    print(f"  WARN        : {n_warn}")
    print(f"  FAIL        : {n_fail}")
    if n:
        print("-" * 60)
        for r in reports:
            print(f"  [{r.overall_status:<4}] {r.sample_id}")
    print("=" * 60)


def main(argv: List[str] | None = None) -> int:
    """Command-line entry point.  Returns the process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        config = _build_config(args)
    except Exception as exc:
        logger.error("invalid configuration: %s", exc)
        return 2

    pipeline = VisualValidationPipeline(config=config)
    generate_video = not args.no_video

    if args.sample:
        sample_dir = Path(args.sample).expanduser().resolve()
        if not sample_dir.is_dir():
            logger.error("sample directory not found: %s", sample_dir)
            return 2
        output_dir = (
            Path(args.output).expanduser().resolve() if args.output else None
        )
        report = pipeline.validate_sample(
            sample_dir=sample_dir,
            generate_video=generate_video,
            output_dir=output_dir,
        )
        _print_report_summary(report)
        return 0 if report.overall_status != "FAIL" else 1

    # Batch mode.
    batch_dir = Path(args.batch).expanduser().resolve()
    if not batch_dir.is_dir():
        logger.error("batch directory not found: %s", batch_dir)
        return 2
    if args.output:
        logger.warning("--output is ignored in batch mode; reports are written next to each sample")
    reports = pipeline.validate_batch(
        output_dir=batch_dir,
        generate_video=generate_video,
    )
    _print_batch_summary(reports)
    if not reports:
        return 2
    return 0 if all(r.overall_status != "FAIL" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
