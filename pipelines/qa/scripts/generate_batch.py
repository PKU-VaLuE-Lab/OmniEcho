"""OmniEcho processing component."""
from __future__ import annotations

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import argparse
import gc
import json
import logging
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


os.environ.setdefault("VIDEO_MODEL", "OmniEcho")

if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set. LLM calls will fail; set it and restart.", flush=True)

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)


try:
    import cv2  # noqa: F401
    from visual_validation.metadata_correction import MetadataCorrector  # noqa: F401
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def _setup_worker_env(api_key: str = "") -> None:
    """OmniEcho processing component."""
    os.environ.setdefault("VIDEO_MODEL", "OmniEcho")
    
    key = api_key or _OPENAI_API_KEY
    if key:
        os.environ["OPENAI_API_KEY"] = key
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def generate_one_sample(args_tuple):
    """OmniEcho processing component."""
    sample_id, output_root, prev_file, scene_duration, stagger_delay, api_key = args_tuple
    
    if stagger_delay > 0:
        import random
        time.sleep(stagger_delay + random.uniform(0, 1.0))
    _setup_worker_env(api_key)

    t_start = time.time()
    try:
        
        from omniecho_audio.orchestrator import (
            PipelineConfig,
            generate_single_sample,
        )
        from omniecho_audio.schema import CameraMode


        from visual_validation.metadata_correction import (  # noqa: F401
            MetadataCorrector,
        )

        config = PipelineConfig(
            output_root=Path(output_root),
            camera_mode=CameraMode.STATIC,
            allow_camera_rotation=False,
            scene_duration=float(scene_duration),
        )
        config.output_root.mkdir(parents=True, exist_ok=True)


        previous_scenes: list[str] = []
        if prev_file and os.path.exists(prev_file):
            try:
                with open(prev_file, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    
                    previous_scenes = [str(x) for x in data[-30:]]
            except Exception:
                previous_scenes = []

        sample_dir = generate_single_sample(
            config=config,
            sample_id=sample_id,
            previous_scenes=previous_scenes,
        )

        elapsed = time.time() - t_start
        metadata_file = (
            Path(sample_dir) / "metadata.json" if sample_dir else None
        )
        if metadata_file and metadata_file.exists():
            return (sample_id, "SUCCESS", f"{sample_dir} ({elapsed:.0f}s)")
        return (
            sample_id,
            "PARTIAL",
            f"{sample_dir if sample_dir else 'no_dir'} ({elapsed:.0f}s)",
        )
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t_start
        return (
            sample_id,
            "FAILED",
            f"{type(e).__name__}: {str(e)[:200]} ({elapsed:.0f}s)",
        )
    finally:
        
        gc.collect()


def _append_prev_scene(prev_file: str, sample_dir: Path) -> None:
    """OmniEcho processing component."""
    scene_path = sample_dir / "scene.json"
    if not scene_path.exists():
        return
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        desc = (scene.get("environment") or {}).get("description", "")
        if not desc:
            return
        prev: list[str] = []
        if os.path.exists(prev_file):
            try:
                with open(prev_file, encoding="utf-8") as f:
                    prev = json.load(f) or []
            except Exception:
                prev = []
        if desc not in prev:
            prev.append(desc)
            with open(prev_file, "w", encoding="utf-8") as f:
                json.dump(prev, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Concurrent OmniEcho batch generation")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument(
        "--output_root", type=str, default="outputs/OmniEchoBench-QA_batch200"
    )
    parser.add_argument("--start_id", type=int, default=1, help="Starting sample index")
    parser.add_argument("--scene_duration", type=float, default=6.0)
    args = parser.parse_args()


    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"ulimit -n set to {target} (was {soft})", flush=True)
    except Exception as e:
        print(f"WARNING: Cannot set RLIMIT_NOFILE: {e}", flush=True)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path(PROJECT_ROOT) / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    prev_file = str(output_root / ".previous_scenes.json")
    if not os.path.exists(prev_file):
        with open(prev_file, "w", encoding="utf-8") as f:
            json.dump([], f)


    tasks = []
    skipped = 0
    for i in range(args.start_id, args.start_id + args.num_samples):
        sample_id = f"omniecho_qa_{i:03d}"
        sample_dir = output_root / sample_id
        metadata_file = sample_dir / "metadata.json"
        if metadata_file.exists():
            skipped += 1
            continue


        stagger_delay = min(len(tasks), 19) * 1.5
        tasks.append(
            (sample_id, str(output_root), prev_file, args.scene_duration, stagger_delay, _OPENAI_API_KEY)
        )

    print("===== Starting OmniEcho batch generation =====", flush=True)
    print(
        f"Target: {args.num_samples} | Pending: {len(tasks)} | Skipped: {skipped}"
        f" | Workers: {args.workers}",
        flush=True,
    )
    print(f"Output: {output_root}", flush=True)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    if not tasks:
        print("No tasks to run.", flush=True)
        return

    results = {"SUCCESS": [], "PARTIAL": [], "FAILED": []}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one_sample, task): task[0] for task in tasks
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                sid, status, info = future.result()
                results[status].append(sid)
                
                if status == "SUCCESS":
                    _append_prev_scene(prev_file, output_root / sid)
            except Exception as e:  # noqa: BLE001
                results["FAILED"].append(sample_id)
                sid, status, info = sample_id, "EXCEPTION", str(e)[:200]

            total_done = sum(len(v) for v in results.values())
            elapsed = time.time() - t_start
            avg = elapsed / max(1, total_done)
            eta = avg * (len(tasks) - total_done)
            print(
                f"[{total_done}/{len(tasks)}] {sid}: {status} - {info[:120]} "
                f"| elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )


            if total_done % 5 == 0 or total_done == len(tasks):
                _write_report(output_root, tasks, results, t_start, partial=True)

    _write_report(output_root, tasks, results, t_start, partial=False)
    print("\n===== Batch generation complete =====", flush=True)
    print(
        f"Succeeded: {len(results['SUCCESS'])} | Partial: {len(results['PARTIAL'])} "
        f"| Failed: {len(results['FAILED'])}",
        flush=True,
    )


def _write_report(
    output_root: Path,
    tasks: list,
    results: dict,
    t_start: float,
    partial: bool,
) -> None:
    report = {
        "total": len(tasks),
        "success": len(results["SUCCESS"]),
        "partial": len(results["PARTIAL"]),
        "failed": len(results["FAILED"]),
        "success_ids": sorted(results["SUCCESS"]),
        "partial_ids": sorted(results["PARTIAL"]),
        "failed_ids": sorted(results["FAILED"]),
        "elapsed_sec": round(time.time() - t_start, 1),
        "in_progress": partial,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_file = output_root / "batch_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
