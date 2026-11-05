#!/usr/bin/env python3
"""OmniEcho processing component."""

import argparse
import json
import os
import re
import shutil
import glob
from pathlib import Path


BASE_DIR = Path.cwd()

BATCH_DIRS = []
QUARANTINE_DIR = BASE_DIR / "quarantine"

FOV_HALF = 45.0  # ±45° FOV
MOVEMENT_THRESHOLD = 120.0  


def interpolate_value(trajectory, t, key="azimuth"):
    """OmniEcho processing component."""
    if not trajectory:
        return 0.0
    if t <= trajectory[0]["t"]:
        return trajectory[0][key]
    if t >= trajectory[-1]["t"]:
        return trajectory[-1][key]
    for i in range(len(trajectory) - 1):
        t0 = trajectory[i]["t"]
        t1 = trajectory[i + 1]["t"]
        if t0 <= t <= t1:
            if t1 == t0:
                return trajectory[i][key]
            ratio = (t - t0) / (t1 - t0)
            v0 = trajectory[i][key]
            v1 = trajectory[i + 1][key]
            return v0 + ratio * (v1 - v0)
    return trajectory[-1][key]


def get_listener_yaw(listener_data, t):
    """OmniEcho processing component."""
    orientation = listener_data.get("orientation", {})
    trajectory = orientation.get("trajectory", [])
    if not trajectory:
        return orientation.get("yaw", 0.0)
    return interpolate_value(trajectory, t, key="yaw")


def is_reasonable_mover(source_type, source_label):
    """OmniEcho processing component."""
    type_lower = source_type.lower() if source_type else ""
    label_lower = source_label.lower() if source_label else ""
    words = set(re.findall(r"[a-z]+", type_lower + " " + label_lower))


    footstep_keywords = ["footstep", "footsteps", "step", "steps"]
    footstep_exact = {"footsteps", "cat_footsteps", "dog_footsteps"}
    if type_lower in footstep_exact or any(k in words for k in footstep_keywords):
        return True


    flight_keywords = ["fly", "flying", "flight", "wing", "wings", "flapping", "soar", "soaring", "glide", "gliding", "bird", "birds", "bat", "bats"]
    flight_exact = {"bird", "bat", "drone"}
    if type_lower in flight_exact or any(k in words for k in flight_keywords):
        return True


    vehicle_keywords = ["vehicle", "vehicles", "car", "cars", "bike", "bikes", "scooter", "scooters", "automobile", "automobiles", "riding", "bicycle", "motorcycle"]
    vehicle_exact = {"car", "bicycle", "motorcycle"}
    if type_lower in vehicle_exact or any(k in words for k in vehicle_keywords):
        return True


    human_keywords = ["walk", "walking", "run", "running", "jog", "jogging", "sprint", "sprinting", "person", "people", "human"]
    if any(k in words for k in human_keywords):
        return True


    animal_keywords = ["fish", "turtle", "turtles", "snake", "snakes", "cat", "cats", "dog", "dogs", "tortoise"]
    if any(k in words for k in animal_keywords):
        return True

    return False


def check_problem_a(source, listener_data, duration):
    """OmniEcho processing component."""
    role = source.get("role", "")
    if role == "ambient":
        return False, ""

    vis = source.get("visibility_window", {})
    enter_t = vis.get("video_enter_t", -1)
    exit_t = vis.get("video_exit_t", -1)


    if enter_t == -1 or exit_t == -1:
        return False, ""


    if enter_t != 0 or exit_t < duration:
        return False, ""

    trajectory = source.get("trajectory", [])
    if not trajectory:
        return False, ""


    time_points = set()
    for pt in trajectory:
        time_points.add(pt["t"])
    
    sorted_times = sorted(time_points)
    mid_points = []
    for i in range(len(sorted_times) - 1):
        mid_points.append((sorted_times[i] + sorted_times[i + 1]) / 2.0)
    time_points.update(mid_points)
    
    t = 0.0
    while t <= duration:
        time_points.add(round(t, 2))
        t += 0.5

    worst_az = 0
    for t in sorted(time_points):
        src_az = interpolate_value(trajectory, t, "azimuth")
        listener_yaw = get_listener_yaw(listener_data, t)
        camera_rel_az = src_az - listener_yaw
        if abs(camera_rel_az) > abs(worst_az):
            worst_az = camera_rel_az

    if abs(worst_az) > FOV_HALF:
        detail = f"{source['id']} '{source.get('label', '')}' az={worst_az:.0f}° (vis=[{enter_t},{exit_t}])"
        return True, detail

    return False, ""


def check_problem_b(source):
    """OmniEcho processing component."""
    role = source.get("role", "")
    if role == "ambient":
        return False, ""

    trajectory = source.get("trajectory", [])
    if len(trajectory) < 2:
        return False, ""

    azimuths = [pt["azimuth"] for pt in trajectory]
    az_range = max(azimuths) - min(azimuths)

    if az_range <= MOVEMENT_THRESHOLD:
        return False, ""

    source_type = source.get("type", "")
    source_label = source.get("label", "")

    if is_reasonable_mover(source_type, source_label):
        return False, ""

    detail = f"{source['id']} '{source_label}' az_range={az_range:.0f}°"
    return True, detail


def scan_samples():
    """OmniEcho processing component."""
    results_a = []  # [(batch_label, sample_dir_name, detail), ...]
    results_b = []
    all_problem_samples = set()  # (batch_label, sample_dir_name)

    total_checked = 0

    for batch_dir in BATCH_DIRS:
        batch_label = batch_dir.name  # e.g., OmniEchoBench-QA_batch200
        short_label = batch_label.replace("OmniEchoBench-QA_", "")  # batch200 or batch800

        pattern = str(batch_dir / "omniecho_qa_*" / "scene.json")
        scene_files = sorted(glob.glob(pattern))

        for scene_path in scene_files:
            scene_path = Path(scene_path)
            sample_dir = scene_path.parent
            sample_name = sample_dir.name  # e.g., omniecho_qa_001

            try:
                with open(scene_path, "r", encoding="utf-8") as f:
                    scene = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARN] Cannot read {scene_path}: {e}")
                continue

            total_checked += 1
            duration = scene.get("duration", scene.get("duration_sec", 6.0))
            listener_data = scene.get("listener", {})
            objects = scene.get("objects", [])

            sample_has_a = False
            sample_has_b = False

            for source in objects:
                
                is_a, detail_a = check_problem_a(source, listener_data, duration)
                if is_a:
                    results_a.append((short_label, sample_name, detail_a))
                    sample_has_a = True


                is_b, detail_b = check_problem_b(source)
                if is_b:
                    results_b.append((short_label, sample_name, detail_b))
                    sample_has_b = True

            if sample_has_a or sample_has_b:
                all_problem_samples.add((short_label, sample_name, str(sample_dir)))

    return total_checked, results_a, results_b, all_problem_samples


def quarantine_samples(problem_samples):
    """OmniEcho processing component."""
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    moved = 0
    for short_label, sample_name, sample_path in sorted(problem_samples):
        dest = QUARANTINE_DIR / short_label / sample_name
        os.makedirs(dest.parent, exist_ok=True)
        if os.path.exists(sample_path):
            shutil.move(sample_path, str(dest))
            moved += 1
    return moved


def main():
    global BATCH_DIRS, QUARANTINE_DIR
    parser = argparse.ArgumentParser(description="Filter generated OmniEchoBench-QA scenes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    args = parser.parse_args()
    BATCH_DIRS = [args.input]
    QUARANTINE_DIR = args.quarantine
    total_checked, results_a, results_b, problem_samples = scan_samples()


    samples_a = set((r[0], r[1]) for r in results_a)
    samples_b = set((r[0], r[1]) for r in results_b)
    overlap = samples_a & samples_b
    unique_problems = len(problem_samples)

    print("=== Data quality screening report ===")
    print(f"Total checked: {total_checked}")
    print(f"Issue A (always visible but outside FOV): {len(samples_a)}")
    print(f"Issue B (implausible large displacement): {len(samples_b)}")
    print(f"Overlap (both A and B): {len(overlap)}")
    print(f"Unique samples to quarantine: {unique_problems}")
    print()
    print("Detailed list:")

    for short_label, sample_name, detail in sorted(results_a):
        print(f"[A] {short_label}/{sample_name}: {detail}")
    for short_label, sample_name, detail in sorted(results_b):
        print(f"[B] {short_label}/{sample_name}: {detail}")

    print()
    print("--- Quarantining samples ---")
    moved = quarantine_samples(problem_samples)
    print(f"Moved {moved} problematic samples to {QUARANTINE_DIR}")

    remaining = total_checked - moved
    need_supplement = max(0, 1000 - remaining)
    print()
    print("--- Statistics ---")
    print(f"Quarantined samples: {moved}")
    print(f"Remaining valid samples: {remaining}")
    print(f"Additional samples needed to reach 1000: {need_supplement}")


if __name__ == "__main__":
    main()
