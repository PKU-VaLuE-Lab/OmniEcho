import argparse
from collections import Counter
import gzip
import json
from pathlib import Path, PurePosixPath
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nav"))
from internnav.benchmark_paths import NAV_DIRECTORY, NAV_SCENES_RELATIVE, recording_relative_path


def read_inventory(root, listing=None, oss_prefix=None):
    if listing is None:
        return {path.relative_to(root).as_posix(): path.stat().st_size
                for path in root.rglob("*") if path.is_file()}
    prefix = oss_prefix.rstrip("/") + "/"
    inventory = {}
    for line in listing.read_text().splitlines():
        if prefix not in line:
            continue
        attributes, relative = line.split(prefix, 1)
        relative = relative.strip()
        if not relative or relative.endswith("/"):
            continue
        fields = attributes.split()
        if len(fields) < 7:
            raise ValueError("Use the complete long-format output of ossutil ls.")
        inventory[relative] = int(fields[4])
    if not inventory:
        raise ValueError("No objects matched the supplied OSS prefix.")
    return inventory


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def scene_key(value):
    return PurePosixPath(value).stem


def check_benchmark(root, inventory):
    errors = []
    required = set()
    missing = set()
    empty = set()
    report = {
        "scope": "Manifest/schema and nonempty-object checks; no media decoding or closed-loop evaluation.",
        "inventory_files": len(inventory),
        "inventory_bytes": sum(inventory.values()),
        "qa": {},
        "navigation": {},
        "errors": errors,
    }

    def require(relative):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"Invalid benchmark-relative path: {relative}")
            return False
        key = path.as_posix()
        required.add(key)
        if key not in inventory:
            missing.add(key)
            return False
        if inventory[key] <= 0:
            empty.add(key)
            return False
        return True

    def load(relative, jsonl=False, compressed=False):
        if not require(relative):
            return None
        try:
            if jsonl:
                return read_jsonl(root / relative)
            if compressed:
                with gzip.open(root / relative, "rt", encoding="utf-8") as stream:
                    return json.load(stream)
            return json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            errors.append(f"Cannot read metadata {relative}: {error}")
            return None

    qa_rows = load("OmniEchoBench-QA/OmniEchoBench-QA.jsonl", jsonl=True)
    if qa_rows is not None:
        media = set()
        categories = Counter()
        for index, row in enumerate(qa_rows):
            try:
                messages = row["messages"]
                user = next(message for message in messages if message["role"] == "user")
                answer = next(message["content"] for message in messages if message["role"] == "assistant")
                if not answer:
                    raise ValueError("empty reference answer")
                parts = user["content"]
                if not isinstance(parts, list):
                    raise ValueError("user content must be a list")
                media_parts = [part for part in parts if part.get("type") in {"audio", "video", "image"}]
                if not media_parts:
                    raise ValueError("no media references")
                for part in media_parts:
                    relative = part[part["type"]]
                    if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or "://" in relative:
                        raise ValueError("media references must be local relative paths")
                    key = "OmniEchoBench-QA/" + relative
                    media.add(key)
                    require(key)
                category = row.get("meta", {}).get("category")
                if category is None:
                    kinds = {part["type"] for part in media_parts}
                    if not {"audio", "image"}.issubset(kinds) or answer.strip() not in "ABCD" or len(answer.strip()) != 1:
                        raise ValueError("unrecognized metadata-free cognitive-map record")
                    category = "cognitive_map"
                categories[category] += 1
            except (KeyError, TypeError, ValueError, StopIteration, AttributeError) as error:
                errors.append(f"QA row {index}: {error}")
        if not qa_rows:
            errors.append("The QA manifest is empty.")
        report["qa"] = {
            "records": len(qa_rows),
            "categories": dict(categories),
            "unique_media": len(media),
            "media_by_extension": dict(Counter(PurePosixPath(key).suffix for key in media)),
            "unreferenced_media": sorted(key for key in inventory
                                         if key.startswith("OmniEchoBench-QA/")
                                         and PurePosixPath(key).suffix in {".wav", ".png", ".mp4"}
                                         and key not in media),
        }

    scenes_prefix = f"{NAV_DIRECTORY}/{NAV_SCENES_RELATIVE}"
    all_recordings = set()
    all_sources = set()
    all_scenes = set()
    for split in ("case3_vln", "case3_smoke"):
        episode_path = f"{scenes_prefix}/episodes/{split}.json.gz"
        sidecar_path = f"{scenes_prefix}/episodes/{split}_audio_sidecar.jsonl"
        dataset = load(episode_path, compressed=True)
        sidecars = load(sidecar_path, jsonl=True)
        if dataset is None or sidecars is None:
            continue
        episodes = dataset.get("episodes", [])
        if not episodes or not sidecars:
            errors.append(f"{split}: empty episodes or audio sidecar")
        episode_keys = []
        sidecar_keys = []
        scene_paths = set()
        recordings = set()
        legacy_paths = 0
        for episode in episodes:
            try:
                episode_keys.append((scene_key(episode["scene_id"]), int(episode["episode_id"])))
                relative = episode["scene_id"]
                if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                    raise ValueError("scene_id must be relative to the scene directory")
                scene_paths.add(relative)
                if len(episode["start_position"]) != 3 or len(episode["start_rotation"]) != 4:
                    raise ValueError("invalid start pose dimensions")
                if not episode["goals"] or len(episode["goals"][0]["position"]) != 3:
                    raise ValueError("missing three-dimensional goal")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{split} episode: {error}")
        for row in sidecars:
            try:
                sidecar_keys.append((scene_key(row["scene_id"]), int(row["episode_id"])))
                matrix = row["coordinate_system"]["forward"]["A"]
                if len(matrix) != 2 or any(len(values) != 2 for values in matrix):
                    raise ValueError("coordinate transform must be 2x2")
                if abs(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) < 1e-9:
                    raise ValueError("singular coordinate transform")
                if not row["recordings"] or not row["task_instruction"].strip():
                    raise ValueError("empty recordings or task instruction")
                if row.get("n_recordings", len(row["recordings"])) != len(row["recordings"]):
                    raise ValueError("n_recordings differs from the recording list")
                source = f"{NAV_DIRECTORY}/audio/case3/{row['scene_id']}/source_audio/{row['sample_id']}.wav"
                require(source)
                all_sources.add(source)
                for recording in row["recordings"]:
                    if len(recording["glb"]) != 3:
                        raise ValueError("recording position must be three-dimensional")
                    original = recording["audio_path"]
                    relative = recording_relative_path(original)
                    if relative.is_absolute():
                        raise ValueError(f"unrecognized absolute recording path: {original}")
                    key = f"{NAV_DIRECTORY}/{relative}"
                    require(key)
                    recordings.add(key)
                    legacy_paths += int(PurePosixPath(original).is_absolute())
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                errors.append(f"{split} sidecar row: {error}")
        for label, keys in (("episode", episode_keys), ("sidecar", sidecar_keys)):
            duplicates = [key for key, count in Counter(keys).items() if count > 1]
            if duplicates:
                errors.append(f"{split}: duplicate {label} keys: {duplicates}")
        unmatched = set(episode_keys) ^ set(sidecar_keys)
        if unmatched:
            errors.append(f"{split}: unmatched episode/sidecar keys: {sorted(unmatched)}")
        for relative in scene_paths:
            scene = PurePosixPath(relative)
            parent = f"{scenes_prefix}/{scene.parent}"
            stem = scene.stem
            require(f"{scenes_prefix}/{scene}")
            require(f"{parent}/{stem}.navmesh")
            require(f"{parent}/{stem}_navproxy.glb")
            load(f"{parent}/nav_meta.json")
            for variant in (stem, stem + "_navproxy"):
                config = load(f"{parent}/{variant}.stage_config.json")
                if config is not None:
                    if not config.get("render_asset"):
                        errors.append(f"{parent}/{variant}: missing render_asset")
                    for field in ("render_asset", "collision_asset", "nav_asset"):
                        if config.get(field):
                            require(f"{parent}/{config[field]}")
        all_recordings.update(recordings)
        all_scenes.update(scene_paths)
        report["navigation"][split] = {
            "episodes": len(episodes),
            "sidecars": len(sidecars),
            "scenes": len(scene_paths),
            "recordings": len(recordings),
            "legacy_absolute_audio_references": legacy_paths,
            "task_types": dict(Counter(row.get("task_type", "unknown") for row in sidecars)),
        }
    report["navigation"]["unique_recordings"] = len(all_recordings)
    report["navigation"]["unique_source_audio"] = len(all_sources)
    report["navigation"]["unique_scenes"] = len(all_scenes)
    report["missing_files"] = sorted(missing)
    report["empty_required_files"] = sorted(empty)
    report["required_files"] = len(required)
    report["passed"] = not (errors or missing or empty)
    return report


def main():
    parser = argparse.ArgumentParser(description="Check the published OmniEchoBench QA and NAV assets.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Local benchmark mirror, or downloaded metadata when using --oss-list.")
    parser.add_argument("--oss-list", type=Path, help="Complete long-format ossutil ls output.")
    parser.add_argument("--oss-prefix", help="The OSS prefix represented by --root, including the bucket.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if bool(args.oss_list) != bool(args.oss_prefix):
        parser.error("--oss-list and --oss-prefix must be supplied together")
    inventory = read_inventory(args.root, args.oss_list, args.oss_prefix)
    report = check_benchmark(args.root, inventory)
    report["inventory_source"] = args.oss_prefix or str(args.root.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
