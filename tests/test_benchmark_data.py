import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nav"))
from internnav import benchmark_paths

SPEC = importlib.util.spec_from_file_location("check_benchmark_data", ROOT / "scripts/check_benchmark_data.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def benchmark_env(monkeypatch, tmp_path):
    for name in ("OMNIECHO_NAV_EP_DIR", "OMNIECHO_NAV_SCENES_DIR", "OMNIECHO_NAV_AUDIO_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMNIECHO_BENCH_ROOT", str(tmp_path))
    return tmp_path


def test_published_navigation_directories(benchmark_env):
    expected = benchmark_env / "OmniEchoBench-NAV"
    assert benchmark_paths.nav_root() == expected
    assert benchmark_paths.scenes_directory() == expected / "scene_data/case3_ce"
    assert benchmark_paths.episodes_directory() == expected / "scene_data/case3_ce/episodes"
    assert benchmark_paths.audio_directory() == expected
    assert benchmark_paths.DEFAULT_SPLIT == "case3_vln"


def test_qa_defaults_use_published_layout(benchmark_env, monkeypatch):
    import ast
    import os

    monkeypatch.delenv("OMNIECHO_QA_GOLD", raising=False)
    expected = str(benchmark_env / "OmniEchoBench-QA/OmniEchoBench-QA.jsonl")
    for filename, target in (("evaluate/infer_qa.py", "DEFAULT_QA_GOLD"),
                             ("qa_eval/eval_qa_metrics.py", "VAL")):
        path = ROOT / filename
        tree = ast.parse(path.read_text())
        names = {"REPO_ROOT", "HERE", "BENCH_ROOT", target}
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(name, ast.Name) and name.id in names for name in node.targets)]
        scope = {"os": os, "__file__": str(path)}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), scope)
        assert scope[target] == expected


@pytest.mark.parametrize("prefix", ["", "OmniEchoBench-NAV/", "/legacy/storage/case3_nav/"])
def test_recordings_relocate_to_published_layout(benchmark_env, prefix):
    relative = "audio/case3/scene_003/sample_001/rec_pos_001.wav"
    actual = benchmark_paths.resolve_recording_path(prefix + relative)
    assert actual == str(benchmark_env / "OmniEchoBench-NAV" / relative)


def test_explicit_audio_root_override(benchmark_env, monkeypatch):
    override = benchmark_env / "override"
    monkeypatch.setenv("OMNIECHO_NAV_AUDIO_ROOT", str(override))
    relative = "audio/case3/scene_003/sample_001/rec_pos_001.wav"
    assert benchmark_paths.resolve_recording_path("/legacy/" + relative) == str(override / relative)


def test_recording_traversal_is_rejected(benchmark_env):
    with pytest.raises(ValueError, match="traversal"):
        benchmark_paths.resolve_recording_path("/legacy/audio/case3/../../outside.wav")


def test_external_absolute_recording_remains_supported(benchmark_env):
    assert benchmark_paths.resolve_recording_path("/custom/recording.wav") == "/custom/recording.wav"


def write_fixture(root):
    def write(relative, data=b"asset"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    qa = {"messages": [
        {"role": "user", "content": [{"type": "video", "video": "videos/vid_0001.mp4"}]},
        {"role": "assistant", "content": "left"}], "meta": {"category": "direction"}}
    write("OmniEchoBench-QA/OmniEchoBench-QA.jsonl", (json.dumps(qa) + "\n").encode())
    write("OmniEchoBench-QA/videos/vid_0001.mp4")
    nav = "OmniEchoBench-NAV"
    scene = "case3/scene_003/scene_003.glb"
    scenes = f"{nav}/scene_data/case3_ce"
    episode = {"scene_id": scene, "episode_id": 0, "start_position": [0, 0, 0],
               "start_rotation": [0, 0, 0, 1], "goals": [{"position": [1, 0, 1]}]}
    recording = "audio/case3/scene_003/sample_001/rec_pos_001.wav"
    sidecar = {"scene_id": "scene_003", "episode_id": 0, "sample_id": "sample_001",
               "task_instruction": "Find the sound.", "task_type": "sound",
               "coordinate_system": {"forward": {"A": [[1, 0], [0, 1]]}},
               "recordings": [{"glb": [0, 0, 0], "audio_path": "/legacy/" + recording}]}
    for split in ("case3_vln", "case3_smoke"):
        write(f"{scenes}/episodes/{split}.json.gz", gzip.compress(json.dumps({"episodes": [episode]}).encode()))
        write(f"{scenes}/episodes/{split}_audio_sidecar.jsonl", (json.dumps(sidecar) + "\n").encode())
    for variant in ("scene_003", "scene_003_navproxy"):
        write(f"{scenes}/case3/scene_003/{variant}.glb")
        write(f"{scenes}/case3/scene_003/{variant}.stage_config.json",
              json.dumps({"render_asset": f"{variant}.glb"}).encode())
    write(f"{scenes}/case3/scene_003/nav_meta.json", b"{}")
    write(f"{scenes}/case3/scene_003/scene_003.navmesh")
    write(f"{nav}/{recording}")
    write(f"{nav}/audio/case3/scene_003/source_audio/sample_001.wav")
    return root


def test_audit_complete_fixture(tmp_path):
    root = write_fixture(tmp_path)
    report = audit.check_benchmark(root, audit.read_inventory(root))
    assert report["passed"], report
    assert report["qa"]["records"] == 1
    assert report["navigation"]["case3_vln"]["legacy_absolute_audio_references"] == 1
    assert report["required_files"] == report["inventory_files"]


def test_audit_missing_and_empty_media(tmp_path):
    root = write_fixture(tmp_path)
    missing = "OmniEchoBench-QA/videos/vid_0001.mp4"
    empty = "OmniEchoBench-NAV/audio/case3/scene_003/sample_001/rec_pos_001.wav"
    (root / missing).unlink()
    (root / empty).write_bytes(b"")
    report = audit.check_benchmark(root, audit.read_inventory(root))
    assert not report["passed"]
    assert report["missing_files"] == [missing]
    assert report["empty_required_files"] == [empty]


def test_audit_rejects_duplicate_sidecars(tmp_path):
    root = write_fixture(tmp_path)
    path = root / "OmniEchoBench-NAV/scene_data/case3_ce/episodes/case3_vln_audio_sidecar.jsonl"
    path.write_text(path.read_text() * 2)
    report = audit.check_benchmark(root, audit.read_inventory(root))
    assert not report["passed"]
    assert any("duplicate sidecar" in error for error in report["errors"])


def test_oss_inventory_excludes_directory_markers(tmp_path):
    listing = tmp_path / "listing.txt"
    attributes = "2026-10-18 00:00:00 +0800 CST {size} Standard ETAG "
    prefix = "oss://example/OmniEchoBench/"
    listing.write_text(attributes.format(size=0) + prefix + "OmniEchoBench-NAV/\n" +
                       attributes.format(size=123) + prefix + "OmniEchoBench-QA/video.mp4\n")
    assert audit.read_inventory(tmp_path, listing, prefix) == {"OmniEchoBench-QA/video.mp4": 123}


def test_splitter_uses_published_default_layout(benchmark_env):
    root = write_fixture(benchmark_env)
    result = subprocess.run([sys.executable, str(ROOT / "nav/scripts/eval/spatial_audio/split_nway.py"),
                             "--n", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / "OmniEchoBench-NAV/scene_data/case3_ce/episodes/case3_smoke_g0.json.gz").is_file()
