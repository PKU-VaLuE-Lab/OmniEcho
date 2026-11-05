import gzip
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPLITTER = ROOT / "nav/scripts/eval/spatial_audio/split_nway.py"
MERGER = ROOT / "nav/scripts/eval/spatial_audio/merge_audio_parallel.py"


def write_episodes(directory, include_sidecar=True):
    episodes = [{"scene_id": f"scene_{number:03d}/scene_{number:03d}.glb", "episode_id": number}
                for number in range(4)]
    with gzip.open(directory / "sample.json.gz", "wt") as stream:
        json.dump({"episodes": episodes}, stream)
    if include_sidecar:
        with (directory / "sample_audio_sidecar.jsonl").open("w") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode) + "\n")


def test_split_preserves_all_episodes_and_sidecars(tmp_path):
    write_episodes(tmp_path)
    result = subprocess.run([sys.executable, str(SPLITTER), "--split", "sample", "--n", "2",
                             "--ep_dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    total = 0
    for shard in range(2):
        with gzip.open(tmp_path / f"sample_g{shard}.json.gz", "rt") as stream:
            episodes = json.load(stream)["episodes"]
        sidecar = (tmp_path / f"sample_g{shard}_audio_sidecar.jsonl").read_text().splitlines()
        assert len(episodes) == len(sidecar) == 2
        total += len(episodes)
    assert total == 4


def test_split_rejects_missing_sidecar(tmp_path):
    write_episodes(tmp_path, include_sidecar=False)
    result = subprocess.run([sys.executable, str(SPLITTER), "--split", "sample", "--n", "2",
                             "--ep_dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing audio sidecar" in result.stderr
    assert not (tmp_path / "sample_g0.json.gz").exists()


def test_merge_refuses_partial_results(tmp_path):
    result = subprocess.run([sys.executable, str(MERGER), "--split", "sample", "--n", "2",
                             "--out_root", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "sample/audio/result.json").exists()


def test_merge_weights_by_episode_count(tmp_path):
    for shard, count, success in [(0, 1, 1.0), (1, 3, 0.0)]:
        directory = tmp_path / f"sample_g{shard}" / "audio"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({
            "length": count, "success": success, "spl": success,
            "oracle_success": success, "distance_to_goal": 1.0, "trajectory_length": 2.0}))
    result = subprocess.run([sys.executable, str(MERGER), "--split", "sample", "--n", "2",
                             "--out_root", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    merged = json.loads((tmp_path / "sample/audio/result.json").read_text())
    assert merged["success"] == 0.25
    assert merged["length"] == 4
