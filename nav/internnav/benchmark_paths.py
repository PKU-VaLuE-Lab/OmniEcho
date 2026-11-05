import os
from pathlib import Path, PurePosixPath


NAV_DIRECTORY = "OmniEchoBench-NAV"
NAV_SCENES_RELATIVE = "scene_data/case3_ce"
DEFAULT_SPLIT = "case3_vln"


def benchmark_root():
    default = Path(__file__).resolve().parents[1] / "data" / "OmniEchoBench"
    return Path(os.environ.get("OMNIECHO_BENCH_ROOT", str(default))).expanduser().resolve()


def nav_root():
    return benchmark_root() / NAV_DIRECTORY


def scenes_directory():
    return Path(os.environ.get("OMNIECHO_NAV_SCENES_DIR", str(nav_root() / NAV_SCENES_RELATIVE)))


def episodes_directory():
    return Path(os.environ.get("OMNIECHO_NAV_EP_DIR", str(scenes_directory() / "episodes")))


def audio_directory():
    return Path(os.environ.get("OMNIECHO_NAV_AUDIO_ROOT", str(nav_root())))


def recording_relative_path(path):
    value = str(path).replace("\\", "/")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"Parent traversal is not allowed in a recording path: {path}")
    marker = "audio/case3/"
    if value.startswith(marker):
        return PurePosixPath(value)
    if "/" + marker in value:
        return PurePosixPath(marker + value.split("/" + marker, 1)[1])
    if value.startswith(NAV_DIRECTORY + "/"):
        return PurePosixPath(value[len(NAV_DIRECTORY) + 1:])
    return PurePosixPath(value)


def resolve_recording_path(path, root=None):
    relative = recording_relative_path(path)
    if relative.is_absolute():
        return str(relative)
    return str((Path(root) if root is not None else audio_directory()) / relative)
