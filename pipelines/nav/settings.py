import os
from pathlib import Path

DATASET = "OmniEchoBench-Nav"
NAV_ROOT = os.environ.get("OMNIECHO_NAV_OUTPUT", str(Path.cwd() / "outputs" / DATASET))
DATA_ROOT = os.environ.get("OMNIECHO_TRAJECTORY_ROOT", "/path/to/trajectories")
MODEL_ROOT = os.environ.get("OMNIECHO_AUX_MODEL_ROOT", "/path/to/auxiliary-models")
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.example.com/v1")
API_MODEL = os.environ.get("OPENAI_MODEL", "OmniEcho")
