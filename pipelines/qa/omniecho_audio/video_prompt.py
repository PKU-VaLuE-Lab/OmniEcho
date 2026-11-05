from __future__ import annotations

import os
from pathlib import Path
import time
from urllib.parse import quote, urlparse

import httpx


class SubVideoTaskFailed(RuntimeError):
    def __init__(self, task_id, code="", message="", raw=""):
        self.task_id = task_id
        self.code = code
        self.message = message
        self.raw = raw
        super().__init__(f"Video task {task_id} failed: {code} {message}")


def _base_url():
    return os.environ.get("VIDEO_BASE_URL", "https://api.example.com/v1").rstrip("/")


def _headers():
    api_key = os.environ.get("VIDEO_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set VIDEO_API_KEY or OPENAI_API_KEY.")
    return {"Authorization": f"Bearer {api_key}"}


def submit_video_job(prompt: str, duration: int = 6) -> str:
    payload = {"model": os.environ.get("VIDEO_MODEL", "OmniEcho"),
               "prompt": prompt, "duration": int(duration),
               "resolution": os.environ.get("VIDEO_RESOLUTION", "720p"),
               "aspect_ratio": os.environ.get("VIDEO_RATIO", "16:9")}
    response = httpx.post(_base_url() + "/videos", json=payload, headers=_headers(), timeout=60)
    response.raise_for_status()
    task_id = response.json().get("id")
    if not task_id:
        raise ValueError("Video service must return an id field.")
    return str(task_id)


def poll_video_result(task_id: str, timeout: float = 600, poll_interval: float = 10) -> str:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=60) as client:
        while time.monotonic() < deadline:
            response = client.get(_base_url() + "/videos/" + quote(task_id, safe=""), headers=_headers())
            response.raise_for_status()
            result = response.json()
            status = result.get("status", "").lower()
            if status in {"completed", "succeeded"}:
                if not result.get("url"):
                    raise ValueError("Completed video task must contain a downloadable url.")
                return result["url"]
            if status in {"failed", "cancelled"}:
                raise SubVideoTaskFailed(task_id, status, str(result.get("error", "")))
            if status not in {"queued", "pending", "running", "in_progress", "processing"}:
                raise ValueError(f"Unsupported video job status: {status!r}")
            time.sleep(poll_interval)
    raise TimeoutError(f"Video task {task_id} exceeded {timeout} seconds.")


def download_video(url: str, output_path: Path) -> Path:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("Video download URL must use HTTP or HTTPS.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    try:
        with httpx.stream("GET", url, timeout=300, follow_redirects=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        if not temporary.stat().st_size:
            raise ValueError("Video service returned an empty file.")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def generate_video(prompt: str, output_path: Path, max_retries: int = 3) -> Path:
    last_error = None
    for attempt in range(max(1, max_retries)):
        try:
            return download_video(poll_video_result(submit_video_job(prompt)), output_path)
        except (httpx.HTTPError, SubVideoTaskFailed, TimeoutError) as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Video generation failed after retries.") from last_error
