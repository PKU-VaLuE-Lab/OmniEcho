import json

import httpx
import pytest

from omniecho_audio import api_client, video_prompt


def test_chat_uses_configured_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "OmniEcho")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "left"}}]})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(api_client.httpx, "Client", lambda **kwargs: client)
    result = api_client.call_llm_with_system("Return a direction", "Where is the sound?")
    assert result == "left"
    assert str(requests[0].url) == "https://example.com/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "OmniEcho"
    assert payload["messages"][0]["role"] == "system"
    assert "contents" not in payload


def test_missing_key_fails_without_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(api_client.APIPermanentError):
        api_client.call_llm_with_system("system", "user")


def test_video_adapter_contract(monkeypatch):
    monkeypatch.setenv("VIDEO_API_KEY", "test-key")
    monkeypatch.setenv("VIDEO_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("VIDEO_MODEL", "OmniEcho")

    def submit(url, *, json, headers, timeout):
        assert url == "https://example.com/v1/videos"
        assert json["prompt"] == "A sounding object"
        assert json["duration"] == 6
        assert headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"id": "job-1"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(video_prompt.httpx, "post", submit)
    assert video_prompt.submit_video_job("A sounding object") == "job-1"


def test_failed_video_is_not_success(monkeypatch):
    monkeypatch.setenv("VIDEO_API_KEY", "test-key")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"status": "failed", "error": "generation failed"})))
    monkeypatch.setattr(video_prompt.httpx, "Client", lambda **kwargs: client)
    with pytest.raises(video_prompt.SubVideoTaskFailed):
        video_prompt.poll_video_result("job-1", timeout=1, poll_interval=0)
