from __future__ import annotations

import os
import time

import httpx


class APIPermanentError(RuntimeError):
    pass


def close_api_client() -> None:
    return None


def call_llm(contents, *, system_instruction=None, max_output_tokens=16384,
             temperature=0.7, timeout=300.0, max_retries=5):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise APIPermanentError("Set OPENAI_API_KEY before requesting annotations.")
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    for item in contents:
        content = []
        for part in item.get("parts", []):
            if "text" in part:
                content.append({"type": "text", "text": part["text"]})
            elif "inline_data" in part or "inlineData" in part:
                media = part.get("inline_data", part.get("inlineData"))
                mime = media.get("mime_type", media.get("mimeType"))
                if not mime or not mime.startswith("image/"):
                    raise ValueError("The annotation API adapter accepts text and images only.")
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{media['data']}"}})
            else:
                raise ValueError("Unsupported annotation content part.")
        role = "assistant" if item.get("role") == "model" else item.get("role", "user")
        messages.append({"role": role, "content": content})
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "OmniEcho"),
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }
    url = os.environ.get("OPENAI_BASE_URL", "https://api.example.com/v1").rstrip("/")
    last_error = None
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max(1, max_retries)):
            try:
                response = client.post(url + "/chat/completions", json=payload,
                                       headers={"Authorization": f"Bearer {api_key}"})
                if 400 <= response.status_code < 500 and response.status_code not in {408, 429}:
                    raise APIPermanentError(f"Annotation API rejected request: HTTP {response.status_code}")
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"]
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("Annotation API returned empty or non-text content.")
                return result
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
                last_error = error
                if attempt + 1 < max_retries:
                    time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Annotation request failed after retries.") from last_error


def call_llm_with_system(system_prompt, user_prompt, **kwargs):
    return call_llm([{"role": "user", "parts": [{"text": user_prompt}]}],
                    system_instruction=system_prompt, **kwargs)
