"""OpenAI 兼容 /audio/transcriptions 后端（云端转写，零本地依赖）。"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .asr import ASRBackend, Utterance


class ApiError(RuntimeError):
    pass


def encode_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """手工构造 multipart/form-data，避免引入 requests 依赖。"""
    boundary = "----vidsum" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode("utf-8")
    for name, (filename, data, content_type) in files.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        body += data
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def parse_transcription(payload, *, fallback_text: str = "") -> list[Utterance]:
    """兼容 verbose_json / json / 纯文本三种返回形态。"""
    if isinstance(payload, dict):
        segments = payload.get("segments")
        utterances: list[Utterance] = []
        if isinstance(segments, list):
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                text = str(seg.get("text") or "").strip()
                if not text:
                    continue
                utterances.append(Utterance(
                    float(seg.get("start") or 0.0),
                    float(seg.get("end") or 0.0),
                    text,
                ))
        if utterances:
            return utterances
        text = str(payload.get("text") or "").strip()
        if text:
            return [Utterance(0.0, float(payload.get("duration") or 0.0), text)]
    if fallback_text.strip():
        return [Utterance(0.0, 0.0, fallback_text.strip())]
    return []


class ApiBackend(ASRBackend):
    name = "api"

    def __init__(self, *, base_url: str, api_key: str, model: str,
                 language: str | None = "zh", timeout: float = 600.0,
                 retries: int = 3):
        if not api_key:
            raise ApiError("使用 --asr api 需要提供 API Key（--asr-api-key）")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout = timeout
        self.retries = retries

    def _request(self, audio_path: Path) -> list[Utterance]:
        data = audio_path.read_bytes()
        content_type = mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg"
        fields = {
            "model": self.model,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if self.language:
            fields["language"] = self.language
        body, ctype = encode_multipart(
            fields, {"file": (audio_path.name, data, content_type)}
        )
        request = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=body, method="POST",
        )
        request.add_header("Content-Type", ctype)
        request.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                ctype_header = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-400:]
            raise ApiError(f"转写接口返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"转写接口连接失败: {exc.reason}") from exc

        if "application/json" in ctype_header:
            try:
                return parse_transcription(json.loads(raw))
            except ValueError:
                pass
        return parse_transcription(None, fallback_text=raw)

    def transcribe(self, audio_path: Path) -> list[Utterance]:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return self._request(audio_path)
            except ApiError as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 20))
        raise ApiError(f"转写重试 {self.retries} 次后仍失败: {last}")
