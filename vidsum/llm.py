"""OpenAI 兼容 chat/completions 客户端。

只用标准库 urllib 实现：总结这一步本身就是网络 IO，
不值得为此引入 requests；同时也让「只装 faster-whisper」成为可能。
"""

from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .util import log


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: float = 300.0
    retries: int = 4

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model)


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# 推理模型（step-5、o1、deepseek-r1 等）把 token 先花在思考上，思考长度不可
# 预测：预算吃满时要么正文为空，要么正文被截断、思考过程漏进 content。两种
# 都靠加大 max_tokens 重试救回来；次数与上限收着，避免失控。
_BUDGET_GROW_LIMIT = 2
_BUDGET_GROW_CAP = 32768


def _sleep(attempt: int) -> None:
    time.sleep(min(2.0 ** attempt, 30.0) + random.uniform(0, 1.0))


def _extract_content(data: dict) -> tuple[str, str]:
    """从 chat/completions 响应里取出 (正文, finish_reason)。

    content 为分段列表时（少数网关的实现）把各段文本拼起来。
    reasoning_content 装的是思考过程而不是答案，故意不拿它当正文。
    """
    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    elif not isinstance(content, str):
        content = "" if content is None else str(content)
    return content, str(choice.get("finish_reason") or "")


def _empty_error(finish_reason: str, tried: list[int]) -> LLMError:
    detail = (f"LLM 返回了空内容（finish_reason={finish_reason or '未提供'}，"
              f"max_tokens 依次用过 {'/'.join(str(t) for t in tried)}）")
    if finish_reason == "length":
        return LLMError(
            detail + "。该模型可能是推理模型，思考耗尽了全部 token；"
            "请调大 --llm-max-tokens（网页控制台里是「token 上限」）后重跑，"
            "或换用非推理模型")
    return LLMError(detail + "。请重试或更换模型")


def _chat_once(config: LLMConfig, messages: list[dict], max_tokens: int) -> str:
    """固定 max_tokens 的一轮请求，返回 (正文, finish_reason)。

    网络类失败按指数退避重试；拿到响应就原样交出，截断与否由上层判断。
    """
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
    }
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    last_error: Exception | None = None

    for attempt in range(1, config.retries + 1):
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST"
        )
        request.add_header("Content-Type", "application/json")
        if config.api_key:
            request.add_header("Authorization", f"Bearer {config.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_content(data)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-400:]
            last_error = LLMError(f"LLM 接口返回 {exc.code}: {detail}")
            if exc.code not in _RETRYABLE_STATUS:
                raise last_error from exc
        except (urllib.error.URLError, socket.timeout, OSError, ValueError) as exc:
            last_error = LLMError(f"LLM 请求失败: {exc}")
        if attempt < config.retries:
            _sleep(attempt)

    raise LLMError(f"LLM 重试 {config.retries} 次后仍失败: {last_error}")


def chat(config: LLMConfig, messages: list[dict], *,
         max_tokens: int | None = None) -> str:
    """发送一轮对话，返回助手文本内容。网络失败按指数退避重试。

    finish_reason=length 说明预算被吃满：正文可能为空，也可能被截断且混进
    思考过程。两种都加大 max_tokens 重试；加到顶仍截断就交出手头最好的正文，
    仍为空则抛出带预算轨迹的报错。
    """
    budget = max_tokens or config.max_tokens
    tried = [budget]
    best = ""
    while True:
        content, finish_reason = _chat_once(config, messages, budget)
        if content.strip() and finish_reason != "length":
            return content
        if content.strip():
            best = content
        if (finish_reason != "length" or len(tried) > _BUDGET_GROW_LIMIT
                or budget >= _BUDGET_GROW_CAP):
            break
        budget = min(budget * 2, _BUDGET_GROW_CAP)
        log(f"[LLM] 预算 {tried[-1]} 不够（finish_reason=length），"
            f"加大到 {budget} 重试")
        tried.append(budget)

    if best:
        return best
    raise _empty_error(finish_reason, tried)


def resolve_config(args) -> LLMConfig:
    """优先级：命令行 > 环境变量 > 默认值。"""
    import os

    def pick(flag, *envs, default=""):
        if flag:
            return flag
        for env in envs:
            value = os.environ.get(env)
            if value:
                return value
        return default

    return LLMConfig(
        base_url=pick(getattr(args, "llm_base_url", None),
                      "VIDSUM_LLM_BASE_URL", "OPENAI_BASE_URL",
                      default="https://api.openai.com/v1"),
        api_key=pick(getattr(args, "llm_api_key", None),
                     "VIDSUM_LLM_API_KEY", "OPENAI_API_KEY"),
        model=pick(getattr(args, "llm_model", None),
                   "VIDSUM_LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini"),
        temperature=float(getattr(args, "llm_temperature", 0.3) or 0.3),
        max_tokens=int(getattr(args, "llm_max_tokens", 0) or 2048),
    )
