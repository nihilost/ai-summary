"""跨模块共用的小工具。"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(text: str, *, maxlen: int = 80) -> str:
    """把任意字符串净化成可用的文件/目录名。"""
    cleaned = _ILLEGAL.sub("_", str(text)).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:maxlen].rstrip(" .") or "untitled"


_LOG_SINKS: list = []


def add_log_sink(sink) -> None:
    """注册一个额外的日志出口（如 Web 控制台按任务收集实时日志）。"""
    _LOG_SINKS.append(sink)


def remove_log_sink(sink) -> None:
    try:
        _LOG_SINKS.remove(sink)
    except ValueError:
        pass


def log(message: str) -> None:
    """带时间戳的单行进度输出，立即刷新以便重定向到日志。"""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    for sink in list(_LOG_SINKS):
        try:
            sink(line)
        except Exception:
            pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """先写临时文件再替换，避免中断时留下半个文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    # Windows 上目标文件会被杀毒/索引进程短暂锁定，os.replace 偶发
    # 拒绝访问；退避重试几次即可，别让一次存盘掀翻整轮总结。
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"
