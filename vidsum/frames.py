"""画面帧的规划、清单与多模态输入。

音频管线只"听"视频；这里补上"看"的部分：按分段计划在每段内均匀
取几个时间点抽 JPEG 帧（执行在 media/ffmpeg 侧），落盘成 frames/
目录并用 frames.json 记录清单，断点续跑时直接复用。

帧有两个用途：
- 直接插图进终稿附录，把文字总结升级成图文笔记；
- 作为多模态 LLM 的输入，让滚动摘要吸收幻灯片、板书、图表等
  只存在于画面里的信息。
"""

from __future__ import annotations

import base64
from pathlib import Path

from .util import read_json, write_json

MANIFEST_FILE = "frames.json"
FRAMES_DIR = "frames"


def plan_frame_times(segments, per_segment: int) -> dict[int, list[float]]:
    """每段内均匀取 per_segment 个时间点，避开首尾各 1 秒。

    避开首尾是因为切点常落在静音/切换处，那里的画面没有信息量；
    段长不足 2 秒时该段不取帧。
    """
    count = max(0, int(per_segment))
    plan: dict[int, list[float]] = {}
    for seg in segments:
        if count == 0 or seg.duration <= 2.0:
            plan[seg.index] = []
            continue
        inner_start = seg.start + 1.0
        inner_end = seg.end - 1.0
        if count == 1:
            times = [(inner_start + inner_end) / 2.0]
        else:
            step = (inner_end - inner_start) / count
            times = [inner_start + step * (i + 0.5) for i in range(count)]
        plan[seg.index] = [round(t, 3) for t in times]
    return plan


def frame_relpath(index: int, seconds: float) -> str:
    """帧在工作目录内的相对路径，用 POSIX 分隔符（Markdown 插图用）。

    文件名即帧在视频中的秒数（08.3f），frame_time() 可从中还原时间戳，
    附录图注不必另存元数据。
    """
    return f"{FRAMES_DIR}/{index:03d}/{seconds:08.3f}.jpg"


def frame_path(work: Path, index: int, seconds: float) -> Path:
    return Path(work) / frame_relpath(index, seconds)


def frame_time(rel: str) -> float:
    """从文件名还原时间戳（f"{t:08.3f}.jpg"）；解析不了返回 0。"""
    try:
        return float(Path(rel).stem)
    except ValueError:
        return 0.0


def load_manifest(work: Path) -> dict:
    """读 frames.json：{"width": int, "frames": {index: [rel, ...]}}。"""
    data = read_json(Path(work) / MANIFEST_FILE) or {}
    frames = data.get("frames") or {}
    return {
        "width": int(data.get("width") or 0),
        "frames": {int(k): [str(p) for p in v] for k, v in frames.items()},
    }


def save_manifest(work: Path, frames: dict[int, list[str]], *,
                  width: int) -> None:
    write_json(Path(work) / MANIFEST_FILE, {
        "width": int(width),
        "frames": {str(k): list(v) for k, v in sorted(frames.items())},
    })


def image_part(path: Path) -> dict:
    """把 JPEG 读成 OpenAI 兼容的 image_url content part（base64 data URI）。"""
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
