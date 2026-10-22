"""Markdown 输出渲染。

三个文件承载「边处理边输出」：
- transcript.md 每转写完一段就重写，用户可以看着稿子变长；
- summary.md 每并入一段就重写，滚动摘要持续增长；
- final.md 在收尾一轮后生成，附带分段索引。
"""

from __future__ import annotations

import time
from pathlib import Path

from .asr import Utterance
from .segment import fmt_range, fmt_ts
from .summarize import SummaryState
from .util import atomic_write_text

from . import frames as frames_mod


def _utterance_lines(utterances, offset: float):
    """把一组 utterances 渲染成带绝对时间戳的 Markdown 行。"""
    for utt in utterances:
        yield f"**[{fmt_ts(offset + utt.start, always_hours=True)}]** {utt.text}"
        yield ""


def render_progress(title: str, state: SummaryState, *, total: int,
                     source: str = "") -> str:
    done = state.next_index
    lines = [
        f"# {title}",
        "",
        f"> 滚动摘要进行中：{done}/{total} 段已完成 · "
        f"已覆盖 {fmt_ts(max((n.end for n in state.notes), default=0.0))} · "
        f"更新于 {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if source:
        lines.append(f"> 来源：`{source}`")
    lines += ["", "## 分段大纲", ""]
    if state.notes:
        lines += [f"- {fmt_range(n.start, n.end)} {n.heading}"
                  for n in state.notes]
    else:
        lines.append("- （尚无已完成的分段）")
    lines += ["", "## 当前滚动摘要", "",
              state.running.strip() or "（尚未生成，请稍候）", ""]
    return "\n".join(lines)


def write_progress(work_dir: Path, title: str, state: SummaryState, *,
                   total: int, source: str = "") -> Path:
    path = Path(work_dir) / "summary.md"
    atomic_write_text(path, render_progress(
        title, state, total=total, source=source))
    return path


def render_transcript(title: str, utterances: list[Utterance]) -> str:
    lines = [f"# {title} · 完整转写", ""]
    body = list(_utterance_lines(utterances, 0.0))
    lines += body or ["（没有转写结果）", ""]
    return "\n".join(lines)


def render_transcript_partial(title: str, chunks: list[ChunkResult], *,
                               total: int) -> str:
    """转写进行中的版本：已完成的段按序拼接，让用户能看着稿子变长。"""
    done = sorted(chunks, key=lambda c: c.index)
    lines = [f"# {title} · 完整转写", "",
             f"> 转写进行中：{len(done)}/{total} 段已完成", ""]
    body = []
    for chunk in done:
        body.extend(_utterance_lines(chunk.utterances, chunk.start))
    lines += body or ["（等待转写结果…）", ""]
    return "\n".join(lines)


def write_transcript_partial(work_dir: Path, title: str, chunks, *,
                             total: int) -> Path:
    path = Path(work_dir) / "transcript.md"
    atomic_write_text(
        path, render_transcript_partial(title, chunks, total=total))
    return path


def write_transcript(work_dir: Path, title: str,
                     utterances: list[Utterance]) -> Path:
    path = Path(work_dir) / "transcript.md"
    atomic_write_text(path, render_transcript(title, utterances))
    return path


def render_final(title: str, state: SummaryState, final_md: str,
                 frames: dict[int, list[str]] | None = None) -> str:
    lines = [f"# {title}", ""]
    lines.append(final_md.strip() or "（未生成终稿）")
    lines += ["", "---", "", "## 附录：分段索引", ""]
    frames = frames or {}
    if state.notes:
        for n in state.notes:
            lines.append(f"- {fmt_range(n.start, n.end)} {n.heading}")
            for rel in frames.get(n.index, []):
                lines.append(f"  - ![第 {n.index + 1} 段画面 "
                             f"{fmt_ts(frames_mod.frame_time(rel))}]({rel})")
    else:
        lines.append("- （无）")
    return "\n".join(lines) + "\n"


def write_final(work_dir: Path, title: str, state: SummaryState,
                final_md: str, *, frames: dict[int, list[str]] | None = None) -> Path:
    path = Path(work_dir) / "final.md"
    atomic_write_text(path, render_final(title, state, final_md,
                                         frames=frames))
    return path
