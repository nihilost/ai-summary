"""滚动摘要状态机。

核心约束：送进模型的上下文大小与视频总时长无关。
每一轮只用「已有滚动摘要 + 新一段转写」，因此三小时的视频和十分钟的
视频单轮开销相同。摘要过长时先压缩再更新，这就是层级式合并。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import llm as llm_mod
from . import frames as frames_mod
from .segment import fmt_range
from .util import read_json, write_json

STATE_FILE = "summary_state.json"

SYSTEM_PROMPT = """\
你是一位严谨的课程笔记助手，负责维护一份"滚动摘要"。
每一轮你会收到一段新的课堂转写，请把其中的新信息合并进已有摘要。

硬性要求：
- 只能依据给定材料书写，禁止补充材料里没有的信息，禁止推测和发挥。
- 保留专有名词、公式、结论、数字和易混点，这些是复习时最有价值的内容。
- 使用 Markdown，多用短列表，语言紧凑，不写口语和寒暄。
- 摘要必须能独立阅读，不要出现"本段""如上""前面提到"这类依赖历史的表述。
- 输出格式严格遵守：
  第一行是「## 更新后的滚动摘要」，其后是完整的新摘要；
  最后一行是「## 本段小标题」，其后给一个不超过 20 字的标题。"""

UPDATE_TEMPLATE = """\
【当前滚动摘要】
{summary}

【新增转写片段 {index}/{total}（{span}）】
{text}

请按系统要求输出更新后的滚动摘要和本段小标题。"""

VISION_HINT = (
    "\n另外附带了该段几个关键时间的画面截图，可用于补充幻灯片标题、板书、"
    "图表等只存在于画面里的信息；与转写矛盾时以转写为准。"
)

COMPRESS_TEMPLATE = """\
下面这份滚动摘要太长了。请在不丢失任何专有名词、结论、数字和易混点的前提下，
把它压缩到 {limit} 字以内，保持 Markdown 列表结构。

【滚动摘要】
{summary}"""

FINAL_TEMPLATE = """\
下面是一份由分段滚动摘要累积而成的课程材料，请整理成一份用于考前复习的总结。

要求：
- 只使用给定材料，禁止补充任何外部信息。
- 开头先给"核心结论"（3 到 6 条），再按主题分节展开。
- 保留关键术语、公式、数字、易混点。
- 材料未覆盖的方面不要写，不要编造目录。

【课程标题】{title}
【分段结构】
{outline}

【滚动摘要】
{summary}"""

FINAL_SYSTEM_PROMPT = """\
你是一位严谨的课程笔记助手，负责把一份分段累积的课程材料整理成考前复习用的总结。
只依据给定材料书写，禁止补充材料里没有的信息，禁止推测和发挥。
保留专有名词、公式、结论、数字和易混点。
使用 Markdown，多用短列表，语言紧凑，不写口语和寒暄。
直接输出总结正文：不要复述题目要求，不要写"我将""以下是我整理"这类开场白，
也不要输出"更新后的滚动摘要""本段小标题"这类只属于中间步骤的标题。"""

_SUMMARY_KEYS = ("更新后的滚动摘要", "滚动摘要", "摘要")
_HEADING_KEYS = ("本段小标题", "小标题", "标题")


@dataclass
class ChunkNote:
    index: int
    start: float
    end: float
    heading: str = ""


@dataclass
class SummaryState:
    running: str = ""
    notes: list[ChunkNote] = field(default_factory=list)
    next_index: int = 0
    final: str = ""

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "next_index": self.next_index,
            "final": self.final,
            "notes": [vars(n) for n in self.notes],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "SummaryState":
        data = data or {}
        notes = [ChunkNote(int(n.get("index", 0)), float(n.get("start", 0)),
                           float(n.get("end", 0)), str(n.get("heading", "")))
                 for n in data.get("notes", [])]
        return cls(str(data.get("running", "")), notes,
                   int(data.get("next_index", 0)), str(data.get("final", "")))


def _find_marker(text: str, keys: tuple[str, ...]):
    for key in keys:
        marker = re.search(
            rf"^[#>\s]*{re.escape(key)}\s*[:：]?\s*$", text, re.M
        )
        if marker:
            return marker
    return None


def parse_update(raw: str) -> tuple[str, str]:
    """从模型回复中拆出「滚动摘要」和「本段小标题」，对格式错误保持宽容。"""
    text = (raw or "").strip()
    if not text:
        return "", ""
    summary_marker = _find_marker(text, _SUMMARY_KEYS)
    heading_marker = _find_marker(text, _HEADING_KEYS)

    if summary_marker and heading_marker and summary_marker.start() < heading_marker.start():
        summary = text[summary_marker.end():heading_marker.start()]
        heading = text[heading_marker.end():]
    elif heading_marker:
        summary = text[:heading_marker.start()]
        heading = text[heading_marker.end():]
    elif summary_marker:
        summary = text[summary_marker.end():]
        heading = ""
    else:
        summary, heading = text, ""

    lines = [ln.strip() for ln in heading.strip().splitlines() if ln.strip()]
    return summary.strip(), lines[0].lstrip("#>-").strip() if lines else ""


def derive_heading(text: str, *, limit: int = 24) -> str:
    """模型没给标题时，从转写首句本地推导一个。"""
    first = (text or "").strip().splitlines()[0] if text and text.strip() else ""
    for sep in ("。", "！", "？", ".", "!", "?"):
        if sep in first:
            first = first.split(sep)[0]
            break
    first = first.strip(" ，,、:：")
    return first[:limit] + ("…" if len(first) > limit else "")


def _outline(state: SummaryState) -> str:
    return "\n".join(
        f"- {fmt_range(n.start, n.end)} {n.heading or '（未命名段落）'}"
        for n in state.notes
    )


def load_state(work_dir: Path) -> SummaryState:
    return SummaryState.from_dict(read_json(Path(work_dir) / STATE_FILE))


def save_state(work_dir: Path, state: SummaryState) -> None:
    write_json(Path(work_dir) / STATE_FILE, state.to_dict())


def compress(config: llm_mod.LLMConfig, summary: str, *, limit: int = 4000) -> str:
    raw = llm_mod.chat(config, [
        {"role": "system", "content": "你是严谨的文本压缩助手，只做删减不做增补。"},
        {"role": "user", "content": COMPRESS_TEMPLATE.format(summary=summary, limit=limit)},
    ])
    return raw.strip() or summary


def update_summary(config: llm_mod.LLMConfig, state: SummaryState,
                   chunk_text: str, index: int, total: int,
                   *, start: float, end: float,
                   max_chunk_chars: int = 6000,
                   max_running_chars: int = 4000,
                   images=None) -> None:
    """把一段新转写并入滚动摘要，并记录该段小标题。

    images: 该段关键帧路径列表；传入时走多模态消息，让画面信息也并入摘要。
    """
    text = (chunk_text or "").strip()
    if len(text) > max_chunk_chars:
        text = text[:max_chunk_chars] + "\n……（本段过长，已截断）"

    if len(state.running) > max_running_chars:
        state.running = compress(config, state.running, limit=max_running_chars)

    prompt = UPDATE_TEMPLATE.format(
        summary=state.running or "（空，这是第一段）",
        index=index + 1, total=total, span=fmt_range(start, end), text=text,
    )
    if images:
        prompt += VISION_HINT
        user_content = [{"type": "text", "text": prompt}]
        user_content += [frames_mod.image_part(p) for p in images]
    else:
        user_content = prompt
    raw = llm_mod.chat(config, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    summary, heading = parse_update(raw)
    if summary:
        state.running = summary
    state.notes.append(ChunkNote(index, start, end,
                                 heading or derive_heading(chunk_text)))
    state.next_index = index + 1


def finalize(config: llm_mod.LLMConfig, state: SummaryState, title: str) -> str:
    """收尾一轮：把滚动摘要整理成结构化终稿。"""
    if not state.running.strip():
        return ""
    raw = llm_mod.chat(config, [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": FINAL_TEMPLATE.format(
            title=title, outline=_outline(state), summary=state.running,
        )},
    ], max_tokens=max(config.max_tokens, 3072))
    state.final = raw.strip()
    return state.final
