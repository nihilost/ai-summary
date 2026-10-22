"""ASR 后端注册与统一数据模型。"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Utterance:
    start: float
    end: float
    text: str


@dataclass
class ChunkResult:
    index: int
    start: float
    end: float
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances if u.text).strip()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "utterances": [asdict(u) for u in self.utterances],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkResult":
        utts = [Utterance(float(u.get("start", 0)), float(u.get("end", 0)),
                          str(u.get("text", "")))
                for u in data.get("utterances", [])]
        return cls(int(data["index"]), float(data["start"]), float(data["end"]), utts)


class ASRBackend:
    """后端协议：输入单段音频路径，返回相对该段起点的时间戳 utterances。"""
    name = "abstract"

    def transcribe(self, audio_path: Path) -> list[Utterance]:  # pragma: no cover
        raise NotImplementedError


def get_backend(name: str, **kwargs) -> ASRBackend:
    from . import asr_api, asr_dummy, asr_faster, asr_sherpa

    registry = {
        "faster-whisper": asr_faster.FasterWhisperBackend,
        "api": asr_api.ApiBackend,
        "dummy": asr_dummy.DummyBackend,
        "sherpa": asr_sherpa.SherpaBackend,
    }
    if name not in registry:
        raise ValueError(f"未知 ASR 后端：{name}（可选：{', '.join(registry)}）")
    # CLI 会一次性传入所有选项，这里按后端签名挑出它认识的那些，
    # 顺带让拼错参数名在后端构造时暴露，而不是被静默吞掉。
    cls = registry[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def normalize(text: str) -> str:
    """边界去重用的归一化：去空白与标点，忽略大小写。"""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def looks_duplicated(a: str, b: str) -> bool:
    """判断两段文本是否本质相同（切点落在同一个短语两端时会出现）。"""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 8 and shorter in longer:
        return True
    sa, sb = set(na), set(nb)
    if not sa or not sb:
        return False
    overlap = len(sa & sb) / len(sa | sb)
    return len(shorter) >= 12 and overlap >= 0.9


def stitch(chunks: list[ChunkResult], *, dedup: bool = True) -> list[Utterance]:
    """把各段结果按顺序拼成全局 utterances，时间戳加上段内偏移。"""
    return [u for _chunk, utts in merge_chunks(chunks, dedup=dedup) for u in utts]


def merge_chunks(chunks: list[ChunkResult], *,
                 dedup: bool = True) -> list[tuple[ChunkResult, list[Utterance]]]:
    """按 stitch 的规则逐段算出「实际保留的 utterances」。

    返回 (段, 保留句) 配对，调用方既能拼全局稿，也能拿到每段各自的干净文本
    ——摘要用的文本因此和用户在 transcript.md 里看到的一致。
    """
    pairs: list[tuple[ChunkResult, list[Utterance]]] = []
    last_text = ""
    for chunk in sorted(chunks, key=lambda c: c.index):
        shifted = [Utterance(chunk.start + u.start, chunk.start + u.end, u.text)
                   for u in chunk.utterances if u.text.strip()]
        if dedup and shifted and last_text \
                and looks_duplicated(last_text, shifted[0].text):
            shifted = shifted[1:]
        if shifted:
            last_text = shifted[-1].text
        pairs.append((chunk, shifted))
    return pairs
