"""确定性假后端：不加载模型、不联网，用于联调整条流水线。"""

from __future__ import annotations

from pathlib import Path

from .asr import ASRBackend, Utterance

_LINES = (
    "这一节我们先回顾上节课留下的问题。",
    "接下来介绍核心定义，请注意它与相邻概念的区别。",
    "这里有一个常用结论，考试中经常直接考查。",
    "最后提醒两个容易混淆的边界条件。",
)


class DummyBackend(ASRBackend):
    name = "dummy"

    def transcribe(self, audio_path: Path) -> list[Utterance]:
        stem = Path(audio_path).stem
        return [
            Utterance(float(i * 7), float(i * 7 + 6),
                      f"{stem}: {line}")
            for i, line in enumerate(_LINES)
        ]
