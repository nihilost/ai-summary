"""按静音点切分音频的纯逻辑。

切分策略：以目标时长为基准，在允许窗口内寻找最接近目标位置的静音中点下刀。
静音处切分可以避免把一句话拦腰截断，显著降低 ASR 在边界处的重复与丢字。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_segments(duration: float, *, target: float = 600.0,
                  minimum: float = 240.0, maximum: float = 1200.0,
                  silences=()) -> list[Segment]:
    """把 [0, duration] 切成若干段，尽量落在静音上。

    先按目标时长算出段数并均分，再把每个切点吸附到附近的静音中点。
    均分是为了让每一段送给模型的文本量大致相等——滚动摘要的单轮开销
    因此可预测；吸附静音则是为了不把语句拦腰截断。

    silences: [(start, end), ...] 静音区间，来自 ffmpeg silencedetect。
    """
    if duration <= 0:
        raise ValueError(f"音频时长必须为正数，得到 {duration}")
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"非法的分段参数：minimum={minimum}, maximum={maximum}")
    target = min(max(target, minimum), maximum)

    midpoints = sorted((s + e) / 2.0 for s, e in silences if e > s)

    count = max(1, math.ceil(duration / target))
    nominal = duration / count
    if nominal > maximum:
        count = max(1, math.ceil(duration / maximum))
        nominal = duration / count
    elif nominal < minimum:
        count = max(1, int(duration // minimum))
        nominal = duration / count

    cuts: list[float] = []
    for i in range(1, count):
        ideal = i * nominal
        lo, hi = ideal - nominal / 2.0, ideal + nominal / 2.0
        window = [m for m in midpoints if lo <= m <= hi]
        cut = min(window, key=lambda m: abs(m - ideal)) if window else ideal
        # 夹在 [已用最短, 剩余最短] 之间，避免出现极短的首尾段
        cut = min(max(cut, i * minimum), duration - (count - i) * minimum)
        cuts.append(cut)

    bounds = [0.0, *cuts, duration]
    return [Segment(i, bounds[i], bounds[i + 1])
            for i in range(len(bounds) - 1)]


def fmt_ts(seconds: float, *, always_hours: bool = False) -> str:
    """秒转 mm:ss 或 h:mm:ss。"""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours or always_hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_range(start: float, end: float) -> str:
    hours = end >= 3600
    return f"{fmt_ts(start, always_hours=hours)}\u2013{fmt_ts(end, always_hours=hours)}"
