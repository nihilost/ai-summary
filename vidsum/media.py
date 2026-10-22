"""ffmpeg / ffprobe 的定位、探测与音频处理。

所有重活都交给 ffmpeg 这个原生二进制，本模块只负责拼参数和解析输出。
整段音频只抽取一次并缓存，后续切片与转写都基于缓存文件，天然支持断点续跑。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .util import log


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Codec:
    name: str
    ext: str
    args: tuple[str, ...]


# 按优先级排列：mp3 体积最小且 ASR 够用，缺失时逐级回退。
CODECS = (
    Codec("libmp3lame", "mp3", ("-c:a", "libmp3lame", "-q:a", "5")),
    Codec("libopus", "ogg", ("-c:a", "libopus", "-b:a", "32k")),
    Codec("aac", "m4a", ("-c:a", "aac", "-b:a", "48k")),
    Codec("pcm_s16le", "wav", ("-c:a", "pcm_s16le")),
)

_COMMON_DIRS = (
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ProgramData\chocolatey\bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/homebrew/bin",
)

INSTALL_HINT = (
    "Windows 可执行 `winget install Gyan.FFmpeg`，macOS 可执行 `brew install ffmpeg`；"
    "也可以用 --ffmpeg/--ffprobe 指定路径，或设置环境变量 "
    "VIDSUM_FFMPEG / VIDSUM_FFPROBE。"
)


def _tool_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    common = _COMMON_DIRS if os.name == "nt" else _COMMON_DIRS[-3:]
    return [
        here,                       # 随包打包进来的二进制
        here.parent,
        Path(sys.executable).parent,
        Path.cwd(),
        *(Path(d) for d in common),
    ]


def find_tool(name: str, explicit: str | None = None) -> str:
    """按 显式参数 > 环境变量 > PATH > 常见目录 的顺序定位 ffmpeg/ffprobe。"""
    if name not in ("ffmpeg", "ffprobe"):
        raise MediaError(f"不支持的工具名：{name}")
    candidates: list[str] = []
    if explicit:
        candidates.append(str(explicit))
    env = os.environ.get(f"VIDSUM_{name.upper()}")
    if env:
        candidates.append(env)
    found = shutil.which(name)
    if found:
        candidates.append(found)
    exe = f"{name}.exe" if os.name == "nt" else name
    candidates.extend(str(d / exe) for d in _tool_dirs())

    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(Path(cand))
    raise MediaError(f"找不到 {name}。{INSTALL_HINT}")


def _run(cmd: list[str], *, timeout: float = 900.0, check: bool = True):
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到可执行文件：{cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{Path(cmd[0]).name} 执行超时（{timeout:.0f}s）") from exc
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        raise MediaError(f"{Path(cmd[0]).name} 退出码 {proc.returncode}: {tail}")
    return proc


def available_encoders(ffmpeg: str) -> set[str]:
    """解析 `ffmpeg -encoders`，返回编码器名称集合。"""
    out = _run([ffmpeg, "-hide_banner", "-encoders"], timeout=60).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # 形如 " V....D libmp3lame   MP3 (MPEG audio layer 3)"
        if len(parts) >= 2 and len(parts[0]) == 6 and set(parts[0]) <= set("AXSVFD.B"):
            names.add(parts[1])
    return names


def pick_codec(ffmpeg: str, prefer: str | None = None) -> Codec:
    """选一个当前 ffmpeg 构建实际支持的音频编码器。"""
    encoders = available_encoders(ffmpeg)
    order = list(CODECS)
    if prefer:
        chosen = [c for c in CODECS if c.name == prefer]
        if not chosen:
            raise MediaError(f"未知的音频编码器：{prefer}")
        order = chosen + [c for c in CODECS if c.name != prefer]
    for codec in order:
        if codec.name in encoders:
            return codec
    raise MediaError(
        "当前 ffmpeg 未提供任何可用音频编码器"
        f"（已检查：{', '.join(c.name for c in CODECS)}）"
    )


def probe_duration(ffprobe: str, source, *, headers: str | None = None,
                   timeout: float = 120.0) -> float:
    cmd = [ffprobe, "-v", "error"]
    if headers:
        cmd += ["-headers", headers]
    cmd += ["-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
            str(source)]
    out = _run(cmd, timeout=timeout).stdout.strip()
    try:
        return float(out.splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise MediaError(f"无法读取媒体时长（ffprobe 输出：{out!r}）") from exc


def extract_audio(ffmpeg: str, source, dst: Path, *, codec: Codec,
                  headers: str | None = None, sample_rate: int = 16000,
                  channels: int = 1, timeout: float = 7200.0) -> Path:
    """整段抽取音轨。视频/网络流都由 ffmpeg 自行解复用，本地不落视频文件。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", headers]
    cmd += ["-i", str(source), "-vn", "-map", "0:a:0?",
            "-ar", str(sample_rate), "-ac", str(channels)]
    cmd += list(codec.args) + [str(dst)]
    _run(cmd, timeout=timeout)
    if not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError("音频抽取失败：输出文件为空")
    return dst


def cut_segment(ffmpeg: str, source, dst: Path, start: float, end: float, *,
                codec: Codec, sample_rate: int = 16000, channels: int = 1,
                timeout: float = 1800.0) -> Path:
    """从已缓存的整段音频上切出一小段，供单个 ASR 任务使用。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(source), "-vn", "-ar", str(sample_rate), "-ac", str(channels)]
    cmd += list(codec.args) + [str(dst)]
    _run(cmd, timeout=timeout)
    if not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError(f"切片失败：{dst.name}")
    return dst


_SILENCE_RE = re.compile(r"silence_(start|end):\s*([0-9.]+)")


def detect_silences(ffmpeg: str, source, *, noise_db: float = 30.0,
                    min_silence: float = 0.5,
                    timeout: float = 3600.0) -> list[tuple[float, float]]:
    """扫描整段音频的静音区间，用于把切片点对齐到语句之间。"""
    # Windows 上 `-f null -` 行为不一致，写 NUL 更稳。
    sink = "NUL" if os.name == "nt" else "/dev/null"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "info", "-i", str(source),
           "-af", f"silencedetect=noise=-{noise_db:g}dB:d={min_silence:g}",
           "-f", "null", sink]
    proc = _run(cmd, timeout=timeout)
    text = (proc.stderr or "") + (proc.stdout or "")

    starts: list[float] = []
    ends: list[float] = []
    for kind, value in _SILENCE_RE.findall(text):
        (starts if kind == "start" else ends).append(float(value))

    spans: list[tuple[float, float]] = []
    for start in starts:
        end = next((e for e in ends if e >= start), None)
        spans.append((start, end if end is not None else start + min_silence))
    return [s for s in spans if s[1] > s[0]]


def grab_frames(ffmpeg: str, source, jobs, *, headers: str | None = None,
                width: int = 480, timeout: float = 600.0) -> list[Path]:
    """按 jobs=[(dst, seconds), ...] 逐个抽 JPEG 帧，返回成功落盘的路径。

    -ss 放在 -i 之前（input seeking）以保证长视频上也能快速定位；
    单帧失败只记警告不中断——抽帧是增强步骤，不该让整条流水线栽在上面。
    """
    saved: list[Path] = []
    for dst, seconds in jobs:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if headers:
            cmd += ["-headers", headers]
        cmd += ["-ss", f"{seconds:.3f}", "-i", str(source),
                "-frames:v", "1", "-vf", f"scale={width}:-2",
                "-q:v", "3", str(dst)]
        try:
            _run(cmd, timeout=timeout)
        except MediaError as exc:
            log(f"[画面] {seconds:.1f}s 抽帧失败：{exc}")
            continue
        if not dst.is_file() or dst.stat().st_size == 0:
            log(f"[画面] {seconds:.1f}s 抽帧结果为空，跳过")
            continue
        saved.append(dst)
    return saved
