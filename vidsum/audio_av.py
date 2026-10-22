"""用 PyAV（FFmpeg 的 Python 绑定）处理音频，作为 ffmpeg 命令行之外的另一条路。

命令行 ffmpeg 需要用户自己装好并加进 PATH；PyAV 的 wheel 自带整套 FFmpeg
动态库，装完即用。两条路输出一致，pipeline 只依赖这里暴露的几个函数。
"""

from __future__ import annotations

from pathlib import Path

from .media import Codec, MediaError

try:
    import av as _av
except ImportError:  # PyAV 是可选依赖，缺失时 available() 返回 False
    _av = None

try:
    import numpy as _np
except ImportError:
    _np = None


def available() -> bool:
    """PyAV 与 numpy 是否都可用。"""
    return _av is not None and _np is not None


def _require():
    if not available():
        raise MediaError(
            "未安装 PyAV。执行 `pip install av` 即可获得内置 FFmpeg 的音频解码能力；"
            "也可以改用 --audio-backend ffmpeg 走系统 ffmpeg。"
        )


def _audio_stream(container):
    if not container.streams.audio:
        raise MediaError("输入没有音轨")
    return container.streams.audio[0]


def _open_input(source, *, headers: str | None = None):
    _require()
    try:
        options = {}
        if headers:
            # FFmpeg 的 http 协议用 headers 选项带自定义请求头
            options["headers"] = str(headers)
        return _av.open(str(source), options=options or None)
    except Exception as exc:
        raise MediaError(f"无法打开音频源：{exc}") from exc


def _to_seconds(ticks, time_base) -> float:
    """把 ticks 换算成秒。

    PyAV 的 time_base 有过两种形态：Fraction(1, N)（13 以前）和 int N（现在）。
    小于 1 的是前者，否则是后者，两种都要认。
    """
    if ticks is None:
        return 0.0
    base = float(time_base)
    if 0 < base < 1:
        return float(ticks) * base
    return float(ticks) / base if base else 0.0


def _to_ticks(seconds: float, time_base) -> int:
    """秒换算成 ticks，与 _to_seconds 互为逆运算。"""
    base = float(time_base)
    if 0 < base < 1:
        return int(float(seconds) / base)
    return int(float(seconds) * base)


def probe_duration(source, *, headers: str | None = None) -> float:
    """读取容器时长（秒）。"""
    container = _open_input(source, headers=headers)
    try:
        duration = container.duration
        if duration is None or duration <= 0:
            # 有些容器不写总时长，退回到逐帧累加
            total = 0.0
            for frame in container.decode(_audio_stream(container)):
                total = max(total, _to_seconds(frame.pts, frame.time_base)
                            + float(frame.samples) / 16000.0)
            return total
        return _to_seconds(duration, _av.time_base)
    finally:
        container.close()


def decode_pcm(source, *, sample_rate: int = 16000, channels: int = 1,
               headers: str | None = None):
    """把整个音频源解码成 16bit int16 一维数组。"""
    _require()
    container = _open_input(source, headers=headers)
    try:
        stream = _audio_stream(container)
        resampler = _av.AudioResampler(
            format="s16", layout="mono" if channels == 1 else "stereo",
            rate=sample_rate,
        )
        parts: list[bytes] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().tobytes())
        # 重采样器内部有缓存，必须再喂一次 None 把尾部排空
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().tobytes())
    finally:
        container.close()
    return _np.frombuffer(b"".join(parts), dtype=_np.int16)


def _add_audio_stream(container, codec: Codec, sample_rate: int, channels: int):
    try:
        stream = container.add_stream(codec.name, rate=sample_rate)
    except Exception as exc:
        raise MediaError(f"PyAV 不支持编码器 {codec.name}: {exc}") from exc
    stream.layout = "mono" if channels == 1 else "stereo"
    if codec.name != "pcm_s16le":
        stream.bit_rate = 64000 if codec.name == "libopus" else 128000
    return stream


def _encode_frames(container, stream, frames):
    for frame in frames:
        frame.sample_rate = stream.rate
        for packet in stream.encode(frame):
            container.mux(packet)


def _flush_stream(container, stream):
    """排空编码器。encode(None) 每个输出流只能调一次。"""
    for packet in stream.encode(None):
        container.mux(packet)


def extract_audio(source, dst: Path, *, codec: Codec, sample_rate: int = 16000,
                  channels: int = 1, headers: str | None = None) -> Path:
    """整段抽取音轨，交给 PyAV 自带的 FFmpeg 解复用与重采样。"""
    _require()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    container = _open_input(source, headers=headers)
    try:
        stream = _audio_stream(container)
        resampler = _av.AudioResampler(
            format="s16", layout="mono" if channels == 1 else "stereo",
            rate=sample_rate,
        )
        with _av.open(str(dst), "w") as out:
            out_stream = _add_audio_stream(out, codec, sample_rate, channels)
            pending: list = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    pending.append(resampled)
                _encode_frames(out, out_stream, pending)
                pending.clear()
            # 重采样器内部有缓存，必须再喂一次 None 把尾部排空
            for resampled in resampler.resample(None):
                pending.append(resampled)
            _encode_frames(out, out_stream, pending)
            _flush_stream(out, out_stream)
    finally:
        container.close()
    if not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError("音频抽取失败：输出文件为空")
    return dst


def cut_segment(source, dst: Path, start: float, end: float, *, codec: Codec,
                sample_rate: int = 16000, channels: int = 1,
                headers: str | None = None) -> Path:
    """从音频源切出一小段。

    先 seek 到起点再按样本数收口：seek 之后 frame.pts 相对起点计时，
    拿它判断会把整段解码到底，所以这里数够样本就停。
    """
    _require()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = max(0, int((float(end) - float(start)) * sample_rate))
    container = _open_input(source, headers=headers)
    try:
        stream = _audio_stream(container)
        # 不传 stream：offset 就用 av.time_base（AV_TIME_BASE，微秒）为单位。
        # 传了 stream 就得换成该流的 time_base，两者混用会 seek 到完全错误的位置。
        container.seek(_to_ticks(start, _av.time_base), backward=True)
        resampler = _av.AudioResampler(
            format="s16", layout="mono" if channels == 1 else "stereo",
            rate=sample_rate,
        )
        with _av.open(str(dst), "w") as out:
            out_stream = _add_audio_stream(out, codec, sample_rate, channels)
            got = 0
            pending: list = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    pending.append(resampled)
                    got += len(resampled.to_ndarray().reshape(-1))
                _encode_frames(out, out_stream, pending)
                pending.clear()
                if got >= want:
                    break
            for resampled in resampler.resample(None):
                pending.append(resampled)
            _encode_frames(out, out_stream, pending)
            _flush_stream(out, out_stream)
    finally:
        container.close()
    if not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError(f"切片失败：{dst.name}")
    return dst


_SILENCE_FRAME_S = 0.02


def detect_silences(source, *, noise_db: float = 30.0, min_silence: float = 0.5,
                    sample_rate: int = 16000,
                    headers: str | None = None) -> list[tuple[float, float]]:
    """用帧能量找静音区间，语义对齐 ffmpeg 的 silencedetect。

    每 20ms 算一次 RMS 并换算成 dB，低于阈值的连续帧合并成一段静音。
    整段音频先解码进内存，静音检测只跑一次，可接受。
    """
    _require()
    pcm = decode_pcm(source, sample_rate=sample_rate, headers=headers)
    if pcm.size == 0:
        return []
    samples = pcm.astype(_np.float32) / 32768.0
    width = max(1, int(_SILENCE_FRAME_S * sample_rate))
    frames = samples.size // width
    if frames == 0:
        return []
    blocks = samples[:frames * width].reshape(frames, width)
    rms = _np.sqrt((blocks ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * _np.log10(rms + 1e-12)
    quiet = db <= -float(noise_db)

    min_frames = max(1, int(round(float(min_silence) / _SILENCE_FRAME_S)))
    spans: list[tuple[float, float]] = []
    i = 0
    while i < frames:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < frames and quiet[j]:
            j += 1
        if j - i >= min_frames:
            spans.append((i * _SILENCE_FRAME_S, j * _SILENCE_FRAME_S))
        i = j
    return spans


def pick_codec(prefer: str | None = None) -> Codec:
    """按当前 PyAV 构建实际支持的编码器挑一个，优先级与 ffmpeg 路一致。"""
    _require()
    from .media import CODECS
    order = list(CODECS)
    if prefer:
        chosen = [c for c in CODECS if c.name == prefer]
        if not chosen:
            raise MediaError(f"未知的音频编码器：{prefer}")
        order = chosen + [c for c in CODECS if c.name != prefer]
    for codec in order:
        try:
            if codec.name in _av.codec.codecs_available:
                return codec
        except Exception:
            continue
    raise MediaError(
        "当前 PyAV 未提供任何可用音频编码器"
        f"（已检查：{', '.join(c.name for c in CODECS)}）"
    )
