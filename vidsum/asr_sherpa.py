"""sherpa-onnx 后端：本地 paraformer 转写，模型可复用 VideoToNo 已下载的那套。

流水线是「解码 → VAD 切段 → 逐段 paraformer → 恢复标点」。
VAD 那一步不能省：paraformer 一次喂太长的音频输出会明显劣化，
而 VAD 切出来的段正好落在语句之间，时间和内容都对得上。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import threading
from pathlib import Path

from .asr import ASRBackend, Utterance
from .audio_av import decode_pcm
from .util import log

@dataclass(frozen=True)
class _ModelLayout:
    """模型目录的标准布局。"""

    asr_subdir: str = "asr"
    asr_model: str = "model.int8.onnx"
    tokens: str = "tokens.txt"
    vad_subdir: str = "vad"
    vad_model: str = "silero_vad.onnx"
    punc_subdir: str = "punc"
    punc_model: str = "model.onnx"


_LAYOUT = _ModelLayout()

# 除了 --sherpa-model-dir 和环境变量，也顺手找几处常见的第三方缓存。
# VideoToNo 把模型放在 workspace\_model_cache\sherpa 下，装过它就不用再下一遍。
_CANDIDATE_DIRS = (
    Path(__file__).resolve().parent.parent / "models" / "sherpa",
    Path.cwd() / "models" / "sherpa",
    Path(os.environ.get("LOCALAPPDATA", "")) / "vidsum" / "models" / "sherpa",
    Path(r"E:\ai视频总结\workspace\_model_cache\sherpa"),
)


class SherpaError(RuntimeError):
    pass


def find_model_dir(explicit: str | None = None) -> Path:
    """按 显式参数 > 环境变量 > 常见位置 的顺序定位模型目录。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("VIDSUM_SHERPA_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend(_CANDIDATE_DIRS)

    searched: list[str] = []
    for cand in candidates:
        if not cand:
            continue
        text = str(cand)
        if text not in searched:
            searched.append(text)
        if not cand.is_dir():
            continue
        if ((cand / _LAYOUT.asr_subdir / _LAYOUT.asr_model).is_file()
                and (cand / _LAYOUT.asr_subdir / _LAYOUT.tokens).is_file()):
            return cand
    raise SherpaError(
        "找不到 sherpa 模型。期望目录下同时有 asr/model.int8.onnx 与 "
        "asr/tokens.txt。已搜索：\n  " + "\n  ".join(searched) +
        "\n可用 --sherpa-model-dir 或环境变量 VIDSUM_SHERPA_MODEL_DIR 指定。"
    )


class SherpaBackend(ASRBackend):
    name = "sherpa"

    def __init__(self, *, model_dir: str | None = None, num_threads: int = 2,
                 feed_samples: int = 1024, min_segment_samples: int = 1600,
                 warmup_seconds: float = 0.5, punctuation: bool = True,
                 vad_threshold: float = 0.5,
                 min_silence_duration: float = 0.5,
                 min_speech_duration: float = 0.25,
                 max_speech_duration: float = 20.0):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SherpaError(
                "未安装 sherpa-onnx。执行 `pip install sherpa-onnx`，"
                "或改用 --asr api 走云端转写。"
            ) from exc

        self._sherpa = sherpa_onnx
        self.model_dir = find_model_dir(model_dir)
        self.num_threads = max(1, int(num_threads))
        # VAD 按多小的块喂入：块越大分段越糙，实测 1s 以上的粒度会把整段
               # 语音并成一句，paraformer 的输出质量随之下滑。默认 1024 样本=64ms。
        self.feed_samples = max(512, int(feed_samples))
        self.min_segment_samples = max(0, int(min_segment_samples))
        self.punctuation = bool(punctuation)
        self.sample_rate = 16000
        # 起流预热：silero 的 LSTM 状态从零开始，音频最开头约 1 秒常被判成
        # 非语音，视频第一句就这么整句丢失。喂 VAD 前垫一段静音让模型进入
        # 稳态，检出后再把这段从时间戳里减掉，首句就能从接近 0 的位置回来。
        self.warmup_samples = max(0, int(float(warmup_seconds)
                                        * self.sample_rate))

        asr_model = self.model_dir / _LAYOUT.asr_subdir / _LAYOUT.asr_model
        tokens = self.model_dir / _LAYOUT.asr_subdir / _LAYOUT.tokens
        vad_model = self.model_dir / _LAYOUT.vad_subdir / _LAYOUT.vad_model
        if not vad_model.is_file():
            raise SherpaError(
                f"缺少 VAD 模型 {vad_model}。没有它就无法把长音频切成短段，"
                "paraformer 直接吃长音频会明显劣化。"
            )

        s = sherpa_onnx
        self._recognizer = s.OfflineRecognizer.from_paraformer(
            paraformer=str(asr_model), tokens=str(tokens),
            num_threads=self.num_threads,
        )
        self._vad = s.VoiceActivityDetector(
            s.VadModelConfig(
                silero_vad=s.SileroVadModelConfig(
                    model=str(vad_model),
                    threshold=float(vad_threshold),
                    min_silence_duration=float(min_silence_duration),
                    min_speech_duration=float(min_speech_duration),
                    max_speech_duration=float(max_speech_duration),
                ),
                sample_rate=self.sample_rate, num_threads=1,
            ),
            buffer_size_in_seconds=30,
        )
        self._punctuator = None
        punc_model = (self.model_dir / _LAYOUT.punc_subdir
                      / _LAYOUT.punc_model)
        if self.punctuation and punc_model.is_file():
            self._punctuator = s.OfflinePunctuation(s.OfflinePunctuationConfig(
                model=s.OfflinePunctuationModelConfig(
                    ct_transformer=str(punc_model), num_threads=self.num_threads,
                )))
        elif self.punctuation:
            log(f"[sherpa] 未找到标点模型 {punc_model}，跳过标点恢复")

        # VoiceActivityDetector 带内部状态，多个 worker 线程共用一个实例会
        # 把彼此的音视频错切到一起。这里用一把实例锁把 transcribe 串起来，
        # 单段推理本身已由 num_threads 做线程级并行，整体吞吐不受影响。
        self._lock = threading.Lock()

    def _segments(self, samples):
        """逐块喂给 VAD， yield (起点样本, 该段样本)。

        必须先取走 samples 再 pop()：pop 会释放 front 返回的内部对象，
        倒过来读只会拿到空列表，所以这里立刻拷成独立 list。

        开头垫一段静音规避 silero 起流漏检，检出后再把这段从起点里减掉，
        所以 yield 出去的 start 仍是相对原始音频的位置。
        """
        import numpy as np

        vad = self._vad
        vad.reset()
        feed = self.feed_samples
        pad = self.warmup_samples
        data = samples if pad <= 0 else np.concatenate(
            [np.zeros(pad, dtype=samples.dtype), samples])
        fed = 0
        total = len(data)
        # 尾部不足一个喂入块的样本也要喂进去，否则结尾会缺一小段
        while fed < total:
            vad.accept_waveform(data[fed:fed + feed])
            fed += feed
            while not vad.empty():
                seg = vad.front
                piece = list(seg.samples)
                start = seg.start
                vad.pop()
                if len(piece) >= self.min_segment_samples:
                    yield max(0, start - pad), piece
        vad.flush()
        while not vad.empty():
            seg = vad.front
            piece = list(seg.samples)
            start = seg.start
            vad.pop()
            if len(piece) >= self.min_segment_samples:
                yield max(0, start - pad), piece

    def transcribe(self, audio_path: Path) -> list[Utterance]:
        import numpy as np

        with self._lock:
            pcm = decode_pcm(audio_path, sample_rate=self.sample_rate)
            if pcm.size == 0:
                return []
            samples = pcm.astype(np.float32) / 32768.0

            utterances: list[Utterance] = []
            for start, piece in self._segments(samples):
                stream = self._recognizer.create_stream()
                stream.accept_waveform(self.sample_rate,
                                       np.asarray(piece, dtype=np.float32))
                self._recognizer.decode_stream(stream)
                text = (stream.result.text or "").strip()
                if not text:
                    continue
                if self._punctuator is not None:
                    text = self._punctuator.add_punctuation(text)
                begin = start / self.sample_rate
                end = (start + len(piece)) / self.sample_rate
                utterances.append(Utterance(begin, end, text))
            return utterances
