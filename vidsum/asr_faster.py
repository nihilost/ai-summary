"""faster-whisper 后端（本地推理，可选依赖）。"""

from __future__ import annotations

import os

# torch 与 ctranslate2 各自静态链接了一份 OpenMP 运行库（libiomp5md.dll），
# 同时加载会触发 OMP Error #15 并直接崩溃（无 Python traceback，整个进程被杀）。
# 在 import torch 之前设好这个变量，让 OpenMP 容忍重复初始化。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

from .asr import ASRBackend, Utterance


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, *, model_size: str = "medium", device: str = "auto",
                 compute_type: str = "int8", language: str | None = "zh",
                 beam_size: int = 5, download_root: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 faster-whisper。执行 `pip install faster-whisper`，"
                "或改用 --asr api 走云端转写。"
            ) from exc

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        self.model_size = model_size
        self.device = device
        self.language = language
        self.beam_size = beam_size
        # ctranslate2 的模型对象可被多线程并发调用（内部为 C++ 推理，会释放 GIL），
        # 因此多个工作线程共用一个实例即可——这避免了每线程一份权重的内存放大。
        # 若某个后端不是线程安全的，用 --workers 1 退化为串行。
        # 未显式指定时默认把模型放在项目目录下的 models\whisper，
        # 与 sherpa 后端的 models\sherpa 约定保持一致，避免每次传参。
        if download_root is None:
            download_root = str(
                Path(__file__).resolve().parent.parent / "models" / "whisper")
        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(self, audio_path: Path) -> list[Utterance]:
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return [Utterance(float(s.start), float(s.end), (s.text or "").strip())
                for s in segments if (s.text or "").strip()]
