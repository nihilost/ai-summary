"""faster-whisper 后端（本地推理，可选依赖）。"""

from __future__ import annotations

import os
import shutil

# ctranslate2 与 onnxruntime 各自静态链接了一份 OpenMP 运行库
# （libiomp5md.dll），同时加载会触发 OMP Error #15 并直接崩溃
# （无 Python traceback，整个进程被杀）。提前设好这个变量，
# 让 OpenMP 容忍重复初始化。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

from .asr import ASRBackend, Utterance
from .util import app_root, human_size, log

# ---- model_size="auto" 的档位选择 --------------------------------------

# 显存档位表：(所需显存 MB, 模型档位)。按 int8 推理的近似峰值占用估的
# （权重 + 推理工作区 + 余量），宁可保守——显存不够时 ctranslate2 是直接
# 崩，不是优雅报错。large-v3 在 8GB 上其实多半跑得动，但笔记本独显还要
# 带屏幕显示，这里留给显式 --model-size large-v3 去 opt-in。
_VRAM_TIERS = [
    (10240, "large-v3"),
    (6144, "medium"),
    (3072, "small"),
    (0, "base"),
]

# 已缓存模型“装不装得下”的显存需求，与 _VRAM_TIERS 同源
_VRAM_NEED_MB = {"tiny": 512, "base": 1024, "small": 3072, "medium": 6144,
                 "distil-large-v3": 10240, "large-v3-turbo": 10240,
                 "large-v2": 10240, "large-v3": 10240}

# 缓存偏好用的大小排序（越靠后越大），挑“已缓存里最大的”
_SIZE_RANK = ["tiny", "base", "small", "medium",
              "distil-large-v3", "large-v3-turbo", "large-v2", "large-v3"]

# 首次下载的近似体积，用于磁盘空间预检和日志提示
_APPROX_BYTES = {
    "tiny": 80_000_000,
    "base": 150_000_000,
    "small": 480_000_000,
    "medium": 1_600_000_000,
    "large-v2": 3_200_000_000,
    "large-v3": 3_200_000_000,
    "distil-large-v3": 1_600_000_000,
    "large-v3-turbo": 1_700_000_000,
}


def _cuda_vram_mb() -> int | None:
    """第一块 NVIDIA 显卡的显存（MB）；探测不到返回 None。

    走 ctypes 直接加载驱动自带的 nvml.dll：不引第三方包，exe 冻结后也
    零额外依赖。非 NVIDIA 机器 / 无驱动 / 初始化失败都安静返回 None，
    由调用方按保守档位处理。
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        nvml = ctypes.WinDLL("nvml.dll")
    except OSError:
        return None
    try:
        if nvml.nvmlInit_v2() != 0:
            return None

        class _Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]

        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
            return None
        info = _Memory()
        if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != 0:
            return None
        return int(info.total // (1024 * 1024))
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def _cached_models(download_root: str) -> dict[str, Path]:
    """扫出 download_root 下已缓存的 whisper 模型：{档位: 缓存目录}。

    HF 缓存布局是 models--Systran--faster-whisper-<size>/snapshots/<rev>/，
    snapshots 里真有 model.bin 才算数（半截缓存不算，交给下载逻辑补齐）。
    """
    root = Path(download_root)
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    prefix = "models--Systran--faster-whisper-"
    for entry in root.glob(prefix + "*"):
        if not entry.is_dir():
            continue
        if any(entry.rglob("model.bin")):
            found[entry.name[len(prefix):]] = entry
    return found


def _resolve_auto_size(requested: str, *, device: str,
                       download_root: str) -> tuple[str, str, dict]:
    """把 "auto" 解析成具体档位，返回 (档位, 理由, 已缓存字典)。

    已经躺在本地的模型不花流量，优先复用；没有再按显存档位下载：
    - CUDA：显存够得着的缓存模型里挑最大的，没有才按档位下载
    - CPU：缓存里有就用最大的（下了没有不用的道理），否则 base
    """
    cached = _cached_models(download_root)
    if requested != "auto":
        return requested, "指定档位", cached
    vram = _cuda_vram_mb() if device == "cuda" else 0
    if device == "cuda" and vram is None:
        vram = 6144  # 有卡但问不出显存（nvml 被禁等）：按 medium 档保守处理
    recommended = next(size for need, size in _VRAM_TIERS if vram >= need)
    if cached:
        pool = ([s for s in cached if vram >= _VRAM_NEED_MB.get(s, 1 << 30)]
                if device == "cuda" else list(cached))
        if pool:
            pick = max(pool, key=lambda s: _SIZE_RANK.index(s)
                       if s in _SIZE_RANK else -1)
            why = "复用已缓存模型" + (f"（显存 {vram}MB）"
                                    if device == "cuda" else "")
            return pick, why, cached
    why = f"按显存 {vram}MB 选择" if device == "cuda" else "CPU 推理选择"
    if recommended in _APPROX_BYTES:
        why += f"，将下载约 {human_size(_APPROX_BYTES[recommended])}"
    return recommended, why, cached


def _ensure_disk_space(download_root: str, size: str) -> None:
    """下载前预检磁盘空间：不够就早失败，别下到一半撑爆。"""
    need = _APPROX_BYTES.get(size)
    if not need:
        return
    probe = Path(download_root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return  # 问不出空间就别拦路
    if free < need * 1.2:
        raise RuntimeError(
            f"模型 {size} 约需 {human_size(need)}，{probe} 仅剩 "
            f"{human_size(free)}，空间不足。用 --whisper-download-root 把"
            f"模型目录指到空间充裕的盘，或换小一档的 --model-size。")


def _looks_like_download_failure(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(k in text for k in (
        "ConnectError", "ConnectionError", "LocalEntryNotFoundError",
        "NewConnectionError", "MaxRetryError", "timed out"))


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, *, model_size: str = "auto", device: str = "auto",
                 compute_type: str = "int8", language: str | None = "zh",
                 beam_size: int = 5, download_root: str | None = None,
                 hf_endpoint: str | None = None):
        endpoint = (hf_endpoint
                    or os.environ.get("VIDSUM_HF_ENDPOINT") or "").strip()
        if endpoint:
            # huggingface_hub 的 ENDPOINT 是 import 期常量，必须在
            # import faster_whisper（连带 huggingface_hub）之前设好
            os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 faster-whisper。执行 `pip install faster-whisper`，"
                "或改用 --asr api 走云端转写。"
            ) from exc

        if device == "auto":
            # 用 ctranslate2 自己探测 CUDA：import torch 只为问一句
            # “有没有卡”要付数 GB 的体积，还会和 ctranslate2 的静态
            # OpenMP 打架；exe 打包时 torch 也进不了 bundle。
            try:
                import ctranslate2
                device = ("cuda" if ctranslate2.get_cuda_device_count() > 0
                          else "cpu")
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
        # 未显式指定时默认把模型放在程序根目录下的 models\whisper，
        # 与 sherpa 后端的 models\sherpa 约定保持一致，避免每次传参。
        # 用 app_root() 而非 __file__：exe 冻结后 models\whisper 放在
        # exe 旁边就能被找到。
        if download_root is None:
            download_root = str(app_root() / "models" / "whisper")
        # "auto" 时按显存和本地缓存定档位；设备/精度/目录一起落一行日志——
        # GPU 用没用上、模型从哪来是用户最关心的两件事
        model_size, why, cached = _resolve_auto_size(
            model_size, device=device, download_root=download_root)
        if model_size not in cached:
            _ensure_disk_space(download_root, model_size)
        log(f"[faster-whisper] 设备={device} 精度={compute_type} "
            f"模型={model_size}（{why}）目录={download_root}")
        try:
            self._model = WhisperModel(
                model_size, device=device, compute_type=compute_type,
                download_root=download_root,
            )
        except Exception as exc:
            if not endpoint and _looks_like_download_failure(exc):
                raise RuntimeError(
                    f"模型下载失败：{exc}。国内网络访问 huggingface.co 不"
                    f"通时，加 --hf-endpoint https://hf-mirror.com 或设"
                    f"环境变量 VIDSUM_HF_ENDPOINT 后重试。"
                ) from exc
            raise

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
