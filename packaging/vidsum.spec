# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 规格：把 vidsum 打成带控制台的 vidsum.exe（onedir 模式）。

构建：  E:\\anaconda\\envs\\testenv\\python.exe -m PyInstaller packaging\\vidsum.spec ^
            --distpath dist --workpath build
或：    powershell -ExecutionPolicy Bypass -File packaging\\build.ps1

要点：
- 为什么是 onedir 而不是 onefile：onefile 的 bootloader 启动时要把整个
  CArchive 自解压到 %TEMP%\\_MEIxxxx，本机火绒 Hips 行为防护会拦掉这个
  "未知 exe 往磁盘释放 DLL" 的动作，报 `Failed to extract xxx.dll:
  failed to open target file`，exe 根本起不来。onedir 的 DLL 构建期就
  落在 exe 旁边，运行期一个字节都不往 temp 写，顺带省掉 onefile 每次
  启动数十秒的自解压延迟。
- console=True 不能关：Web 控制台的实时日志走 stdout，关了只剩空窗口
- torch 全家整包排除：体积数 GB，且本项目已不依赖它
- models\\ 与 vidsum-out\\ 不打进 bundle：放 exe 旁边即可被 app_root() 找到
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

# 构建期只要有子进程 import 带 OpenMP 的原生包（ctranslate2、onnxruntime
# 各自静态链接了一份 libiomp5md.dll），两份撞上就是 OMP Error #15 直接退出。
# 提前设好让 OpenMP 容忍重复初始化。
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(SPECPATH).parent

datas = []
binaries = []
hiddenimports = [
    # 分发与两种子进程形态
    "vidsum.entry",
    "vidsum.webapp",
    "vidsum._job_runner",
    # 四个 ASR 后端：get_backend() 是函数内动态 import，显式列出保险
    "vidsum.asr_api",
    "vidsum.asr_dummy",
    "vidsum.asr_faster",
    "vidsum.asr_sherpa",
    # 后端的二进制/数据依赖（纯 Python 部分 analysis 自己能跟到）
    "faster_whisper",
    # faster-whisper 运行时按属性访问 ctranslate2.models.Whisper，
    # import 图跟不到这一层，必须显式列出
    "ctranslate2.models",
    # faster_whisper.vad 是函数内 import onnxruntime（延迟导入），静态
    # 分析看不到；而 asr_faster 传了 vad_filter=True，不带上它冻结后
    # 第一段转写就报 "Applying the VAD filter requires the onnxruntime"
    "onnxruntime",
]

# 原生包的 DLL 收集。用 collect_dynamic_libs 而不是 collect_all：后者会
# 枚举全部子模块，ctranslate2.converters.transformers 会把 transformers/
# sklearn/cv2 整条链拖进 bundle，还要起隔离子进程 import 包——之前
# cudnn64_9.dll 丢失正是那条路上的 OMP #15 把收集结果静默吞掉的。
# av 不用动手：contrib 的 hook-av.py 会收 av.libs 全部 DLL 和子模块。
for pkg in ("ctranslate2", "sherpa_onnx", "onnxruntime"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception as exc:
        # 收集失败 = 该后端的 DLL 进不了 bundle，运行时才炸，必须嚷嚷
        print(f"[vidsum.spec] 警告：collect_dynamic_libs({pkg}) 失败：{exc}")
        continue  # 环境里没装这个后端就跳过，其余照打

# conda 的 _ctypes.pyd 链接 <env>\Library\bin\ffi.dll（libffi），该目录
# 只进 PATH、不进 sys.path，PyInstaller 的依赖扫描够不着，冻结后第一次
# import ctypes 就炸（ctranslate2\__init__.py 第 4 行就要 ctypes），
# 报错还是句 "DLL load failed while importing _ctypes" 很难联想到这。
# 官方 python 把 libffi 放 DLLs\ 里能被扫到；conda 得显式兜底。
import sys as _sys

_ffi = Path(_sys.executable).resolve().parent / "Library" / "bin" / "ffi.dll"
if _ffi.is_file():
    binaries.append((str(_ffi), "."))
else:
    print("[vidsum.spec] 警告：没找到 ffi.dll，冻结后 import ctypes 可能失败")

# 这些包不在运行路径上，但 PyInstaller 的静态分析会跟进函数体内的 import，
# 把整条科研栈拖进 bundle（还会让隔离子进程 import scipy 的 torch 兼容层时
# 撞 OMP Error #15 直接死）。逐个点名排除：
# - onnxruntime：sherpa_onnx\lib 里自带 onnxruntime.dll，pip 版用不到；
#   它的 transformers/quantization 子包正是 pandas/scipy/matplotlib 的入口
# - onnxruntime.transformers/quantization：faster-whisper 的 VAD 只要
#   onnxruntime 核心包，这两个子包是 pandas/scipy/matplotlib 的入口
# - tqdm/rich/huggingface_hub 的可选分支：pandas()/notebook/jupyter 展示
EXCLUDES = [
    "torch", "torchvision", "torchaudio",
    "onnxruntime.transformers", "onnxruntime.quantization",
    # 只可能经 ctranslate2.converters.transformers（模型格式转换器）进来，
    # 转写路径完全不碰；cv2/sklearn 又是 transformers 的可选依赖
    "transformers", "sklearn", "cv2", "regex",
    "pandas", "pyarrow", "scipy", "matplotlib", "matplotlib_inline",
    "IPython", "tkinter",
    "jedi", "parso", "prompt_toolkit",
    "zmq", "tornado", "jupyter_client", "jupyter_core", "ipykernel",
    "debugpy", "comm", "openpyxl", "lxml", "notebook",
]

# Web 前端：frozen 下 vidsum/web 必须在 bundle 里，否则控制台开出来是 404
datas.append((str(PROJECT_ROOT / "vidsum" / "web"), "vidsum/web"))

# faster_whisper 的 VAD 权重是包内数据文件（assets\silero_vad_v6.onnx），
# PyInstaller 只收 .py 不收 .onnx。asr_faster 传了 vad_filter=True，
# 缺这个文件每段转写都会倒在
# "Load model ... silero_vad_v6.onnx failed: File doesn't exist"。
# 用 find_spec 定位而不是 import：spec 执行期不该真加载这个包。
import importlib.util as _ilu

_fw_spec = _ilu.find_spec("faster_whisper")
_fw_assets = (Path(list(_fw_spec.submodule_search_locations)[0]) / "assets"
              if _fw_spec and _fw_spec.submodule_search_locations else None)
if _fw_assets and _fw_assets.is_dir():
    datas.append((str(_fw_assets), "faster_whisper/assets"))
else:
    print("[vidsum.spec] 警告：没找到 faster_whisper/assets，冻结后 VAD 会缺权重")

a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="vidsum",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="vidsum",
)
