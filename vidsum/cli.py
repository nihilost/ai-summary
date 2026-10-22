"""命令行入口。"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def default_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(4, cpu // 2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidsum",
        description="长视频转写 + 滚动摘要：边转写边输出总结，内存与视频时长无关。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m vidsum lecture.mp4 --asr api --asr-model whisper-large-v3\n"
            "  python -m vidsum lecture.mp4 --asr faster-whisper --model-size medium\n"
            "  python -m vidsum lecture.mp4 --asr sherpa\n"
            "  python -m vidsum https://example.com/x.m3u8 --dry-run\n"
        ),
    )
    parser.add_argument("input", help="本地视频/音频文件路径，或 http(s)/m3u8 直链")
    parser.add_argument("-o", "--output", help="输出根目录（默认 ./vidsum-out）")

    media = parser.add_argument_group("音频与分段")
    media.add_argument("--chunk-minutes", type=float, default=10.0,
                       help="每段目标时长（分钟），默认 10")
    media.add_argument("--no-silence-split", action="store_true",
                       help="关闭静音对齐，按固定时长硬切")
    media.add_argument("--silence-db", type=float, default=30.0,
                       help="静音判定阈值（dB），默认 30")
    media.add_argument("--min-silence", type=float, default=0.5,
                       help="计入静音的最短时长（秒），默认 0.5")
    media.add_argument("--audio-codec", choices=["libmp3lame", "libopus", "aac",
                                                 "pcm_s16le"],
                        help="音频编码器（默认按可用性自动选择）")
    media.add_argument("--audio-backend", choices=["auto", "ffmpeg", "pyav"],
                        default="auto",
                        help="音频处理走系统 ffmpeg 还是 PyAV，默认 auto"
                             "（有 ffmpeg 就用，否则退回 PyAV）")
    media.add_argument("--ffmpeg", help="ffmpeg 可执行文件路径")
    media.add_argument("--ffprobe", help="ffprobe 可执行文件路径")
    media.add_argument("--keep-audio", action="store_true",
                       help="保留整段音频缓存（默认转写完删除）")

    asr = parser.add_argument_group("语音识别")
    asr.add_argument("--asr", default="faster-whisper",
                     choices=["faster-whisper", "api", "dummy", "sherpa"],
                     help="转写后端，默认 faster-whisper")
    asr.add_argument("--model-size", default="medium",
                     help="faster-whisper 模型大小，默认 medium")
    asr.add_argument("--device", default="auto",
                     help="推理设备 auto/cpu/cuda，默认 auto")
    asr.add_argument("--compute-type", default="int8",
                     help="计算精度 int8/float16，默认 int8")
    asr.add_argument("--whisper-download-root",
                     help="faster-whisper 模型的下载/缓存目录，默认用项目目录"
                          "下的 models\\whisper；想放别处就指过来，"
                          "如 E:\\whisper-models")
    asr.add_argument("--language", default="zh", help="音频语言，默认 zh")
    asr.add_argument("--workers", type=int, default=default_workers(),
                     help="并发转写数，默认 min(4, CPU/2)")
    asr.add_argument("--keep-chunks", action="store_true",
                     help="保留各段音频切片（默认转写后删除）")
    asr.add_argument("--no-boundary-dedup", action="store_true",
                     help="关闭跨段边界的重复文本去除")
    asr.add_argument("--asr-base-url", default="https://api.openai.com/v1",
                     help="--asr api 使用的接口地址")
    asr.add_argument("--asr-api-key", help="--asr api 使用的 API Key")
    asr.add_argument("--asr-model", default="whisper-1",
                     help="--asr api 使用的模型名")
    asr.add_argument("--sherpa-model-dir",
                     help="--asr sherpa 的模型目录（含 asr/ punc/ vad/ 子目录），"
                          "默认自动搜索，也可用环境变量 VIDSUM_SHERPA_MODEL_DIR")
    asr.add_argument("--sherpa-threads", type=int, default=2,
                     help="--asr sherpa 的推理线程数，默认 2")
    asr.add_argument("--no-punctuation", action="store_true",
                     help="--asr sherpa 关闭标点恢复")

    llm = parser.add_argument_group("总结")
    llm.add_argument("--no-summary", action="store_true", help="只转写，不总结")
    llm.add_argument("--llm-base-url", help="LLM 接口地址")
    llm.add_argument("--llm-api-key", help="LLM API Key")
    llm.add_argument("--llm-model", help="LLM 模型名")
    llm.add_argument("--llm-temperature", type=float, default=0.3)
    llm.add_argument("--llm-max-tokens", type=int, default=2048)

    frame = parser.add_argument_group("画面处理")
    frame.add_argument("--no-frames", action="store_true",
                       help="关闭抽帧（默认每段抽关键帧并插入终稿附录）")
    frame.add_argument("--frames-per-segment", type=int, default=2,
                       help="每段抽取的关键帧数，默认 2")
    frame.add_argument("--frame-width", type=int, default=480,
                       help="关键帧宽度（像素，等比缩放），默认 480")
    frame.add_argument("--vision", action="store_true",
                       help="总结时把关键帧喂给多模态 LLM"
                            "（需要模型支持图片输入，如 gpt-4o / "
                            "gpt-4o-mini / claude / qwen-vl）")

    ctl = parser.add_argument_group("流程控制")
    ctl.add_argument("--force", action="store_true",
                     help="忽略已有缓存，从头重跑")
    ctl.add_argument("--dry-run", action="store_true",
                     help="只抽音频并打印分段计划，不转写（音频会被缓存复用）")
    ctl.add_argument("--user-agent", help="网络源请求用的 User-Agent")
    ctl.add_argument("--referer", help="网络源请求用的 Referer")
    ctl.add_argument("--cookie", help="网络源请求用的 Cookie")
    ctl.add_argument("-V", "--version", action="version",
                     version=f"vidsum {__version__}")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_minutes <= 0:
        print("错误：--chunk-minutes 必须为正数", file=sys.stderr)
        return 2

    from . import pipeline
    try:
        return pipeline.run(args)
    except KeyboardInterrupt:
        print("\n已中断。重跑同一命令可从断点继续。", file=sys.stderr)
        return 130
    except Exception as exc:  # 统一出口：把异常翻成可读的一行
        print(f"错误：{exc}", file=sys.stderr)
        return 1
