"""流水线编排：抽音频 → 静音切分 → 并行转写 → 滚动总结 → 终稿。

断点续跑由工作目录里的缓存文件自然实现：整段音频、每段转写 JSON、
摘要状态各自落盘，任一步骤中断后重跑同一命令即可接着来。
"""

from __future__ import annotations

import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import (__version__, asr as asr_mod, audio_av, bili, llm as llm_mod,
               frames, media, render, segment, summarize)
from .asr import ChunkResult
from .util import atomic_write_text, human_size, log, read_json, safe_name, write_json

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
PLAN_FILE = "plan.json"
SILENCE_FILE = "silences.json"
META_FILE = "meta.json"


def _looks_like_url(text: str) -> bool:
    return bool(_URL_RE.match(text or ""))


def _clean_source(raw: str) -> str:
    """剥掉填写路径时误粘的包裹引号（从资源管理器复制地址常带引号）。

    只剥成对出现的首尾引号；URL 不含引号，原样返回。
    """
    text = (raw or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _stem_for(source: str) -> str:
    if _looks_like_url(source):
        parsed = urlparse(source)
        name = Path(unquote(parsed.path)).stem
        return safe_name(name or parsed.netloc or "video")
    return safe_name(Path(source).stem)


def _header_map(args) -> dict:
    headers = {}
    if getattr(args, "user_agent", None):
        headers["User-Agent"] = args.user_agent
    if getattr(args, "referer", None):
        headers["Referer"] = args.referer
    if getattr(args, "cookie", None):
        headers["Cookie"] = args.cookie
    return headers


def _headers_str(headers: dict) -> str | None:
    return ("\r\n".join(f"{k}: {v}" for k, v in headers.items()) + "\r\n"
            if headers else None)


class _FfmpegAudio:
    """系统 ffmpeg 路：重活全部交给命令行二进制。"""
    name = "ffmpeg"

    def __init__(self, ffmpeg: str, ffprobe: str):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def probe_duration(self, source, headers):
        return media.probe_duration(self.ffprobe, source, headers=headers)

    def pick_codec(self, prefer):
        return media.pick_codec(self.ffmpeg, prefer)

    def extract_audio(self, source, dst, codec, headers):
        return media.extract_audio(self.ffmpeg, source, dst, codec=codec,
                                   headers=headers)

    def cut_segment(self, source, dst, start, end, codec):
        return media.cut_segment(self.ffmpeg, source, dst, start, end,
                                 codec=codec)

    def detect_silences(self, source, noise_db, min_silence):
        return media.detect_silences(self.ffmpeg, source, noise_db=noise_db,
                                     min_silence=min_silence)

    def grab_frames(self, source, jobs, headers, width):
        return media.grab_frames(self.ffmpeg, source, jobs,
                                 headers=headers, width=width)


class _PyavAudio:
    """PyAV 路：wheel 自带整套 FFmpeg 动态库，用户不必另装 ffmpeg。"""
    name = "pyav"

    def probe_duration(self, source, headers):
        return audio_av.probe_duration(source, headers=headers)

    def pick_codec(self, prefer):
        return audio_av.pick_codec(prefer)

    def extract_audio(self, source, dst, codec, headers):
        return audio_av.extract_audio(source, dst, codec=codec, headers=headers)

    def cut_segment(self, source, dst, start, end, codec):
        return audio_av.cut_segment(source, dst, start, end, codec=codec)

    def detect_silences(self, source, noise_db, min_silence):
        return audio_av.detect_silences(source, noise_db=noise_db,
                                        min_silence=min_silence)


def _resolve_audio(args):
    """按 --audio-backend 选一条音频处理路。

    auto 优先系统 ffmpeg（mp3 体积小），找不到就退回 PyAV。
    对本地文件两条路结果一致；PyAV 的价值在于只装 pip 包也能跑通。
    """
    prefer = getattr(args, "audio_backend", "auto") or "auto"
    if prefer in ("auto", "ffmpeg"):
        try:
            return _FfmpegAudio(
                media.find_tool("ffmpeg", getattr(args, "ffmpeg", None)),
                media.find_tool("ffprobe", getattr(args, "ffprobe", None)),
            )
        except media.MediaError:
            if prefer == "ffmpeg":
                raise
    if prefer == "pyav" and not audio_av.available():
        raise media.MediaError(
            "指定了 --audio-backend pyav 但 PyAV 不可用。执行 `pip install av`。"
        )
    if prefer == "auto" and not audio_av.available():
        raise media.MediaError(
            "既没找到 ffmpeg，也没装 PyAV。二者有其一即可："
            "`winget install Gyan.FFmpeg` 或 `pip install av`。"
        )
    if prefer == "auto":
        log("[音频] 未找到 ffmpeg，改用 PyAV 内置的 FFmpeg")
    return _PyavAudio()


def _load_or_plan_segments(work: Path, duration: float, args,
                           audio_path: Path, audio) -> list[segment.Segment]:
    target = max(60.0, float(args.chunk_minutes) * 60.0)
    minimum = max(30.0, target / 2.5)
    maximum = target * 2.0

    silences = None if args.no_silence_split else read_json(work / SILENCE_FILE)
    if silences is None and not args.no_silence_split:
        log(f"[静音] 扫描静音区间（阈值 -{args.silence_db:g}dB，最短 "
            f"{args.min_silence:g}s）...")
        t0 = time.time()
        spans = audio.detect_silences(
            audio_path, noise_db=args.silence_db, min_silence=args.min_silence,
        )
        silences = [list(s) for s in spans]
        write_json(work / SILENCE_FILE, silences)
        log(f"[静音] 命中 {len(silences)} 段，用时 {time.time() - t0:.1f}s")

    segments = segment.plan_segments(
        duration, target=target, minimum=minimum, maximum=maximum,
        silences=silences or [],
    )
    write_json(work / PLAN_FILE, {
        "duration": duration,
        "target": target,
        "minimum": minimum,
        "maximum": maximum,
        "segments": [vars(s) for s in segments],
    })
    return segments


def _grab_all_frames(work: Path, media_source, segments, audio, headers,
                     args) -> dict[int, list[str]]:
    """按分段计划抽关键帧，写 frames.json 清单，返回 {段号: [相对路径]}。

    续跑时清单存在且帧宽一致就直接复用，不重复抽帧。
    """
    want_width = int(getattr(args, "frame_width", 480) or 480)
    manifest = frames.load_manifest(work)
    if manifest["frames"] and manifest["width"] == want_width:
        total = sum(len(v) for v in manifest["frames"].values())
        log(f"[画面] 复用缓存 {total} 帧")
        return manifest["frames"]

    grab = getattr(audio, "grab_frames", None)
    if grab is None:
        log("[画面] 当前音频后端不支持抽帧，跳过（ffmpeg 后端支持）")
        return {}

    plan = frames.plan_frame_times(segments,
                                   getattr(args, "frames_per_segment", 2))
    result: dict[int, list[str]] = {}
    for seg in segments:
        times = plan.get(seg.index) or []
        if not times:
            continue
        jobs = [(frames.frame_path(work, seg.index, t), t)
                for t in times]
        saved = grab(media_source, jobs, headers, want_width)
        rels = [dst.relative_to(work).as_posix() for dst in saved]
        if rels:
            result[seg.index] = rels
        log(f"[画面] #{seg.index:03d} 抽得 {len(rels)}/{len(jobs)} 帧")
    frames.save_manifest(work, result, width=want_width)
    total = sum(len(v) for v in result.values())
    log(f"[画面] 共 {total} 帧（每段目标 "
        f"{getattr(args, 'frames_per_segment', 2)} 帧，宽 {want_width}px）")
    return result


def _chunk_matches_plan(path: Path, seg: segment.Segment) -> bool:
    """已缓存的分段结果是否仍对应当前分段计划。

    用户改了 --chunk-minutes 后分段边界会变，旧的结果必须作废重转，
    否则会把上一套切分的文本当成这一段的。
    """
    data = read_json(path)
    if not data:
        return False
    return (abs(float(data.get("start", -1.0)) - seg.start) < 0.5
            and abs(float(data.get("end", -1.0)) - seg.end) < 0.5)


def _transcribe_one(seg: segment.Segment, *, work: Path, audio_path: Path,
                    codec: media.Codec, audio,
                    backend: asr_mod.ASRBackend,
                    keep_chunks: bool) -> ChunkResult:
    """在worker线程内执行：切片 → 转写 → 立即落盘（崩溃也不丢已完成段落）。"""
    chunk_dir = work / "chunks"
    chunk_audio = chunk_dir / f"{seg.index:03d}.{codec.ext}"
    audio.cut_segment(audio_path, chunk_audio, seg.start, seg.end, codec=codec)
    try:
        utterances = backend.transcribe(chunk_audio)
    finally:
        if not keep_chunks:
            chunk_audio.unlink(missing_ok=True)

    result = ChunkResult(seg.index, seg.start, seg.end,
                         # 存段内相对时间：后端协议返回的就是相对该段起点的戳，
                         # merge_chunks / render_transcript_partial 会再统一加 chunk.start。
                         # 这里若提前加 seg.start，合并时会被二次偏移，越靠后的段错得越多。
                         [asr_mod.Utterance(u.start, u.end, u.text)
                          for u in utterances if u.text])
    write_json(chunk_dir / f"{seg.index:03d}.json", result.to_dict())
    return result


def run(args) -> int:
    started = time.time()
    source = _clean_source(args.input)
    is_url = _looks_like_url(source)
    header_map = _header_map(args)
    headers = _headers_str(header_map)
    media_source = source
    if bili.is_bilibili_page(source):
        log("[解析] 检测到 B 站视频页，正在取直音频流地址...")
        audio_url, bili_headers = bili.resolve_audio_url(
            source,
            user_agent=getattr(args, "user_agent", None),
            cookie=getattr(args, "cookie", None),
        )
        media_source = audio_url
        # Referer/UA 以 B 站 CDN 的要求为准，覆盖用户可能填的同名头
        header_map.update(bili_headers)
        headers = _headers_str(header_map)
        log("[解析] 已取得音频流，拉流将带上 Referer")
    # Web 控制台用：返回 True 时在段落边界中断，已完成的 chunks 仍可续跑。
    cancel = getattr(args, "cancel_check", None)

    audio = _resolve_audio(args)
    log(f"[音频] 后端={audio.name}")

    out_root = (Path(args.output).expanduser() if args.output
                else Path.cwd() / "vidsum-out")
    title = _stem_for(source)
    work = out_root / title

    if args.force and work.exists():
        log(f"[清理] --force 指定，删除旧工作目录 {work}")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    log(f"[工作目录] {work}")

    meta = read_json(work / META_FILE) or {}
    meta.update({"input": source, "is_url": is_url, "version": __version__,
                 "title": title,
                 "params": {"chunk_minutes": args.chunk_minutes,
                            "asr": args.asr, "model_size": args.model_size,
                            "language": args.language}})
    write_json(work / META_FILE, meta)

    log(f"[探测] 读取媒体信息：{media_source}")
    duration = audio.probe_duration(media_source, headers)
    log(f"[探测] 时长 {segment.fmt_ts(duration, always_hours=True)}"
        f"（{duration:.1f}s）")
    if duration <= 0:
        raise media.MediaError("媒体时长为 0，可能没有音轨或链接已失效")

    codec = audio.pick_codec(args.audio_codec)
    audio_path = work / f"audio.{codec.ext}"
    if audio_path.is_file() and audio_path.stat().st_size > 0:
        log(f"[音频] 复用缓存 {audio_path.name}"
            f"（{human_size(audio_path.stat().st_size)}）")
    else:
        log(f"[音频] 抽取 16kHz 单声道音轨（{codec.name}）...")
        t0 = time.time()
        audio.extract_audio(media_source, audio_path, codec, headers)
        log(f"[音频] 完成 {audio_path.name}"
            f"（{human_size(audio_path.stat().st_size)}，"
            f"{time.time() - t0:.1f}s）")

    segments = _load_or_plan_segments(work, duration, args, audio_path, audio)
    log(f"[分段] 共 {len(segments)} 段（目标 "
        f"{args.chunk_minutes:g} 分钟/段）")
    for seg in segments:
        log(f"        #{seg.index:03d} {segment.fmt_range(seg.start, seg.end)}"
            f"（{seg.duration / 60:.1f} 分钟）")

    if args.dry_run:
        log("[dry-run] 流程验证完成，未执行转写")
        return 0

    frame_map: dict[int, list[str]] = {}
    if not getattr(args, "no_frames", False):
        frame_map = _grab_all_frames(work, media_source, segments, audio,
                                     headers, args)

    backend = asr_mod.get_backend(
        args.asr,
        model_size=args.model_size, device=args.device,
        compute_type=args.compute_type, language=args.language,
        base_url=args.asr_base_url, api_key=args.asr_api_key,
        model=args.asr_model,
        model_dir=getattr(args, "sherpa_model_dir", None),
        num_threads=getattr(args, "sherpa_threads", 2),
        punctuation=not getattr(args, "no_punctuation", False),
        download_root=getattr(args, "whisper_download_root", None),
        hf_endpoint=getattr(args, "hf_endpoint", None),
    )

    chunk_dir = work / "chunks"
    pending = [s for s in segments
               if not _chunk_matches_plan(chunk_dir / f"{s.index:03d}.json", s)]
    if len(pending) != len(segments):
        log(f"[转写] {len(segments) - len(pending)} 段已有结果，跳过")
    log(f"[转写] 后端={backend.name}，并发={args.workers}，"
        f"待处理 {len(pending)} 段")

    if pending:
        t0 = time.time()
        completed: dict[int, ChunkResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(_transcribe_one, seg, work=work,
                            audio_path=audio_path, codec=codec, audio=audio,
                            backend=backend,
                            keep_chunks=args.keep_chunks): seg
                for seg in pending
            }
            finished = 0
            for future in as_completed(futures):
                seg = futures[future]
                if cancel is not None and cancel():
                    raise RuntimeError("run cancelled by user")
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"第 {seg.index:03d} 段"
                        f"（{segment.fmt_range(seg.start, seg.end)}）失败：{exc}"
                    ) from exc
                finished += 1
                completed[result.index] = result
                # 边转写边落盘：已完成段落按序写进 transcript.md
                render.write_transcript_partial(work, title,
                                                list(completed.values()),
                                                total=len(segments))
                log(f"[转写] #{seg.index:03d} 完成"
                    f"（{len(result.utterances)} 句，{finished}/{len(pending)}，"
                    f"累计 {time.time() - t0:.0f}s）")

    chunks = []
    for seg in segments:
        if _chunk_matches_plan(chunk_dir / f"{seg.index:03d}.json", seg):
            data = read_json(chunk_dir / f"{seg.index:03d}.json")
            if data:
                chunks.append(ChunkResult.from_dict(data))
    if not chunks:
        raise RuntimeError("没有任何转写结果，请检查 ASR 后端配置")

    merged = asr_mod.merge_chunks(chunks, dedup=not args.no_boundary_dedup)
    utterances = [u for _chunk, utts in merged for u in utts]
    log(f"[转写] 合并完成：{len(utterances)} 句，"
        f"{sum(len(u.text) for u in utterances)} 字")

    transcript_path = render.write_transcript(work, title, utterances)
    plain_path = work / "transcript.txt"
    atomic_write_text(plain_path, "\n".join(
        f"[{segment.fmt_ts(u.start, always_hours=True)}] {u.text}"
        for u in utterances) + "\n")
    log(f"[输出] 转写：{transcript_path.name} / {plain_path.name}")

    config = llm_mod.resolve_config(args)
    if args.no_summary or not config.available:
        reason = "--no-summary 已指定" if args.no_summary else "未配置 LLM API Key"
        log(f"[总结] 跳过（{reason}）。设置 --llm-api-key 或 "
            f"VIDSUM_LLM_API_KEY 后可生成摘要")
    else:
        log(f"[总结] 模型={config.model}，逐段滚动更新")
        state = summarize.load_state(work)
        t0 = time.time()
        changed = False
        for i, (chunk, kept) in enumerate(merged):
            if state.next_index > i:
                continue
            if cancel is not None and cancel():
                raise RuntimeError("run cancelled by user")
            changed = True
            images = None
            if getattr(args, "vision", False) and frame_map.get(chunk.index):
                images = [work / rel for rel in frame_map[chunk.index]]
            summarize.update_summary(config, state,
                                      " ".join(u.text for u in kept).strip(),
                                      i, len(merged),
                                      start=chunk.start, end=chunk.end,
                                      images=images)
            summarize.save_state(work, state)
            render.write_progress(work, title, state, total=len(chunks),
                                  source=source)
            log(f"[总结] 第 {i + 1}/{len(chunks)} 段已并入滚动摘要"
                f"（累计 {time.time() - t0:.0f}s）")

        if state.final and not changed:
            # 断点续跑且本轮没有新内容时，不必再花一次调用重写终稿
            log("[总结] 终稿已存在且无新增内容，跳过收尾")
            final_path = render.write_final(work, title, state, state.final,
                                             frames=frame_map)
        else:
            final_md = summarize.finalize(config, state, title)
            summarize.save_state(work, state)
            if not final_md.strip():
                log("[警告] 滚动摘要为空，终稿只有分段索引")
            final_path = render.write_final(work, title, state, final_md,
                                             frames=frame_map)
            log(f"[输出] 终稿：{final_path}")

    if not args.keep_audio:
        audio_path.unlink(missing_ok=True)
        log("[清理] 已删除音频缓存（--keep-audio 可保留）")

    log(f"[完成] 总耗时 {time.time() - started:.1f}s，输出目录 {work}")
    return 0
