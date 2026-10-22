"""端到端联调：mock 掉 ffmpeg 与 LLM 接口，跑通整条流水线。"""

import json
import contextlib
import io
import re
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.support import TempDirCase
from vidsum import cli, media, pipeline, render
from vidsum.media import Codec
from vidsum import frames as frames_mod

DURATION = 2000.0
FAKE_CODEC = Codec("libmp3lame", "mp3", ("-c:a", "libmp3lame", "-q:a", "5"))


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _completion(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class CleanSourceTest(unittest.TestCase):
    def test_unwraps_matching_quotes(self):
        self.assertEqual(pipeline._clean_source('"demo.mp4"'), "demo.mp4")
        self.assertEqual(pipeline._clean_source("'demo.mp4'"), "demo.mp4")
        self.assertEqual(pipeline._clean_source('  "C:\\a b\\x.mp4"  '),
                         "C:\\a b\\x.mp4")
        self.assertEqual(pipeline._clean_source('""x.mp4""'), "x.mp4")

    def test_leaves_other_input_alone(self):
        self.assertEqual(pipeline._clean_source("demo.mp4"), "demo.mp4")
        self.assertEqual(pipeline._clean_source("https://x/y.mp4"),
                         "https://x/y.mp4")
        self.assertEqual(pipeline._clean_source('"demo.mp4'), '"demo.mp4')
        self.assertEqual(pipeline._clean_source(""), "")
        self.assertEqual(pipeline._clean_source(None), "")


class PipelineTestCase(TempDirCase):
    def test_quoted_path_is_unwrapped(self):
        code = cli.main(['"demo.mp4"', "-o", str(self.out), "--asr", "dummy",
                         "--keep-audio", "--llm-api-key", "k"])
        self.assertEqual(code, 0)
        self.assertTrue((self.work / "final.md").is_file())
        meta = json.loads((self.work / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["input"], "demo.mp4")

    def setUp(self):
        self.out = self.make_dir("pipeline-")
        self.llm_calls = []
        self.cut_calls = []
        self.grab_calls = []

        def fake_urlopen(request, timeout=None):
            self.llm_calls.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(_completion(f"模型回复 {len(self.llm_calls)}"))

        def fake_grab(ffmpeg, source, jobs, *, headers=None, width=480,
                      timeout=600.0):
            saved = []
            for dst, _seconds in jobs:
                dst = Path(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
                saved.append(dst)
            self.grab_calls.append([(dst, seconds) for dst, seconds in jobs])
            return saved

        def fake_cut(ffmpeg, src, dst, start, end, *, codec, **kwargs):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            self.cut_calls.append((start, end))
            Path(dst).write_bytes(b"RIFFfake")
            return Path(dst)

        def fake_extract(ffmpeg, src, dst, **kwargs):
            Path(dst).write_bytes(b"ID3fake")
            return Path(dst)

        patches = [
            mock.patch.object(
                media, "find_tool",
                side_effect=lambda name, explicit=None: f"fake-{name}"),
            mock.patch.object(media, "probe_duration", return_value=DURATION),
            mock.patch.object(media, "pick_codec", return_value=FAKE_CODEC),
            mock.patch.object(media, "extract_audio", side_effect=fake_extract),
            mock.patch.object(media, "detect_silences", return_value=[]),
            mock.patch.object(media, "cut_segment", side_effect=fake_cut),
            mock.patch.object(media, "grab_frames", side_effect=fake_grab),
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_tool(self, *extra):
        return cli.main(["demo.mp4", "-o", str(self.out), "--asr", "dummy",
                         "--keep-audio", *extra])

    @property
    def work(self):
        return self.out / "demo"

    def test_full_run(self):
        code = self.run_tool("--llm-api-key", "k", "--llm-model", "m")
        self.assertEqual(code, 0)

        plan = json.loads((self.work / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(len(plan["segments"]), 4)  # 2000s / 600s 目标 -> 4 段

        # 每段 4 句假转写，共 16 句
        transcript = (self.work / "transcript.md").read_text(encoding="utf-8")
        self.assertEqual(transcript.count("**["), 16)
        self.assertIn("这一节我们先回顾上节课留下的问题。", transcript)

        # 4 次滚动更新 + 1 次收尾
        self.assertEqual(len(self.llm_calls), 5)
        self.assertEqual(len(self.cut_calls), 4)

        final = (self.work / "final.md").read_text(encoding="utf-8")
        self.assertIn("# demo", final)
        self.assertIn("## 附录：分段索引", final)

        progress = (self.work / "summary.md").read_text(encoding="utf-8")
        self.assertIn("4/4 段已完成", progress)

        state = json.loads(
            (self.work / "summary_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["next_index"], 4)

    def test_resume_skips_finished_work(self):
        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        first_calls = len(self.llm_calls)
        first_cuts = len(self.cut_calls)

        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        self.assertEqual(len(self.llm_calls), first_calls)
        self.assertEqual(len(self.cut_calls), first_cuts)
        self.assertTrue((self.work / "transcript.md").is_file())

    def test_resume_after_partial_summary(self):
        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        state_path = self.work / "summary_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["next_index"] = 2  # 模拟只总结到第 2 段就中断
        state_path.write_text(json.dumps(state), encoding="utf-8")
        calls_before = len(self.llm_calls)

        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        # 只补第 3、4 段的更新 + 收尾，共 3 次
        self.assertEqual(len(self.llm_calls) - calls_before, 3)

    def test_empty_finalize_reply_fails_loudly(self):
        """收尾那轮返回空正文（推理模型思考吃满预算）时要报错退出。"""
        calls = {"n": 0}

        def flaky(request, timeout=None):
            calls["n"] += 1
            if calls["n"] >= 5:  # 前 4 次是滚动更新，第 5 次起是收尾
                return FakeResponse({"choices": [
                    {"message": {"role": "assistant", "content": ""},
                     "finish_reason": "length"}]})
            return FakeResponse(_completion(f"模型回复 {calls['n']}"))

        stderr = io.StringIO()
        with mock.patch("urllib.request.urlopen", side_effect=flaky), \
                contextlib.redirect_stderr(stderr):
            code = self.run_tool("--llm-api-key", "k", "--llm-model", "m")
        self.assertEqual(code, 1)
        message = stderr.getvalue()
        self.assertIn("空内容", message)
        self.assertIn("--llm-max-tokens", message)
        # 转写稿和滚动摘要都还在盘上，重跑只会补收尾这一轮
        self.assertTrue((self.work / "transcript.md").is_file())
        self.assertTrue((self.work / "summary.md").is_file())

    def test_dry_run_stops_before_transcription(self):
        code = self.run_tool("--dry-run")
        self.assertEqual(code, 0)
        self.assertTrue((self.work / "plan.json").is_file())
        self.assertFalse((self.work / "transcript.md").exists())
        self.assertFalse((self.work / "chunks").exists())
        self.assertEqual(len(self.llm_calls), 0)
        self.assertEqual(len(self.cut_calls), 0)

    def _manifest(self):
        return json.loads(
            (self.work / "frames.json").read_text(encoding="utf-8"))

    def _plan_bounds(self):
        plan = json.loads((self.work / "plan.json").read_text(encoding="utf-8"))
        return {s["index"]: (s["start"], s["end"]) for s in plan["segments"]}

    def test_frames_grabbed_and_linked_in_final(self):
        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        manifest = self._manifest()
        self.assertEqual(manifest["width"], 480)
        # 4 段各抽一次，每段 2 帧
        self.assertEqual(len(self.grab_calls), 4)
        self.assertEqual([len(jobs) for jobs in self.grab_calls], [2, 2, 2, 2])
        self.assertEqual(len(manifest["frames"]), 4)
        self.assertEqual(sum(len(v) for v in manifest["frames"].values()), 8)

        # 帧文件名即时间戳，必须落在对应段内
        bounds = self._plan_bounds()
        for idx, rels in manifest["frames"].items():
            for rel in rels:
                t = frames_mod.frame_time(rel)
                self.assertGreaterEqual(t, bounds[int(idx)][0])
                self.assertLessEqual(t, bounds[int(idx)][1])

        final = (self.work / "final.md").read_text(encoding="utf-8")
        self.assertIn("![第 1 段画面", final)
        self.assertIn("frames/000/", final)

    def test_no_frames_flag_skips_grabbing(self):
        self.assertEqual(self.run_tool("--no-frames", "--llm-api-key", "k"), 0)
        self.assertEqual(self.grab_calls, [])
        self.assertFalse((self.work / "frames.json").exists())
        final = (self.work / "final.md").read_text(encoding="utf-8")
        self.assertNotIn("![第", final)

    def test_frames_per_segment_controls_count(self):
        self.assertEqual(self.run_tool("--no-summary",
                                       "--frames-per-segment", "4"), 0)
        manifest = self._manifest()
        self.assertEqual(sum(len(v) for v in manifest["frames"].values()), 16)

    def test_vision_sends_multimodal_messages(self):
        self.assertEqual(self.run_tool("--vision", "--llm-api-key", "k"), 0)
        # 4 次滚动更新走多模态，收尾一次仍是纯文本
        multimodal = [c for c in self.llm_calls
                      if isinstance(c["messages"][1]["content"], list)]
        self.assertEqual(len(multimodal), 4)
        parts = self.llm_calls[0]["messages"][1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(len(parts), 3)  # 文本 + 2 帧
        for part in parts[1:]:
            self.assertEqual(part["type"], "image_url")
            self.assertTrue(part["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"))
        self.assertIsInstance(self.llm_calls[-1]["messages"][1]["content"], str)

    def test_frames_reused_on_resume(self):
        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        first = len(self.grab_calls)
        self.assertEqual(first, 4)

        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        self.assertEqual(len(self.grab_calls), first)  # 命中 frames.json 缓存
        self.assertTrue((self.work / "frames.json").is_file())

    def test_transcript_grows_as_chunks_finish(self):
        snapshots = []
        original = render.write_transcript_partial

        def spy(work_dir, title, chunks, *, total):
            snapshots.append(len(chunks))
            return original(work_dir, title, chunks, total=total)

        with mock.patch.object(render, "write_transcript_partial",
                               side_effect=spy):
            self.assertEqual(self.run_tool("--no-summary"), 0)
        self.assertEqual(snapshots, [1, 2, 3, 4])

        final = (self.work / "transcript.md").read_text(encoding="utf-8")
        self.assertNotIn("转写进行中", final)
        self.assertEqual(final.count("**["), 16)

    def test_transcript_timestamps_stay_within_each_chunk(self):
        # 每段的时间戳必须是「段起点 + 段内相对戳」。若在存盘时提前加了段起点，
        # 合并阶段会再加一次，越靠后的段整体后移越多——这里逐行卡住上界。
        self.assertEqual(self.run_tool("--no-summary"), 0)
        plan = json.loads((self.work / "plan.json").read_text(encoding="utf-8"))
        bounds = {s["index"]: (s["start"], s["end"]) for s in plan["segments"]}
        transcript = (self.work / "transcript.md").read_text(encoding="utf-8")
        rows = re.findall(r"\*\*\[(\d+):(\d{2}):(\d{2})\]\*\* (\d{3}):", transcript)
        self.assertEqual(len(rows), 16)  # 4 段 x 4 句
        for hour, minute, second, idx in rows:
            t = int(hour) * 3600 + int(minute) * 60 + int(second)
            lo, hi = bounds[int(idx)]
            self.assertGreaterEqual(t, lo)
            self.assertLessEqual(
                t, hi, f"chunk {idx} 时间戳 {t}s 越过段终点 {hi}s（疑似二次偏移）")

    def test_no_summary_without_api_key(self):
        code = self.run_tool()
        self.assertEqual(code, 0)
        self.assertTrue((self.work / "transcript.md").is_file())
        self.assertFalse((self.work / "final.md").exists())
        self.assertEqual(len(self.llm_calls), 0)

    def test_no_summary_flag(self):
        code = self.run_tool("--no-summary", "--llm-api-key", "k")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.llm_calls), 0)
        self.assertTrue((self.work / "transcript.md").is_file())

    def test_force_wipes_previous_work(self):
        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        marker = self.work / "marker.txt"
        marker.write_text("keep?", encoding="utf-8")

        self.assertEqual(self.run_tool("--force", "--llm-api-key", "k"), 0)
        self.assertFalse(marker.exists())
        self.assertEqual(len(self.llm_calls), 10)

    def test_changing_chunk_size_invalidates_cache(self):
        self.assertEqual(self.run_tool("--no-summary"), 0)
        first_cuts = len(self.cut_calls)
        self.assertEqual(first_cuts, 4)  # 2000s / 600s 目标

        # 改成 5 分钟一段后分段边界全变，旧结果必须作废重转
        self.assertEqual(self.run_tool("--no-summary", "--chunk-minutes", "5"), 0)
        self.assertEqual(len(self.cut_calls) - first_cuts, 7)
        plan = json.loads((self.work / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(len(plan["segments"]), 7)
        transcript = (self.work / "transcript.md").read_text(encoding="utf-8")
        self.assertEqual(transcript.count("**["), 28)  # 7 段 x 4 句

    def test_audio_cache_is_reused(self):
        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        audio = self.work / "audio.mp3"
        self.assertTrue(audio.is_file())

        self.assertEqual(self.run_tool("--llm-api-key", "k"), 0)
        self.assertTrue(audio.is_file())

    def test_url_input_is_supported(self):
        code = cli.main(["https://example.com/course/lec7.mp4",
                         "-o", str(self.out), "--asr", "dummy",
                         "--keep-audio", "--no-summary"])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "lec7" / "transcript.md").is_file())

    def test_bilibili_page_resolves_to_audio_stream(self):
        probed = {}
        extracted = {}

        def spy_probe(ffprobe, source, *, headers=None):
            probed["source"] = source
            probed["headers"] = headers
            return DURATION

        def spy_extract(ffmpeg, src, dst, **kwargs):
            extracted["src"] = src
            Path(dst).write_bytes(b"ID3fake")
            return Path(dst)

        with mock.patch.object(media, "probe_duration", side_effect=spy_probe), \
                mock.patch.object(media, "extract_audio", side_effect=spy_extract), \
                mock.patch("vidsum.bili.resolve_audio_url",
                           return_value=("https://up.bili/hi.m4s",
                                         {"Referer": "https://www.bilibili.com/",
                                          "User-Agent": "UA-test"})):
            code = cli.main(["https://www.bilibili.com/video/BV1mnG6HECt",
                             "-o", str(self.out), "--asr", "dummy",
                             "--keep-audio", "--no-summary"])
        self.assertEqual(code, 0)
        # 探测与抽取都改用解析出的直音频流，并带上 B 站 Referer
        self.assertEqual(probed["source"], "https://up.bili/hi.m4s")
        self.assertEqual(extracted["src"], "https://up.bili/hi.m4s")
        self.assertIn("Referer: https://www.bilibili.com/", probed["headers"])
        self.assertIn("User-Agent: UA-test", probed["headers"])
        # 工作目录以 BV 号命名
        self.assertTrue((self.out / "BV1mnG6HECt" / "transcript.md").is_file())

    def test_invalid_chunk_minutes_rejected(self):
        self.assertEqual(cli.main(["demo.mp4", "-o", str(self.out),
                                   "--asr", "dummy",
                                   "--chunk-minutes", "0"]), 2)

    def test_media_error_is_reported_not_raised(self):
        with mock.patch.object(media, "probe_duration",
                               side_effect=media.MediaError("链接已失效")):
            self.assertEqual(self.run_tool(), 1)


ENCODERS_OUTPUT = """\
Encoders:
 ------
 A....D libmp3lame           MP3 (MPEG audio layer 3)
 ------
"""

SILENCE_OUTPUT = (
    "[silencedetect @ 0x1] silence_start: 480.5\n"
    "[silencedetect @ 0x1] silence_end: 483.2 | silence_duration: 2.7\n"
)


def _result(stdout="", stderr=""):
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)


class RealMediaLayerTest(TempDirCase):
    """只替换最底层的 media._run，上面的 ffmpeg 封装全部真实执行。

    这是不装 ffmpeg 时能做的最高保真联调：探测、抽音频、静音扫描、
    切片都跑真实代码路径，只有真正的进程调用被替掉。
    """

    def setUp(self):
        self.out = self.make_dir("realmedia-")
        self.commands = []

        def fake_run(cmd, *, timeout=900.0, check=True):
            self.commands.append(list(cmd))
            if "-show_entries" in cmd:
                return _result(stdout="2000.0\n")
            if "-encoders" in cmd:
                return _result(stdout=ENCODERS_OUTPUT)
            if "silencedetect" in " ".join(cmd):
                return _result(stderr=SILENCE_OUTPUT)
            # 只有抽取/切片才写输出文件，避免把无关命令的末参数当成路径
            out = cmd[-1]
            if "-i" in cmd and out not in ("-", "NUL", "/dev/null"):
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_bytes(b"FAKE" * 64)
            return _result()

        self.media_patcher = mock.patch.object(media, "_run",
                                              side_effect=fake_run)
        self.media_patcher.start()
        self.addCleanup(self.media_patcher.stop)

        self.tool_patcher = mock.patch.object(
            media, "find_tool",
            side_effect=lambda name, explicit=None: f"fake-{name}")
        self.tool_patcher.start()
        self.addCleanup(self.tool_patcher.stop)

        self.llm_patcher = mock.patch(
            "urllib.request.urlopen",
            side_effect=lambda request, timeout=None: FakeResponse(
                _completion("## 更新后的滚动摘要\n- 假摘要\n\n## 本段小标题\n假标题")))
        self.llm_patcher.start()
        self.addCleanup(self.llm_patcher.stop)

    def test_full_pipeline_with_real_media_wrappers(self):
        code = cli.main(["lec.mp4", "-o", str(self.out), "--asr", "dummy",
                         "--llm-api-key", "k"])
        self.assertEqual(code, 0)

        work = self.out / "lec"
        # 探测 + 编码器探测 + 抽音频 + 静音扫描 + 8 次抽帧 + 4 次切片
        self.assertEqual(len(self.commands), 16)
        self.assertIn("-show_entries", self.commands[0])
        self.assertIn("-encoders", self.commands[1])
        self.assertIn("-i", self.commands[2])
        self.assertNotIn("-ss", self.commands[2])  # 整段抽取不带切片参数
        self.assertIn("silencedetect", " ".join(self.commands[3]))
        # 抽帧排在转写之前：每段 2 帧，input seeking + 单帧 + 缩放
        for cmd in self.commands[4:12]:
            self.assertLess(cmd.index("-ss"), cmd.index("-i"))
            self.assertIn("-frames:v", cmd)
            self.assertIn("scale=480:-2", cmd)
        for cmd in self.commands[4:]:
            self.assertIn("-ss", cmd)
            self.assertIn("-i", cmd)

        manifest = json.loads(
            (work / "frames.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["width"], 480)
        self.assertEqual(sum(len(v) for v in manifest["frames"].values()), 8)
        for rels in manifest["frames"].values():
            for rel in rels:
                self.assertTrue((work / rel).is_file())

        transcript = (work / "transcript.md").read_text(encoding="utf-8")
        self.assertEqual(transcript.count("**["), 16)
        self.assertTrue((work / "final.md").is_file())

        # 480.5-483.2 的静音中点约 481.85，第一段切点应吸附到它附近
        plan = json.loads((work / "plan.json").read_text(encoding="utf-8"))
        first_cut = plan["segments"][0]["end"]
        self.assertAlmostEqual(first_cut, 481.85, delta=1.0)

    def test_cached_audio_skips_reextraction(self):
        self.assertEqual(cli.main(["lec.mp4", "-o", str(self.out),
                                   "--asr", "dummy", "--keep-audio",
                                   "--no-summary"]), 0)
        after_first = len(self.commands)

        self.assertEqual(cli.main(["lec.mp4", "-o", str(self.out),
                                   "--asr", "dummy", "--keep-audio",
                                   "--no-summary"]), 0)
        # 第二次只探测 + 复用音频 + 静音缓存 + 4 次切片（都没有待转写段）
        self.assertEqual(len(self.commands) - after_first, 2)

    def test_no_frames_issues_no_frame_commands(self):
        self.assertEqual(cli.main(["lec.mp4", "-o", str(self.out),
                                   "--asr", "dummy", "--no-summary",
                                   "--no-frames"]), 0)
        work = self.out / "lec"
        self.assertFalse((work / "frames.json").exists())
        self.assertFalse((work / "frames").exists())
        for cmd in self.commands:
            self.assertNotIn("-frames:v", cmd)

    def test_resume_reuses_grabbed_frames(self):
        self.assertEqual(cli.main(["lec.mp4", "-o", str(self.out),
                                   "--asr", "dummy", "--no-summary"]), 0)

        self.assertEqual(cli.main(["lec.mp4", "-o", str(self.out),
                                   "--asr", "dummy", "--no-summary"]), 0)
        # 第二次命中 frames.json 缓存，一条抽帧命令都不该再发
        frame_cmds = [c for c in self.commands if "-frames:v" in c]
        self.assertEqual(len(frame_cmds), 8)


if __name__ == "__main__":
    unittest.main()
