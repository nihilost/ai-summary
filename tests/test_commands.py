"""直接断言 ffmpeg / ffprobe 的命令行构造。

不依赖真实 ffmpeg：把 media._run 换成记录器，既能验证参数顺序与取值，
又能顺带造出输出文件让上层逻辑跑通。
"""

import unittest
from pathlib import Path
from unittest import mock

from vidsum import media
from vidsum.media import Codec

MP3 = Codec("libmp3lame", "mp3", ("-c:a", "libmp3lame", "-q:a", "5"))
URL = "https://example.com/course/lec7.mp4"
HEADERS = "User-Agent: Mozilla/5.0\r\nReferer: https://example.com/\r\n"


class Result:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


SILENCE_STDERR = (
    "[silencedetect @ 0x1] silence_start: 480.5\n"
    "[silencedetect @ 0x1] silence_end: 483.2 | silence_duration: 2.7\n"
)


class CommandRecorder:
    """记录每次 _run 的命令，并按命令类型给出合理的假输出。"""

    def __init__(self):
        self.commands = []

    def __call__(self, cmd, *, timeout=900.0, check=True):
        self.commands.append(list(cmd))
        if "-show_entries" in cmd:
            return Result(stdout="2000.0\n")
        if "-encoders" in cmd:
            return Result(stdout=ENCODERS)
        if "silencedetect" in " ".join(cmd):
            return Result(stderr=SILENCE_STDERR)
        # 抽取 / 切片：把最后一个参数当输出路径造出来。
        # 只在带 -i 时这么做，否则会把无关命令的末参数误当成路径写出文件。
        out = cmd[-1]
        if "-i" in cmd and out not in ("-", "NUL", "/dev/null"):
            Path(out).write_bytes(b"FAKE" * 32)
        return Result()

    def find(self, needle):
        return next(c for c in self.commands if needle in c)


ENCODERS = """\
Encoders:
 ------
 A....D libmp3lame           MP3 (MPEG audio layer 3)
 ------
"""


class CommandConstructionTest(unittest.TestCase):
    def setUp(self):
        from tests.support import TempDirCase
        self._tmp = TempDirCase.make_dir(self)
        self.rec = CommandRecorder()
        patcher = mock.patch.object(media, "_run", side_effect=self.rec)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_probe_duration_puts_headers_before_input(self):
        media.probe_duration("ffprobe", URL, headers=HEADERS)
        cmd = self.rec.find("-show_entries")
        self.assertLess(cmd.index("-headers"), cmd.index("-show_entries"))
        self.assertIn(HEADERS, cmd)
        self.assertEqual(cmd[-1], URL)
        self.assertIn("format=duration", cmd)

    def test_extract_audio_flags_and_order(self):
        dst = self._tmp / "audio.mp3"
        media.extract_audio("ffmpeg", URL, dst, codec=MP3, headers=HEADERS)
        cmd = self.rec.find("-i")
        self.assertLess(cmd.index("-headers"), cmd.index("-i"))
        self.assertLess(cmd.index("-i"), cmd.index("-vn"))
        self.assertEqual(cmd[cmd.index("-i") + 1], URL)
        self.assertIn("-vn", cmd)
        self.assertIn("-map", cmd)
        self.assertIn("0:a:0?", cmd)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertEqual(cmd[-5:],
                         ["-c:a", "libmp3lame", "-q:a", "5", str(dst)])
        self.assertIn("libmp3lame", cmd)
        self.assertEqual(cmd[-1], str(dst))

    def test_cut_segment_seeks_before_input(self):
        src = self._tmp / "audio.mp3"
        src.write_bytes(b"x")
        dst = self._tmp / "chunks" / "000.mp3"
        media.cut_segment("ffmpeg", src, dst, 480.5, 960.25, codec=MP3)
        cmd = self.rec.commands[-1]
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-ss") + 1], "480.500")
        self.assertEqual(cmd[cmd.index("-t") + 1], "479.750")
        self.assertEqual(cmd[cmd.index("-i") + 1], str(src))
        self.assertEqual(cmd[-1], str(dst))

    def test_detect_silences_uses_null_sink(self):
        src = self._tmp / "audio.mp3"
        src.write_bytes(b"x")
        spans = media.detect_silences("ffmpeg", src, noise_db=30, min_silence=0.5)
        cmd = self.rec.commands[-1]
        self.assertEqual(cmd[cmd.index("-af") + 1],
                         "silencedetect=noise=-30dB:d=0.5")
        self.assertIn("-f", cmd)
        self.assertIn("null", cmd)
        self.assertEqual(cmd[-1], "NUL" if media.os.name == "nt" else "/dev/null")
        self.assertEqual(spans, [(480.5, 483.2)])

    def test_pick_codec_reads_real_encoder_list(self):
        self.assertEqual(media.pick_codec("ffmpeg").name, "libmp3lame")

    def test_extract_audio_rejects_empty_output(self):
        with mock.patch.object(media, "_run", return_value=Result()):
            dst = self._tmp / "empty.mp3"
            with self.assertRaises(media.MediaError) as ctx:
                media.extract_audio("ffmpeg", URL, dst, codec=MP3)
        self.assertIn("输出文件为空", str(ctx.exception))


class RunBehaviorTest(unittest.TestCase):
    """media._run 自身的错误处理。单独放一个类，避免被命令记录器挡住。"""

    def test_failure_raises_with_truncated_stderr(self):
        proc = Result(returncode=1, stderr="x" * 900)
        with mock.patch.object(media.subprocess, "run", return_value=proc):
            with self.assertRaises(media.MediaError) as ctx:
                media._run(["ffmpeg", "-version"])
        self.assertIn("退出码 1", str(ctx.exception))
        self.assertLess(len(str(ctx.exception)), 700)  # stderr 只保留尾部

    def test_missing_binary_raises_clear_error(self):
        with mock.patch.object(media.subprocess, "run",
                               side_effect=FileNotFoundError("no such file")):
            with self.assertRaises(media.MediaError) as ctx:
                media._run(["ffmpeg", "-version"])
        self.assertIn("找不到可执行文件", str(ctx.exception))

    def test_timeout_raises(self):
        import subprocess
        with mock.patch.object(
                media.subprocess, "run",
                side_effect=subprocess.TimeoutExpired("ffmpeg", 30)):
            with self.assertRaises(media.MediaError) as ctx:
                media._run(["ffmpeg", "-i", "x"], timeout=30)
        self.assertIn("超时", str(ctx.exception))

    def test_check_false_does_not_raise(self):
        proc = Result(returncode=1, stderr="boom")
        with mock.patch.object(media.subprocess, "run", return_value=proc):
            self.assertEqual(media._run(["ffmpeg"], check=False).returncode, 1)


if __name__ == "__main__":
    unittest.main()
