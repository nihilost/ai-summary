import unittest
import types
from unittest import mock

from pathlib import Path
from tests.support import TempDirCase
from vidsum import media


ENCODERS_OUTPUT = """\
Encoders:
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC
 A..... aac                  AAC (Advanced Audio Coding)
 A....D libmp3lame           MP3 (MPEG audio layer 3)
 A....D libopus              Opus
 ------
"""


class AvailableEncodersTest(unittest.TestCase):
    def test_parses_only_encoder_lines(self):
        with mock.patch.object(media, "_run") as run:
            run.return_value.stdout = ENCODERS_OUTPUT
            names = media.available_encoders("ffmpeg")
        self.assertEqual(names, {"libx264", "aac", "libmp3lame", "libopus"})

    def test_empty_output(self):
        with mock.patch.object(media, "_run") as run:
            run.return_value.stdout = ""
            self.assertEqual(media.available_encoders("ffmpeg"), set())


class PickCodecTest(unittest.TestCase):
    def test_prefers_mp3_when_available(self):
        with mock.patch.object(media, "available_encoders",
                               return_value={"libmp3lame", "aac", "pcm_s16le"}):
            codec = media.pick_codec("ffmpeg")
        self.assertEqual(codec.name, "libmp3lame")
        self.assertEqual(codec.ext, "mp3")

    def test_falls_back_when_mp3_missing(self):
        with mock.patch.object(media, "available_encoders",
                               return_value={"pcm_s16le"}):
            codec = media.pick_codec("ffmpeg")
        self.assertEqual(codec.name, "pcm_s16le")
        self.assertEqual(codec.ext, "wav")

    def test_explicit_preference_is_honoured(self):
        with mock.patch.object(media, "available_encoders",
                               return_value={"libmp3lame", "libopus"}):
            codec = media.pick_codec("ffmpeg", prefer="libopus")
        self.assertEqual(codec.name, "libopus")

    def test_unknown_preference_raises(self):
        with self.assertRaises(media.MediaError):
            media.pick_codec("ffmpeg", prefer="nope")

    def test_nothing_available_raises(self):
        with mock.patch.object(media, "available_encoders", return_value=set()):
            with self.assertRaises(media.MediaError):
                media.pick_codec("ffmpeg")


class FindToolTest(unittest.TestCase):
    def setUp(self):
        from tests.support import TempDirCase
        self._dir = TempDirCase.make_dir(self)
        self.exe = self._dir / "ffmpeg.exe"
        self.exe.write_bytes(b"MZ")

    def test_explicit_path_wins(self):
        self.assertEqual(media.find_tool("ffmpeg", str(self.exe)), str(self.exe))

    def test_env_var_is_used(self):
        with mock.patch.dict("os.environ", {"VIDSUM_FFMPEG": str(self.exe)}):
            self.assertEqual(media.find_tool("ffmpeg"), str(self.exe))

    def test_not_found_raises_with_hint(self):
        with mock.patch.object(media.shutil, "which", return_value=None), \
                mock.patch.object(media, "_tool_dirs", return_value=[]):
            with self.assertRaises(media.MediaError) as ctx:
                media.find_tool("ffmpeg")
        self.assertIn("winget", str(ctx.exception))

    def test_unknown_tool_name(self):
        with self.assertRaises(media.MediaError):
            media.find_tool("ffplay")


class SilenceParsingTest(unittest.TestCase):
    STDERR = (
        "[silencedetect @ 0x1] silence_start: 1.5\n"
        "[silencedetect @ 0x1] silence_end: 3.25 | silence_duration: 1.75\n"
        "[silencedetect @ 0x1] silence_start: 10\n"
    )

    def test_pairs_starts_with_ends(self):
        with mock.patch.object(media, "_run") as run:
            run.return_value.stderr = self.STDERR
            run.return_value.stdout = ""
            spans = media.detect_silences("ffmpeg", "a.mp3")
        self.assertEqual(spans, [(1.5, 3.25), (10.0, 10.5)])


class GrabFramesTest(TempDirCase):
    """抽帧是增强步骤：命令形状要稳，单帧失败不能掀翻整批。"""

    def setUp(self):
        self.work = self.make_dir("grabframes-")
        self.commands = []

        def fake_run(cmd, *, timeout=900.0, check=True):
            self.commands.append(list(cmd))
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        def writing_run(cmd, *, timeout=900.0, check=True):
            fake_run(cmd)
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        self.fake_run = fake_run
        self.writing_run = writing_run

    def _grab(self, jobs, run, **kwargs):
        with mock.patch.object(media, "_run", side_effect=run):
            return media.grab_frames("ffmpeg", "lecture.mp4", jobs, **kwargs)

    def test_seek_precedes_input_and_frame_options(self):
        jobs = [(self.work / "frames" / "000" / "0100.500.jpg", 100.5)]
        self._grab(jobs, self.writing_run)
        cmd = self.commands[0]
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertIn("-frames:v", cmd)
        self.assertIn("1", cmd)
        self.assertIn("scale=480:-2", cmd)
        self.assertIn("-q:v", cmd)
        self.assertEqual(cmd[-1], str(jobs[0][0]))

    def test_headers_and_width_are_passed(self):
        jobs = [(self.work / "a.jpg", 5.0)]
        self._grab(jobs, self.writing_run, headers="Referer: x\r\n",
                   width=640)
        cmd = self.commands[0]
        self.assertIn("-headers", cmd)
        self.assertIn("Referer: x\r\n", cmd)
        self.assertIn("scale=640:-2", cmd)

    def test_saved_paths_returned_in_order(self):
        jobs = [(self.work / "a.jpg", 1.0), (self.work / "b.jpg", 2.0)]
        saved = self._grab(jobs, self.writing_run)
        self.assertEqual([p.name for p in saved], ["a.jpg", "b.jpg"])

    def test_empty_output_file_is_skipped(self):
        jobs = [(self.work / "a.jpg", 1.0)]
        saved = self._grab(jobs, self.fake_run)
        self.assertEqual(saved, [])

    def test_media_error_does_not_stop_the_batch(self):
        jobs = [(self.work / "a.jpg", 1.0), (self.work / "b.jpg", 2.0)]

        def flaky_run(cmd, *, timeout=900.0, check=True):
            self.commands.append(list(cmd))
            if cmd[-1].endswith("a.jpg"):
                raise media.MediaError("定位失败")
            Path(cmd[-1]).write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        saved = self._grab(jobs, flaky_run)
        self.assertEqual([p.name for p in saved], ["b.jpg"])
        self.assertEqual(len(self.commands), 2)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
