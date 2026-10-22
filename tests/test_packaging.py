"""打包（PyInstaller 冻结）相关行为：路径解析、子进程启动、入口分发。

这些是 exe 能否真正跑起来的关键路径，但全都不需要真的构建：
frozen 形态靠 patch sys.frozen / sys.executable 模拟。
"""

from __future__ import annotations

import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from vidsum import entry, webapp
from vidsum.asr_faster import FasterWhisperBackend
from vidsum._job_runner import _force_utf8_output
from vidsum.util import app_root

EXE_DIR = Path(__file__).resolve().parent.parent  # 开发态 = 项目根


def _fake_frozen(exe_path: str):
    """模拟 PyInstaller 冻结态：frozen 标记 + exe 路径。"""
    # create=True：解释器态下 sys.frozen 不存在，mock 默认不许 patch 不存在的属性
    return mock.patch.multiple(sys, frozen=True, executable=exe_path,
                               create=True)


class AppRootTest(unittest.TestCase):
    def test_dev_mode_is_project_root(self):
        self.assertEqual(app_root(), EXE_DIR)

    def test_frozen_mode_is_exe_dir(self):
        exe = str(EXE_DIR / "dist" / "vidsum.exe")
        with _fake_frozen(exe):
            self.assertEqual(app_root(), EXE_DIR / "dist")


class ChildCmdTest(unittest.TestCase):
    def test_dev_mode_uses_module(self):
        cmd = webapp._child_cmd("job.json")
        self.assertEqual(cmd, [sys.executable, "-m", "vidsum._job_runner",
                               "job.json"])


class JobRunnerOutputEncodingTest(unittest.TestCase):
    """冻结态子进程必须吐 UTF-8：PYTHONIOENCODING 在 exe 里被忽略。"""

    def _pipe_stream(self, encoding):
        raw = io.BytesIO()
        return io.TextIOWrapper(raw, encoding=encoding, newline=""), raw

    def test_pipe_output_forced_to_utf8(self):
        out, raw_out = self._pipe_stream("gbk")
        err, _raw_err = self._pipe_stream("gbk")
        with mock.patch.object(sys, "stdout", out), \
                mock.patch.object(sys, "stderr", err):
            _force_utf8_output()
            print("中文日志")
            out.flush()
        self.assertEqual(raw_out.getvalue(), "中文日志\n".encode("utf-8"))

    def test_tty_output_keeps_locale_encoding(self):
        out, raw = self._pipe_stream("gbk")
        out.isatty = lambda: True  # 交互式终端：GBK 才是对的
        with mock.patch.object(sys, "stdout", out), \
                mock.patch.object(sys, "stderr", out):
            _force_utf8_output()
            print("中文日志")
            out.flush()
        self.assertEqual(raw.getvalue(), "中文日志\n".encode("gbk"))

    def test_frozen_mode_self_invokes(self):
        exe = str(EXE_DIR / "dist" / "vidsum.exe")
        with _fake_frozen(exe):
            cmd = webapp._child_cmd("job.json")
        self.assertEqual(cmd, [exe, "--job-runner", "job.json"])


class EntryDispatchTest(unittest.TestCase):
    """entry.main 按首参数把活派给 webapp / _job_runner / cli。"""

    def _dispatch(self, argv):
        with mock.patch("vidsum.webapp.main", return_value=0) as w, \
                mock.patch("vidsum._job_runner.main", return_value=0) as j, \
                mock.patch("vidsum.cli.main", return_value=0) as c:
            entry.main(argv)
        return w, j, c

    def test_no_args_starts_web(self):
        w, j, c = self._dispatch([])
        w.assert_called_once_with([])
        j.assert_not_called()
        c.assert_not_called()

    def test_web_flag_passes_through_options(self):
        w, j, c = self._dispatch(["--web", "--port", "9000"])
        w.assert_called_once_with(["--port", "9000"])

    def test_job_runner_goes_to_child(self):
        w, j, c = self._dispatch(["--job-runner", "job.json"])
        j.assert_called_once_with(["job.json"])
        w.assert_not_called()
        c.assert_not_called()

    def test_plain_args_go_to_cli(self):
        w, j, c = self._dispatch(["lecture.mp4", "--asr", "api"])
        c.assert_called_once_with(["lecture.mp4", "--asr", "api"])
        w.assert_not_called()
        j.assert_not_called()


class _RecordingWhisperModel:
    """记录构造参数，不加载任何真实权重。"""

    last_kwargs = None

    def __init__(self, model_size, **kwargs):
        _RecordingWhisperModel.last_kwargs = dict(kwargs, model_size=model_size)


class FasterWhisperDeviceTest(unittest.TestCase):
    """CUDA 探测改走 ctranslate2（torch 不进 exe），默认模型目录跟 app_root 走。"""

    def setUp(self):
        fake_ct2 = types.ModuleType("ctranslate2")
        fake_ct2.get_cuda_device_count = lambda: 0
        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = _RecordingWhisperModel
        for name, mod in (("ctranslate2", fake_ct2),
                          ("faster_whisper", fake_fw)):
            patcher = mock.patch.dict(sys.modules, {name: mod})
            patcher.start()
            self.addCleanup(patcher.stop)
        _RecordingWhisperModel.last_kwargs = None

    def _backend(self, **kwargs):
        return FasterWhisperBackend(**kwargs)

    def test_auto_device_cpu_without_cuda(self):
        backend = self._backend(device="auto")
        self.assertEqual(backend.device, "cpu")
        # CPU 上 float16 不被 ctranslate2 支持，自动降级 int8
        self.assertEqual(_RecordingWhisperModel.last_kwargs["compute_type"],
                         "int8")

    def test_auto_device_cuda_when_available(self):
        fake_ct2 = types.ModuleType("ctranslate2")
        fake_ct2.get_cuda_device_count = lambda: 1
        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
            backend = self._backend(device="auto", compute_type="float16")
        self.assertEqual(backend.device, "cuda")
        self.assertEqual(_RecordingWhisperModel.last_kwargs["compute_type"],
                         "float16")

    def test_default_download_root_follows_app_root(self):
        with _fake_frozen(str(EXE_DIR / "dist" / "vidsum.exe")):
            self._backend()
        expected = str(EXE_DIR / "dist" / "models" / "whisper")
        self.assertEqual(_RecordingWhisperModel.last_kwargs["download_root"],
                         expected)

    def test_explicit_download_root_wins(self):
        self._backend(download_root=r"D:\whisper-models")
        self.assertEqual(_RecordingWhisperModel.last_kwargs["download_root"],
                         r"D:\whisper-models")


if __name__ == "__main__":
    unittest.main()
