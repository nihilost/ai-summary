"""Web 控制台的后端行为：参数映射、任务生命周期、状态快照与 HTTP 接口。

不起真实转写：子进程一律打桩。HTTP 层用裸 socket 发请求，
避免客户端先把 "/../x" 归一化掉，目录穿越防护就测不到了。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import ThreadingHTTPServer
from pathlib import Path

from vidsum import webapp
from tests.support import TempDirCase


class _FakePopen:
    """假的子进程：stdout 逐行流出，可设退出码，也可挂住等人终止。"""

    def __init__(self, lines=(), returncode=0, hang=False, hold=None):
        self._lines = list(lines)
        self.returncode = returncode
        self._hang = hang
        self._hold = hold
        self.terminated = False
        self.stdout = self

    def __iter__(self):  # 父进程那边是 for line in proc.stdout
        if self._hold is not None:
            self._hold.wait(timeout=10)
        if self._hang:
            while not self.terminated:
                time.sleep(0.01)
            return
        yield from self._lines

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return None if (self._hang and not self.terminated) else self.returncode

    def terminate(self):
        self.terminated = True
        if self._hang:
            self.returncode = 1


class BuildArgsTest(unittest.TestCase):
    def test_empty_params_keep_parser_defaults(self):
        args = webapp._build_args({})
        self.assertEqual(args.input, "")
        self.assertEqual(args.asr, "faster-whisper")
        self.assertEqual(args.chunk_minutes, 10.0)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.no_summary)

    def test_params_map_to_args(self):
        args = webapp._build_args({
            "input": "  lecture.mp4  ",
            "output": "D:/out",
            "chunk_minutes": "2.5",
            "workers": "0",            # 下限钳到 1
            "language": "en",
            "audio_backend": "pyav",
            "asr": "sherpa",
            "model_size": "small",
            "sherpa_model_dir": "D:/models",
            "sherpa_threads": "0",     # 下限钳到 1
            "no_silence_split": True,
            "keep_audio": True,
            "keep_chunks": True,
            "no_boundary_dedup": True,
            "no_punctuation": True,
            "no_summary": True,
            "force": True,
            "dry_run": True,
            "llm_base_url": "https://llm.example/v1",
            "llm_api_key": "sk-x",
            "llm_model": "gpt-4o-mini",
            "llm_max_tokens": "1000",
        })
        self.assertEqual(args.input, "lecture.mp4")
        self.assertEqual(args.output, str(Path("D:/out").resolve()))
        self.assertEqual(args.chunk_minutes, 2.5)
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.language, "en")
        self.assertEqual(args.audio_backend, "pyav")
        self.assertEqual(args.asr, "sherpa")
        self.assertEqual(args.sherpa_model_dir, "D:/models")
        self.assertEqual(args.sherpa_threads, 1)
        for key in ("no_silence_split", "keep_audio", "keep_chunks",
                    "no_boundary_dedup", "no_punctuation", "no_summary",
                    "force", "dry_run"):
            self.assertTrue(getattr(args, key), key)
        self.assertEqual(args.llm_base_url, "https://llm.example/v1")
        self.assertEqual(args.llm_api_key, "sk-x")
        self.assertEqual(args.llm_model, "gpt-4o-mini")
        self.assertEqual(args.llm_max_tokens, 1000)

    def test_blank_optional_fields_are_skipped(self):
        args = webapp._build_args({
            "input": "x.mp4",
            "output": "",
            "sherpa_model_dir": "  ",
            "llm_base_url": "",
            "llm_api_key": "",
            "llm_model": "",
        })
        self.assertIsNone(args.output)
        self.assertIsNone(args.sherpa_model_dir)
        # argparse 里这两个参数没有默认值；openai 兜底发生在 resolve_config
        self.assertIsNone(args.llm_base_url)
        self.assertIsNone(args.llm_api_key)
        self.assertIsNone(args.llm_model)

    def test_bad_numbers_fall_back(self):
        args = webapp._build_args({"input": "x.mp4", "chunk_minutes": "abc",
                                   "workers": None, "llm_max_tokens": "x"})
        self.assertEqual(args.chunk_minutes, 10.0)
        self.assertEqual(args.llm_max_tokens, 2048)

    def test_asr_api_and_network_params_map(self):
        args = webapp._build_args({
            "input": "https://example.com/x.m3u8",
            "asr": "api",
            "asr_base_url": "https://asr.example/v1",
            "asr_api_key": "sk-asr",
            "asr_model": "whisper-large-v3",
            "user_agent": "Mozilla/5.0",
            "referer": "https://ref.example/",
            "cookie": "SESS=abc",
        })
        self.assertEqual(args.asr, "api")
        self.assertEqual(args.asr_base_url, "https://asr.example/v1")
        self.assertEqual(args.asr_api_key, "sk-asr")
        self.assertEqual(args.asr_model, "whisper-large-v3")
        self.assertEqual(args.user_agent, "Mozilla/5.0")
        self.assertEqual(args.referer, "https://ref.example/")
        self.assertEqual(args.cookie, "SESS=abc")

    def test_asr_api_and_network_blanks_keep_defaults(self):
        args = webapp._build_args({
            "input": "x.mp4",
            "asr_base_url": "",
            "asr_api_key": "",
            "asr_model": "",
            "user_agent": "",
            "referer": "  ",
            "cookie": "",
        })
        # 未提供时保留 argparse 默认：asr_base_url/model 有默认，header 类为 None
        self.assertEqual(args.asr_base_url, "https://api.openai.com/v1")
        self.assertEqual(args.asr_model, "whisper-1")
        self.assertIsNone(args.user_agent)
        self.assertIsNone(args.referer)
        self.assertIsNone(args.cookie)

    def test_frame_params_map(self):
        args = webapp._build_args({
            "input": "x.mp4",
            "frames_per_segment": "5",
            "frame_width": "640",
            "no_frames": True,
            "vision": True,
        })
        self.assertEqual(args.frames_per_segment, 5)
        self.assertEqual(args.frame_width, 640)
        self.assertTrue(args.no_frames)
        self.assertTrue(args.vision)

    def test_frame_params_defaults_and_clamps(self):
        args = webapp._build_args({"input": "x.mp4"})
        self.assertEqual(args.frames_per_segment, 2)
        self.assertEqual(args.frame_width, 480)
        self.assertFalse(args.no_frames)
        self.assertFalse(args.vision)
        clamped = webapp._build_args({"input": "x.mp4",
                                      "frames_per_segment": "-1",
                                      "frame_width": "10"})
        self.assertEqual(clamped.frames_per_segment, 0)
        self.assertEqual(clamped.frame_width, 160)

    def test_whisper_download_root_maps(self):
        args = webapp._build_args({
            "input": "x.mp4", "asr": "faster-whisper",
            "whisper_download_root": "  E:\\whisper-models  ",
        })
        self.assertEqual(args.whisper_download_root, "E:\\whisper-models")

    def test_whisper_download_root_blank_keeps_default(self):
        args = webapp._build_args({"input": "x.mp4", "whisper_download_root": "  "})
        self.assertIsNone(args.whisper_download_root)


class StartRunTest(TempDirCase):
    def setUp(self):
        self.addCleanup(setattr, webapp, "_current", None)
        webapp._current = None

    def _start(self, params, fake):
        params = dict(params)
        params.setdefault("output", str(self.make_dir()))
        # 任务在后台线程里跑，patch 必须活到任务结束，不能只用 with 块
        patcher = mock.patch.object(webapp.subprocess, "Popen",
                                    lambda *a, **kw: fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return webapp.start_run(params)

    def _wait_state(self, job, want, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline and job.state == "running":
            time.sleep(0.01)
        self.assertEqual(job.state, want)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            webapp.start_run({"input": "  "})

    def test_bad_chunk_minutes_rejected(self):
        with self.assertRaises(ValueError):
            webapp.start_run({"input": "x.mp4", "chunk_minutes": "0"})

    def test_job_completes_and_logs_flow_in(self):
        lines = ["[00:00:01] [音频] 后端=ffmpeg", "[00:00:02] [转写] 开始"]
        job = self._start({"input": "lecture.mp4"}, _FakePopen(lines))
        self.assertEqual(job.work.name, "lecture")
        self._wait_state(job, "done")
        logs = job.logs()
        self.assertTrue(any("[音频]" in ln for ln in logs))
        self.assertTrue(any("[转写]" in ln for ln in logs))

    def test_failure_marks_error(self):
        proc = _FakePopen(["[00:00:01] 错误：模型没加载"], returncode=1)
        job = self._start({"input": "x.mp4"}, proc)
        self._wait_state(job, "error")
        self.assertEqual(job.error, "子进程退出码 1")
        self.assertTrue(any("错误：模型没加载" in ln for ln in job.logs()))

    def test_cancel_marks_stopped(self):
        proc = _FakePopen(hang=True)
        job = self._start({"input": "x.mp4"}, proc)
        deadline = time.time() + 5
        while time.time() < deadline and job.proc is None:
            time.sleep(0.01)
        webapp.stop_run()
        self._wait_state(job, "stopped")
        self.assertTrue(proc.terminated)
        self.assertTrue(job.cancel.is_set())

    def test_second_run_rejected_while_running(self):
        job = self._start({"input": "x.mp4"}, _FakePopen(hang=True))
        try:
            with self.assertRaises(RuntimeError) as ctx:
                webapp.start_run({"input": "y.mp4", "output": job.work.parent})
            self.assertIn("已有任务在运行", str(ctx.exception))
        finally:
            webapp.stop_run()
        self._wait_state(job, "stopped")

    def test_stop_without_job_is_noop(self):
        webapp.stop_run()  # 不炸即可


class ChildProcessEncodingTest(TempDirCase):
    """子进程日志编码：中文 Windows 的 GBK locale 不能把日志搅成乱码。"""

    def test_child_env_forces_utf8(self):
        env = webapp._child_env()
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")

    def test_runner_output_decodes_as_utf8(self):
        fd, json_path = tempfile.mkstemp(prefix="vidsum-enc-", suffix=".json")
        os.close(fd)
        payload = {"input": "不存在的视频文件.mp4", "asr": "dummy",
                   "output": str(self.make_dir())}
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "vidsum._job_runner", json_path],
                cwd=str(webapp.PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=webapp._child_env(),
            )
            out, _ = proc.communicate(timeout=120)
        finally:
            os.unlink(json_path)
        # 输入文件不存在，子进程必然非零退出；关键是中文一行不能变乱码
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("\ufffd", out)
        self.assertIn("错误：", out)


class StatusTest(TempDirCase):
    def setUp(self):
        self.addCleanup(setattr, webapp, "_current", None)
        webapp._current = None

    def test_idle_status_is_minimal(self):
        self.assertEqual(webapp._status(), {"state": "idle"})

    def test_status_reads_disk_artifacts(self):
        work = self.make_dir()
        (work / "chunks").mkdir()
        (work / "plan.json").write_text(json.dumps({
            "segments": [{"index": 0, "start": 0, "end": 60},
                         {"index": 1, "start": 60, "end": 120}]}),
            encoding="utf-8")
        (work / "summary_state.json").write_text(
            json.dumps({"next_index": 1, "running": "摘要", "notes": []}),
            encoding="utf-8")
        (work / "transcript.md").write_text("# t\n\n**[0:00:01]** 你好", encoding="utf-8")
        for name in ("chunks/000.json", "chunks/001.json"):
            (work / name).write_text("{}", encoding="utf-8")

        job = webapp.Job({"input": "x.mp4", "asr": "sherpa"}, work)
        webapp._current = job
        status = webapp._status()

        self.assertEqual(status["state"], "running")
        self.assertEqual(status["asr"], "sherpa")
        self.assertEqual(status["progress"],
                         {"total": 2, "done": 2, "summarized": 1})
        self.assertEqual([f["name"] for f in status["files"]],
                         ["chunks\\000.json", "chunks\\001.json", "plan.json",
                          "summary_state.json", "transcript.md"])
        self.assertIn("你好", status["transcript"])
        self.assertEqual(status["summary"], "")


def _http(port, method, path, body=None):
    """裸 socket 发一个请求，返回 (status, headers, body_bytes)。"""
    payload = b"" if body is None else json.dumps(body).encode("utf-8")
    headers = {"Host": "127.0.0.1", "Connection": "close"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(payload))
    head = [f"{method} {path} HTTP/1.1"]
    head += [f"{k}: {v}" for k, v in headers.items()]
    request = "\r\n".join(head) + "\r\n\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request.encode("ascii") + payload)
        chunks = []
        while True:
            block = sock.recv(65536)
            if not block:
                break
            chunks.append(block)
    raw = b"".join(chunks)
    head, _, body_bytes = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    fields = dict(
        line.split(": ", 1) for line in lines[1:] if ": " in line
    )
    return status, fields, body_bytes


class HttpApiTest(TempDirCase):
    def setUp(self):
        self.addCleanup(setattr, webapp, "_current", None)
        webapp._current = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        # addCleanup 是 LIFO：注册反过来，执行顺序才是 停服务→关 socket→join
        self.addCleanup(self.thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_static_files_served(self):
        status, fields, body = _http(self.port, "GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", fields["Content-Type"])
        self.assertIn("vidsum", body.decode("utf-8"))

        status, fields, body = _http(self.port, "GET", "/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", fields["Content-Type"])
        self.assertIn("renderMarkdown", body.decode("utf-8"))

        status, fields, _ = _http(self.port, "GET", "/styles.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", fields["Content-Type"])

    def test_asr_local_and_cloud_are_parallel_tracks(self):
        # 转写方式把「本地」和「云端 API」摆成两条平行轨道，各带自己的配置
        status, _, body = _http(self.port, "GET", "/")
        self.assertEqual(status, 200)
        html = body.decode("utf-8")
        self.assertIn('id="asr-track"', html)
        self.assertIn('data-track="local"', html)
        self.assertIn('data-track="cloud"', html)
        self.assertIn('id="track-local"', html)
        self.assertIn('id="track-cloud"', html)
        self.assertIn('id="asr-local"', html)
        # 旧的单一下拉已经移除
        self.assertNotIn('id="asr"', html)

        status, _, body = _http(self.port, "GET", "/app.js")
        self.assertEqual(status, 200)
        js = body.decode("utf-8")
        self.assertIn("currentTrack", js)
        self.assertIn("setTrack", js)
        self.assertIn('$("asr-local").value', js)
        self.assertIn('"api"', js)

    def test_traversal_outside_web_dir_blocked(self):
        status, _, _ = _http(self.port, "GET", "/../webapp.py")
        self.assertEqual(status, 403)

    def _current_job_with_frame(self):
        work = self.make_dir()
        frame = work / "frames" / "000" / "0125.500.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        webapp._current = webapp.Job({"input": "x.mp4"}, work)
        return work, frame

    def test_work_file_served_for_current_job(self):
        _work, frame = self._current_job_with_frame()
        status, fields, body = _http(
            self.port, "GET", "/work/frames/000/0125.500.jpg")
        self.assertEqual(status, 200)
        self.assertIn("image/jpeg", fields["Content-Type"])
        self.assertEqual(body, frame.read_bytes())

    def test_work_file_without_job_is_404(self):
        status, _, _ = _http(self.port, "GET", "/work/frames/000/a.jpg")
        self.assertEqual(status, 404)

    def test_work_missing_file_is_404(self):
        self._current_job_with_frame()
        status, _, _ = _http(self.port, "GET", "/work/frames/000/nope.jpg")
        self.assertEqual(status, 404)

    def test_work_traversal_outside_work_dir_blocked(self):
        work = self.make_dir()
        outside = self.make_dir()
        secret = outside / "secret.txt"
        secret.write_text("secret", encoding="utf-8")
        webapp._current = webapp.Job({"input": "x.mp4"}, work)
        rel = f"../{secret.relative_to(work.parent).as_posix()}"
        status, _, _ = _http(self.port, "GET", f"/work/{rel}")
        self.assertEqual(status, 403)

    def test_unknown_static_path_redirects_home(self):
        status, fields, _ = _http(self.port, "GET", "/nope.js")
        self.assertEqual(status, 302)
        self.assertEqual(fields["Location"], "/")

    def test_pasted_garbage_path_redirects_home(self):
        # 地址栏里粘进整句中文之类的场景
        # 浏览器实际发出去的是百分号编码形式
        status, fields, _ = _http(
            self.port, "GET", "/%E5%B0%B1%E8%83%BD%E8%BF%9B%E6%8E%A7%E5%88%B6%E4%BA%86%EF%BC%8CB")
        self.assertEqual(status, 302)
        self.assertEqual(fields["Location"], "/")

    def test_idle_status_endpoint(self):
        status, _, body = _http(self.port, "GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode("utf-8")), {"state": "idle"})

    def test_run_rejects_empty_input(self):
        status, _, body = _http(self.port, "POST", "/api/run", {"input": ""})
        self.assertEqual(status, 400)
        self.assertFalse(json.loads(body.decode("utf-8"))["ok"])

    def test_stop_without_running_job_is_400(self):
        status, _, body = _http(self.port, "POST", "/api/stop", {})
        self.assertEqual(status, 400)
        self.assertIn("没有正在运行的任务", body.decode("utf-8"))

    def test_unknown_api_path_is_404(self):
        status, _, _ = _http(self.port, "POST", "/api/nope", {})
        self.assertEqual(status, 404)

    def test_unknown_api_get_is_404(self):
        status, _, _ = _http(self.port, "GET", "/api/nope")
        self.assertEqual(status, 404)

    def test_run_then_status_then_stop_flow(self):
        out = str(self.make_dir())
        hold = threading.Event()
        proc = _FakePopen(["[00:00:01] [音频] 后端=ffmpeg"], hold=hold)
        with mock.patch.object(webapp.subprocess, "Popen",
                               lambda *a, **kw: proc):
            status, _, body = _http(self.port, "POST", "/api/run",
                                    {"input": "lecture.mp4", "output": out})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body.decode("utf-8"))["work"],
                             str(Path(out) / "lecture"))

            status, _, body = _http(self.port, "GET", "/api/status")
            live = json.loads(body.decode("utf-8"))
            self.assertEqual(live["state"], "running")
            self.assertEqual(live["progress"]["total"], 0)

            # 运行中再提交要被拒绝
            status, _, body = _http(self.port, "POST", "/api/run",
                                    {"input": "y.mp4", "output": out})
            self.assertEqual(status, 400)
            self.assertIn("已有任务在运行", body.decode("utf-8"))

            hold.set()
            job = webapp._current
            deadline = time.time() + 10
            while time.time() < deadline and job.state == "running":
                time.sleep(0.01)
            self.assertEqual(job.state, "done")

        # 任务结束后停止应当被拒
        status, _, _ = _http(self.port, "POST", "/api/stop", {})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
