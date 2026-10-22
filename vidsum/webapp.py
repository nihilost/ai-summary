"""vidsum 的本地 Web 控制台。

把命令行流水线包进一个标准库 http.server：任意时刻只有一个任务在跑，
前端轮询 /api/status 拿实时日志与产物（transcript/summary/final 都是边跑边写盘的，
因此轮询即“流式”观感）。不引第三方依赖，testenv 里直接 `python -m vidsum.webapp` 即可。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from . import pipeline
from .cli import build_parser
from .util import app_root, human_size, read_json

WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECT_ROOT = app_root()
DEFAULT_OUTPUT = PROJECT_ROOT / "vidsum-out"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class Job:
    """一次运行的全部可变状态。日志由子进程 stdout 逐行汇进来。"""

    def __init__(self, params: dict, work: Path):
        self.params = params
        self.work = work
        self.state = "running"          # running / done / error / stopped
        self.started = time.time()
        self.ended: float | None = None
        self.error = ""
        self.cancel = threading.Event()
        self.proc: subprocess.Popen | None = None
        self._logs: list[str] = []
        self._lock = threading.Lock()

    def log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)
            if len(self._logs) > 4000:
                del self._logs[:-1500]

    def logs(self) -> list[str]:
        with self._lock:
            return list(self._logs)


_current: Job | None = None
guard = threading.Lock()


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _list_files(work: Path) -> list[dict]:
    root = Path(work)
    if not root.is_dir():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            files.append({"name": str(path.relative_to(root)),
                          "size": size, "pretty": human_size(size)})
    return files


def _progress(work: Path) -> dict:
    plan = read_json(Path(work) / "plan.json") or {}
    total = len(plan.get("segments", []))
    chunks_dir = Path(work) / "chunks"
    done = len(list(chunks_dir.glob("*.json"))) if chunks_dir.is_dir() else 0
    state = read_json(Path(work) / "summary_state.json") or {}
    return {"total": total, "done": min(done, total) if total else done,
            "summarized": int(state.get("next_index", 0))}


def _status() -> dict:
    job = _current
    if job is None:
        return {"state": "idle"}
    return {
        "state": job.state,
        "input": job.params.get("input", ""),
        "asr": job.params.get("asr", ""),
        "work": str(job.work),
        "started": job.started,
        "elapsed": (job.ended or time.time()) - job.started,
        "error": job.error,
        "logs": job.logs()[-500:],
        "progress": _progress(job.work),
        "files": _list_files(job.work),
        "transcript": _read_text(Path(job.work) / "transcript.md"),
        "summary": _read_text(Path(job.work) / "summary.md"),
        "final": _read_text(Path(job.work) / "final.md"),
    }


def _build_args(params: dict):
    args = build_parser().parse_args([""])   # 拿一套默认值
    args.input = str(params.get("input") or "").strip()
    if params.get("output"):
        # 绝对化：子进程以项目根为 cwd 启动，相对路径必须在父进程就解析好
        args.output = str(Path(params["output"]).expanduser().resolve())
    args.chunk_minutes = _num(params.get("chunk_minutes"), args.chunk_minutes)
    args.workers = max(1, int(_num(params.get("workers"), args.workers)))
    args.language = str(params.get("language") or args.language)
    args.audio_backend = str(params.get("audio_backend") or args.audio_backend)
    if str(params.get("audio_codec") or "").strip():
        args.audio_codec = str(params["audio_codec"]).strip()
    args.asr = str(params.get("asr") or args.asr)
    args.model_size = str(params.get("model_size") or args.model_size)
    if str(params.get("sherpa_model_dir") or "").strip():
        args.sherpa_model_dir = str(params["sherpa_model_dir"]).strip()
    if str(params.get("whisper_download_root") or "").strip():
        args.whisper_download_root = str(params["whisper_download_root"]).strip()
    args.sherpa_threads = max(1, int(_num(params.get("sherpa_threads"),
                                          args.sherpa_threads)))
    args.frames_per_segment = max(0, int(_num(
        params.get("frames_per_segment"), args.frames_per_segment)))
    args.frame_width = max(160, int(_num(params.get("frame_width"),
                                         args.frame_width)))
    for key in ("no_silence_split", "keep_audio", "keep_chunks",
                "no_boundary_dedup", "no_punctuation", "no_summary",
               "force", "dry_run", "no_frames", "vision"):
        if key in params:
            setattr(args, key, bool(params[key]))
    for key in ("llm_base_url", "llm_api_key", "llm_model"):
        if params.get(key):
            setattr(args, key, str(params[key]))
    if "llm_max_tokens" in params:
        args.llm_max_tokens = int(_num(params.get("llm_max_tokens"),
                                       args.llm_max_tokens))
    # 纯空白视为未填：strip 后为空则保留 argparse 默认，与 sherpa_model_dir 一致
    for key in ("asr_base_url", "asr_api_key", "asr_model",
                "user_agent", "referer", "cookie"):
        value = str(params.get(key) or "").strip()
        if value:
            setattr(args, key, value)
    return args


def start_run(params: dict) -> Job:
    global _current
    source = str(params.get("input") or "").strip()
    if not source:
        raise ValueError("请填写视频路径或直链")
    chunk_minutes = _num(params.get("chunk_minutes"), 10.0)
    if chunk_minutes <= 0:
        raise ValueError("每段时长必须为正数")
    with guard:
        if _current is not None and _current.state == "running":
            raise RuntimeError("已有任务在运行，先停止或等它结束")
        out_root = Path(params.get("output") or DEFAULT_OUTPUT).expanduser()
        work = out_root / pipeline._stem_for(source)
        job = Job(params, work)
        _current = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def _jsonable(data: dict) -> dict:
    """只保留可 JSON 序列化的基础类型，argparse 产出的值都在此列。"""
    return {k: v for k, v in data.items()
            if v is None or isinstance(v, (str, int, float, bool))}


def _child_env() -> dict:
    """子进程的环境变量。

    强制 PYTHONIOENCODING=utf-8：Windows 上管道 stdout 默认跟随 locale 编码
    （中文系统是 GBK），父进程按 UTF-8 读会把中文日志搅成乱码。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def _child_cmd(json_path: str) -> list:
    """子进程命令行：开发态走 `python -m`，exe 冻结后自调用。

    PyInstaller 单文件模式下 sys.executable 就是 vidsum.exe 本身，
    没有解释器可依赖；exe 内建的 --job-runner 分支由 vidsum.entry 分发。
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--job-runner", json_path]
    return [sys.executable, "-m", "vidsum._job_runner", json_path]


def _run_job(job: Job) -> None:
    """把任务丢进独立子进程：卡死时 terminate() 能连 CUDA 初始化一起杀掉。"""
    args = _build_args(job.params)
    fd, json_path = tempfile.mkstemp(prefix="vidsum-job-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(vars(args)), fh, ensure_ascii=False)
        proc = subprocess.Popen(
            _child_cmd(json_path),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_child_env(),
        )
        job.proc = proc
        if job.cancel.is_set():  # 启动与停止抢跑：直接别跑
            proc.terminate()
        for line in proc.stdout or []:
            job.log(line.rstrip("\r\n"))
        code = proc.wait()
        if job.cancel.is_set():
            job.state = "stopped"
        elif code == 0:
            job.state = "done"
        else:
            job.state = "error"
            job.error = f"子进程退出码 {code}"
    except Exception as exc:  # 统一收口，异常翻成界面可见的一行
        job.state = "stopped" if job.cancel.is_set() else "error"
        job.error = str(exc)
        job.log(f"错误：{exc}")
    finally:
        job.ended = time.time()
        try:
            os.unlink(json_path)
        except OSError:
            pass


def stop_run() -> None:
    with guard:
        job = _current
    if job is None or job.state != "running":
        return
    job.cancel.set()
    proc = job.proc
    if proc is not None and proc.poll() is None:
        # Windows 上 terminate 即 TerminateProcess，卡死的 CUDA 进程也杀得掉
        try:
            proc.terminate()
        except OSError:
            pass


def _kill_orphan_runners() -> None:
    """清掉上一次控制台遗留的任务子进程（只有本模块会拉起它们）。

    开发态子进程是 `python.exe -m vidsum._job_runner`，exe 冻结后是
    `vidsum.exe --job-runner`，按当前自己的形态匹配才对得上。
    """
    if os.name != "nt":
        return
    exe_name = Path(sys.executable).name
    marker = ("--job-runner" if getattr(sys, "frozen", False)
              else "vidsum._job_runner")
    script = (f"Get-CimInstance Win32_Process -Filter \"Name='{exe_name}'\" "
              f"| Where-Object {{ $_.CommandLine -like '*{marker}*' }} "
              "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       capture_output=True, timeout=30)
    except Exception:
        pass  # 清理失败不碍事：顶多下次手动杀


class Handler(BaseHTTPRequestHandler):
    server_version = "vidsum-web/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 静音默认的逐请求日志
        pass

    def _send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect_home(self) -> None:
        self.send_response(302)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            target, ctype = WEB_DIR / "index.html", "text/html; charset=utf-8"
        else:
            target = WEB_DIR / path.lstrip("/")
            ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        if not target.is_file():
            # 地址栏粘错路径之类：拉回首页，别甩一脸 JSON
            self._redirect_home()
            return
        # 防目录穿越：解析后必须仍在 WEB_DIR 内
        try:
            target.resolve().relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, 403)
            return
        self._send_bytes(target.read_bytes(), ctype)

    def _serve_work_file(self, rel: str) -> None:
        """服务当前任务工作目录里的文件（终稿插图等网页内可见的产物）。"""
        with guard:
            job = _current
        if job is None:
            self._send_json({"error": "没有正在显示的任务"}, 404)
            return
        root = Path(job.work).resolve()
        try:
            target = (root / unquote(rel)).resolve()
            target.relative_to(root)  # 防目录穿越
        except (ValueError, OSError):
            self._send_json({"error": "forbidden"}, 403)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), ctype)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send_json(_status())
        elif path.startswith("/api/"):
            self._send_json({"error": "not found"}, 404)
        elif path.startswith("/work/"):
            self._serve_work_file(path[len("/work/"):])
        else:
            self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            params = {}
        if path == "/api/run":
            try:
                job = start_run(params)
                self._send_json({"ok": True, "work": str(job.work)})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 400)
        elif path == "/api/stop":
            with guard:
                running = _current is not None and _current.state == "running"
            if not running:
                self._send_json({"ok": False, "error": "没有正在运行的任务"}, 400)
            else:
                stop_run()
                self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="vidsum-web",
                                     description="vidsum 本地 Web 控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    _kill_orphan_runners()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"vidsum 控制台已启动： http://{args.host}:{args.port}/  （Ctrl+C 停止）",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
